from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, ConfigDict, Field, model_validator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
    IMAGEIO_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    imageio = None  # type: ignore[assignment]
    IMAGEIO_AVAILABLE = False
    IMAGEIO_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


START_TIME = datetime(2026, 10, 31, 8, 0, 0, tzinfo=timezone.utc)

DEVICE_IDS = [f"sensor-{i:02d}" for i in range(1, 13)]
STATUS_VALUES = ["OK", "WARN", "ERROR", "MAINT"]

EXPECTED_COLUMNS = [
    "event_id",
    "device_id",
    "event_time",
    "ingested_at",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "battery_pct",
    "status",
]

NUMERIC_RANGES = {
    "temperature_c": (-40.0, 85.0),
    "humidity_pct": (0.0, 100.0),
    "pressure_hpa": (870.0, 1085.0),
    "battery_pct": (0.0, 100.0),
}

FAULT_KEYS = [
    "missing",
    "outlier",
    "duplicate",
    "stale",
    "schema_drift",
    "volume_spike",
    "missing_column",
]

COMPOSITE_FIELDS = [
    "completeness",
    "validity",
    "uniqueness",
    "consistency",
    "freshness",
]

METRIC_FIELDS = COMPOSITE_FIELDS + [
    "rows",
    "throughput_rows_per_sec",
]

RESET_KEYS = (
    "step",
    "history",
    "alerts",
    "latest_alerts",
    "faults",
    "fault_active",
    "recovery",
    "alert_state",
    "playing",
    "speed",
    "latest_df",
    "latest_manifest",
    "latest_window_end",
)

Severity = Literal["info", "warn", "critical"]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for attr in ("window_id", "rows", "composite", "event", "exc"):
            value = getattr(record, attr, None)
            if value is not None:
                payload[attr] = value

        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("dqms")
    logger.setLevel(os.getenv("DQMS_LOG_LEVEL", "INFO").upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = configure_logging()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def short_hash(value: str, length: int = 12) -> str:
    return value[:length] if value else ""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fault_widget_key(fault_key: str) -> str:
    return f"fault_checkbox_{fault_key}"


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert only real numeric-like values to float.

    This prevents Pydantic methods, enums, lists, or unexpected objects from
    being passed to float() and crashing the alert engine.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return 1.0 if value else 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    app_name: str = "Real-Time Data Quality Monitor"
    artifact_version: str = "0.1.1"
    schema_version: str = "1.0"
    random_seed: int = 1337

    window_seconds: int = Field(default=5, ge=1, le=3600)
    base_rows: int = Field(default=80, ge=1, le=5000)
    max_rows: int = Field(default=250, ge=1, le=20000)

    late_threshold_seconds: int = Field(default=15, ge=1)
    ewma_alpha: float = Field(default=0.25, gt=0.0, le=1.0)
    z_alert: float = Field(default=3.0, gt=0.0)

    video_seconds: int = Field(default=6, ge=3, le=60)
    video_fps: int = Field(default=8, ge=1, le=30)
    history_limit: int = Field(default=180, ge=10, le=5000)

    @model_validator(mode="after")
    def check_row_bounds(self) -> "AppSettings":
        if self.max_rows < self.base_rows:
            raise ValueError("max_rows must be >= base_rows")
        return self

    @classmethod
    def load(cls) -> "AppSettings":
        overrides: dict[str, Any] = {}
        for name in cls.model_fields:
            env_key = f"DQMS_{name.upper()}"
            if env_key in os.environ:
                overrides[name] = os.environ[env_key]

        try:
            return cls(**overrides)
        except Exception as exc:
            LOGGER.warning(
                "Invalid DQMS settings; using defaults.",
                extra={"event": "config", "exc": f"{type(exc).__name__}: {exc}"},
            )
            return cls()

    def config_hash(self) -> str:
        return sha256_hex(self.model_dump_json().encode("utf-8"))


class ErrorCode(StrEnum):
    SCHEMA_MISSING_FIELD = "DQ-E001"
    SCHEMA_UNKNOWN_FIELD = "DQ-E002"
    NULL_SPIKE = "DQ-E003"
    OUT_OF_RANGE = "DQ-E004"
    DUPLICATE_KEY = "DQ-E005"
    FRESHNESS_LAG = "DQ-E006"
    VOLUME_ANOMALY = "DQ-E007"
    DISTRIBUTION_DRIFT = "DQ-E008"
    VIDEO_GENERATION_FAILED = "DQ-E009"


class BatchManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    source_id: str
    window_id: str
    row_count: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    created_at: datetime
    config_hash: str


class QualitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    window_id: str
    timestamp: datetime

    rows: int = Field(ge=0)
    completeness: float = Field(ge=0.0, le=1.0)
    validity: float = Field(ge=0.0, le=1.0)
    uniqueness: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    freshness: float = Field(ge=0.0, le=1.0)

    throughput_rows_per_sec: float = Field(ge=0.0)
    peak_memory_mb: float = Field(ge=0.0)

    schema_drift: bool
    error_codes: list[ErrorCode] = Field(default_factory=list)
    notes: str = ""


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: datetime
    severity: Severity
    code: ErrorCode
    title: str
    detail: str
    metric: str
    value: float
    baseline: float | None = None
    z_score: float | None = None
    acknowledged: bool = False
    suppressed: bool = False


def default_faults() -> dict[str, bool]:
    return {key: False for key in FAULT_KEYS}


def scheduled_faults(step: int) -> dict[str, bool]:
    """
    Preloaded demo schedule.

    Default generated demo is 6 seconds at 8 fps = 48 windows.
    This schedule shows all major fault types and leaves a short clean tail.
    """
    flags = default_faults()

    if 6 <= step < 12:
        flags["missing"] = True
    elif 12 <= step < 18:
        flags["outlier"] = True
    elif 18 <= step < 23:
        flags["duplicate"] = True
    elif 23 <= step < 29:
        flags["stale"] = True
    elif 29 <= step < 35:
        flags["schema_drift"] = True
    elif 35 <= step < 41:
        flags["volume_spike"] = True
    elif 41 <= step < 46:
        flags["missing_column"] = True

    return flags


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(
        index=False,
        float_format="%.6f",
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
    ).encode("utf-8")


def generate_batch(
    step: int,
    settings: AppSettings,
    faults: dict[str, bool],
) -> tuple[pd.DataFrame, BatchManifest, bytes, datetime]:
    rng = np.random.default_rng(settings.random_seed + step * 7919)

    window_start = START_TIME + timedelta(seconds=step * settings.window_seconds)
    window_end = window_start + timedelta(seconds=settings.window_seconds)

    n = settings.base_rows + int(rng.normal(0, 4))
    if faults.get("volume_spike"):
        n += int(rng.integers(90, 170))
    n = int(np.clip(n, 1, settings.max_rows))

    device_ids = rng.choice(DEVICE_IDS, size=n, replace=True)
    offsets = rng.uniform(0.0, float(settings.window_seconds), size=n)

    event_times = pd.Timestamp(window_start) + pd.to_timedelta(offsets, unit="s")
    ingest_lag = rng.normal(1.5, 0.6, size=n).clip(0.1, 8.0)

    if faults.get("stale"):
        ingest_lag = ingest_lag + rng.uniform(25.0, 90.0, size=n)

    ingested_times = pd.Timestamp(window_start) + pd.to_timedelta(
        offsets + ingest_lag,
        unit="s",
    )

    temperature = 22.0 + 3.0 * np.sin(step / 12.0) + rng.normal(0.0, 0.8, n)
    humidity = 45.0 + 8.0 * np.cos(step / 18.0) + rng.normal(0.0, 2.5, n)
    pressure = 1013.0 + rng.normal(0.0, 1.2, n)
    battery = np.clip(82.0 - 0.03 * step + rng.normal(0.0, 2.0, n), 0.0, 100.0)

    status_probs = (0.82, 0.10, 0.05, 0.03)
    if faults.get("outlier"):
        status_probs = (0.55, 0.18, 0.20, 0.07)

    status = rng.choice(STATUS_VALUES, size=n, p=status_probs)
    event_ids = np.array([f"evt-{step:06d}-{i:06d}" for i in range(n)], dtype=object)

    df = pd.DataFrame(
        {
            "event_id": event_ids,
            "device_id": device_ids,
            "event_time": event_times,
            "ingested_at": ingested_times,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "pressure_hpa": pressure,
            "battery_pct": battery,
            "status": status,
        }
    )

    # Mild cross-field inconsistency for consistency monitoring.
    if "battery_pct" in df.columns and "status" in df.columns:
        inconsistent_mask = (df["battery_pct"] < 12.0) & (rng.random(len(df)) < 0.35)
        df.loc[inconsistent_mask, "status"] = "OK"

    if faults.get("missing"):
        miss_idx = rng.random(len(df)) < 0.28
        for col in ["temperature_c", "humidity_pct", "pressure_hpa"]:
            if col in df.columns:
                df.loc[miss_idx, col] = np.nan

    if faults.get("outlier"):
        out_size = max(1, len(df) // 8)
        out_idx = rng.choice(len(df), size=out_size, replace=False)

        df.loc[out_idx, "temperature_c"] += rng.normal(28.0, 6.0, size=out_size)
        df.loc[out_idx, "pressure_hpa"] += rng.normal(90.0, 25.0, size=out_size)
        df.loc[out_idx, "humidity_pct"] = np.clip(
            df.loc[out_idx, "humidity_pct"] + rng.normal(35.0, 8.0, size=out_size),
            0.0,
            130.0,
        )

    if faults.get("duplicate"):
        dup_size = min(12, len(df))
        dup_idx = rng.choice(len(df), size=dup_size, replace=False)
        duplicates = df.iloc[dup_idx].copy()
        df = pd.concat([df, duplicates], ignore_index=True)

    if faults.get("schema_drift"):
        df["firmware_v2"] = rng.choice(["alpha", "beta"], size=len(df))

    if faults.get("missing_column") and "pressure_hpa" in df.columns:
        df = df.drop(columns=["pressure_hpa"])

    if len(df) > settings.max_rows:
        df = df.iloc[: settings.max_rows].reset_index(drop=True)

    csv_bytes = dataframe_to_csv_bytes(df)

    manifest = BatchManifest(
        schema_version=settings.schema_version,
        source_id="local-file-drop-demo",
        window_id=f"w-{step:06d}",
        row_count=len(df),
        sha256=sha256_hex(csv_bytes),
        created_at=window_end,
        config_hash=settings.config_hash(),
    )

    return df, manifest, csv_bytes, window_end


def evaluate_batch(
    df: pd.DataFrame,
    manifest: BatchManifest,
    settings: AppSettings,
    process_seconds: float,
    window_timestamp: datetime,
) -> QualitySnapshot:
    rows = len(df)

    missing_expected = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    unknown_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    if rows == 0:
        completeness = 0.0
        validity = 0.0
        uniqueness = 0.0
        consistency = 0.0
        freshness = 0.0
    else:
        nonnull = 0
        for col in EXPECTED_COLUMNS:
            if col in df.columns:
                nonnull += int(df[col].notna().sum())
        completeness = nonnull / float(rows * len(EXPECTED_COLUMNS))

        valid_num = 0
        valid_den = 0

        for col, (lo, hi) in NUMERIC_RANGES.items():
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                valid_den += int(len(series))
                valid_num += int(((series >= lo) & (series <= hi)).sum())

        if "status" in df.columns:
            series = df["status"].dropna().astype(str)
            valid_den += int(len(series))
            valid_num += int(series.isin(STATUS_VALUES).sum())

        if "event_time" in df.columns and "ingested_at" in df.columns:
            event = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
            ingested = pd.to_datetime(df["ingested_at"], errors="coerce", utc=True)
            pair = pd.DataFrame({"event": event, "ingested": ingested}).dropna()
            valid_den += int(len(pair))
            valid_num += int((pair["event"] <= pair["ingested"]).sum())

        validity = valid_num / float(valid_den) if valid_den else 0.0

        if "event_id" in df.columns:
            unique_count = int(df["event_id"].nunique(dropna=True))
            uniqueness = unique_count / float(rows)
        else:
            uniqueness = 0.0

        consistency_mask = pd.Series(True, index=df.index)

        if "battery_pct" in df.columns and "status" in df.columns:
            battery = pd.to_numeric(df["battery_pct"], errors="coerce")
            consistency_mask &= ~((battery < 12.0) & (df["status"].astype(str) == "OK"))

        if "temperature_c" in df.columns and "humidity_pct" in df.columns:
            temp = pd.to_numeric(df["temperature_c"], errors="coerce")
            hum = pd.to_numeric(df["humidity_pct"], errors="coerce")
            consistency_mask &= ~((temp > 65.0) & (hum > 90.0))

        consistency = float(consistency_mask.mean())

        if "event_time" in df.columns and "ingested_at" in df.columns:
            event = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
            ingested = pd.to_datetime(df["ingested_at"], errors="coerce", utc=True)
            pair = pd.DataFrame({"event": event, "ingested": ingested}).dropna()
            if len(pair):
                lags = (pair["ingested"] - pair["event"]).dt.total_seconds()
                freshness = float(
                    (lags.between(0.0, float(settings.late_threshold_seconds))).mean()
                )
            else:
                freshness = 0.0
        else:
            freshness = 0.0

    throughput = float(rows) / max(1e-6, float(process_seconds))
    peak_memory_mb = float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)

    error_codes: list[ErrorCode] = []

    if missing_expected:
        error_codes.append(ErrorCode.SCHEMA_MISSING_FIELD)
    if unknown_columns:
        error_codes.append(ErrorCode.SCHEMA_UNKNOWN_FIELD)
    if rows == 0 or completeness < 0.95:
        error_codes.append(ErrorCode.NULL_SPIKE)
    if validity < 0.95:
        error_codes.append(ErrorCode.OUT_OF_RANGE)
    if uniqueness < 0.99:
        error_codes.append(ErrorCode.DUPLICATE_KEY)
    if consistency < 0.95:
        error_codes.append(ErrorCode.DISTRIBUTION_DRIFT)
    if freshness < 0.90:
        error_codes.append(ErrorCode.FRESHNESS_LAG)

    notes = (
        f"missing={missing_expected}; "
        f"unknown={unknown_columns}; "
        f"rows={rows}"
    )

    return QualitySnapshot(
        schema_version=settings.schema_version,
        window_id=manifest.window_id,
        timestamp=window_timestamp,
        rows=rows,
        completeness=completeness,
        validity=validity,
        uniqueness=uniqueness,
        consistency=consistency,
        freshness=freshness,
        throughput_rows_per_sec=throughput,
        peak_memory_mb=peak_memory_mb,
        schema_drift=bool(missing_expected or unknown_columns),
        error_codes=error_codes,
        notes=notes,
    )


def composite_score(snapshot: QualitySnapshot) -> float:
    values = [safe_float(getattr(snapshot, field, 0.0)) for field in COMPOSITE_FIELDS]
    return float(np.mean(values)) if values else 0.0


def ewma_update(state: dict[str, Any], x: float, alpha: float) -> None:
    count = int(state.get("count", 0))
    if count == 0:
        state["mean"] = float(x)
        state["var"] = 0.0
    else:
        old_mean = float(state["mean"])
        diff = float(x) - old_mean
        state["mean"] = old_mean + alpha * diff
        state["var"] = (1.0 - alpha) * (float(state["var"]) + alpha * diff * diff)
    state["count"] = count + 1


def compute_baseline(
    history: Iterable[QualitySnapshot],
    alpha: float,
) -> dict[str, dict[str, Any]]:
    baseline: dict[str, dict[str, Any]] = {
        metric: {"mean": 0.0, "var": 0.0, "count": 0}
        for metric in METRIC_FIELDS
    }

    for snapshot in history:
        snapshot_data = snapshot.model_dump(mode="python")
        for metric in METRIC_FIELDS:
            value = safe_float(snapshot_data.get(metric), 0.0)
            ewma_update(baseline[metric], value, alpha)

    return baseline


RULE_METRIC = {
    ErrorCode.NULL_SPIKE: "completeness",
    ErrorCode.OUT_OF_RANGE: "validity",
    ErrorCode.DUPLICATE_KEY: "uniqueness",
    ErrorCode.FRESHNESS_LAG: "freshness",
    ErrorCode.DISTRIBUTION_DRIFT: "consistency",
    ErrorCode.SCHEMA_MISSING_FIELD: "schema_drift",
    ErrorCode.SCHEMA_UNKNOWN_FIELD: "schema_drift",
    ErrorCode.VOLUME_ANOMALY: "rows",
}

METRIC_ALERT_CODE = {
    "completeness": ErrorCode.NULL_SPIKE,
    "validity": ErrorCode.OUT_OF_RANGE,
    "uniqueness": ErrorCode.DUPLICATE_KEY,
    "consistency": ErrorCode.DISTRIBUTION_DRIFT,
    "freshness": ErrorCode.FRESHNESS_LAG,
    "rows": ErrorCode.VOLUME_ANOMALY,
    "throughput_rows_per_sec": ErrorCode.VOLUME_ANOMALY,
}


def build_alerts(
    snapshot: QualitySnapshot,
    baseline: dict[str, dict[str, Any]],
    settings: AppSettings,
    step: int,
    alert_state: dict[str, int],
) -> list[Alert]:
    alerts: list[Alert] = []
    added_keys: set[str] = set()
    cooldown_windows = 5

    snapshot_data = snapshot.model_dump(mode="python")

    def push(
        code: ErrorCode,
        metric: str,
        value: float,
        severity: Severity,
        title: str,
        detail: str,
        baseline_value: float | None = None,
        z: float | None = None,
    ) -> None:
        key = f"{code.value}:{metric}"
        if key in added_keys:
            return

        last_step = alert_state.get(key)
        if last_step is not None and step - int(last_step) < cooldown_windows:
            return

        alert_state[key] = step
        added_keys.add(key)

        alert_id = hashlib.sha256(
            f"{key}:{step}:{snapshot.window_id}".encode("utf-8")
        ).hexdigest()[:16]

        alerts.append(
            Alert(
                id=alert_id,
                created_at=snapshot.timestamp,
                severity=severity,
                code=code,
                title=title,
                detail=detail,
                metric=metric,
                value=float(value),
                baseline=None if baseline_value is None else float(baseline_value),
                z_score=None if z is None else float(z),
            )
        )

    # Deterministic rule-based alerts.
    for code in snapshot.error_codes:
        metric = RULE_METRIC.get(code, "quality")
        value = safe_float(snapshot_data.get(metric), 0.0)

        if code in {
            ErrorCode.SCHEMA_MISSING_FIELD,
            ErrorCode.SCHEMA_UNKNOWN_FIELD,
        }:
            severity: Severity = "critical"
        elif value < 0.80:
            severity = "critical"
        else:
            severity = "warn"

        push(
            code=code,
            metric=metric,
            value=value,
            severity=severity,
            title=code.name.replace("_", " ").title(),
            detail=(
                f"Window {snapshot.window_id} triggered {code.value}. "
                f"{metric}={value:.3f}. Notes: {snapshot.notes}"
            ),
        )

    # Statistical EWMA/z-score alerts for drift and volume anomalies.
    threshold = float(settings.z_alert)

    for metric in METRIC_FIELDS:
        state = baseline.get(metric)
        if not state or int(state.get("count", 0)) < 8:
            continue

        x = safe_float(snapshot_data.get(metric), 0.0)
        mean = safe_float(state.get("mean"), 0.0)
        var = safe_float(state.get("var"), 0.0)
        std = math.sqrt(max(var, 0.0))

        if std < 1e-9:
            continue

        z = (x - mean) / std

        if metric in {"rows", "throughput_rows_per_sec"}:
            if abs(z) <= threshold:
                continue
            code = METRIC_ALERT_CODE[metric]
            severity = "critical" if abs(z) > threshold * 1.5 else "warn"
        else:
            # For quality scores, only unusually low values are concerning.
            if z >= -threshold:
                continue
            code = METRIC_ALERT_CODE.get(metric, ErrorCode.DISTRIBUTION_DRIFT)
            severity = "critical" if z < -threshold * 1.5 else "warn"

        push(
            code=code,
            metric=metric,
            value=x,
            severity=severity,
            title=f"Statistical {metric.replace('_', ' ')} anomaly",
            detail=(
                f"{metric}={x:.4f}, EWMA baseline={mean:.4f}, "
                f"z={z:.2f}, threshold={threshold:.2f}"
            ),
            baseline_value=mean,
            z=z,
        )

    return alerts


def update_recovery(snapshot: QualitySnapshot, step: int) -> None:
    recovery = st.session_state.get("recovery")
    if not isinstance(recovery, dict):
        return

    if not recovery.get("pending", False):
        return

    score = composite_score(snapshot)

    if score >= 0.95:
        recovery["good_streak"] = int(recovery.get("good_streak", 0)) + 1
    else:
        recovery["good_streak"] = 0

    if int(recovery.get("good_streak", 0)) >= 2:
        recovery["pending"] = False
        recovery["total"] = int(recovery.get("total", 0)) + 1
        recovery["success"] = int(recovery.get("success", 0)) + 1
        recovery["last_result"] = "success"
    elif step - int(recovery.get("started_step", step)) > 10:
        recovery["pending"] = False
        recovery["total"] = int(recovery.get("total", 0)) + 1
        recovery["last_result"] = "failed"


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buffer = fig.canvas.buffer_rgba()
    array = np.asarray(buffer)
    rgb = np.ascontiguousarray(array[:, :, :3], dtype=np.uint8)
    return rgb


def make_video_frame(
    history: list[QualitySnapshot],
    end_index: int,
    total_frames: int,
    settings: AppSettings,
    fingerprint: str,
) -> np.ndarray:
    recent = history[max(0, end_index - 30) : end_index]
    xs = [s.timestamp for s in recent]

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.6), dpi=100)

    quality_ax = axes[0, 0]
    for metric in COMPOSITE_FIELDS:
        quality_ax.plot(
            xs,
            [getattr(s, metric) for s in recent],
            label=metric.replace("_", " "),
            linewidth=1.2,
        )
    quality_ax.set_ylim(0.0, 1.05)
    quality_ax.set_title("Quality scores", fontsize=8)
    quality_ax.legend(fontsize=5, ncol=2)
    quality_ax.tick_params(axis="x", labelsize=5, rotation=20)
    quality_ax.tick_params(axis="y", labelsize=5)

    volume_ax = axes[0, 1]
    volume_ax.plot(xs, [s.rows for s in recent], color="#2b7de9", linewidth=1.4)
    volume_ax.set_title("Rows per window", fontsize=8)
    volume_ax.set_ylabel("rows", fontsize=6)
    volume_ax.tick_params(axis="x", labelsize=5, rotation=20)
    volume_ax.tick_params(axis="y", labelsize=5)

    alert_ax = axes[1, 0]
    alert_counts = [len(s.error_codes) for s in recent]
    alert_ax.bar(range(len(recent)), alert_counts, color="#c0392b", width=0.8)
    alert_ax.set_title("Rule violations per window", fontsize=8)
    alert_ax.set_ylabel("count", fontsize=6)
    alert_ax.tick_params(axis="both", labelsize=5)

    bar_ax = axes[1, 1]
    latest = recent[-1]
    labels = [m.replace("_", "\n") for m in COMPOSITE_FIELDS]
    values = [getattr(latest, m) for m in COMPOSITE_FIELDS]
    colors = [
        "#27ae60" if v >= 0.95 else "#e67e22" if v >= 0.85 else "#c0392b"
        for v in values
    ]
    bar_ax.barh(labels, values, color=colors)
    bar_ax.set_xlim(0.0, 1.0)
    bar_ax.set_title("Latest window", fontsize=8)
    bar_ax.tick_params(axis="both", labelsize=5)
    bar_ax.invert_yaxis()

    fig.suptitle(
        f"Real-Time Data Quality Monitor | frame {end_index}/{total_frames} | "
        f"artifact {settings.artifact_version}",
        fontsize=9,
    )
    fig.text(
        0.01,
        0.01,
        f"synthetic demo • dataset sha256 {short_hash(fingerprint, 16)} • no API key",
        fontsize=5,
    )
    fig.tight_layout(rect=[0.0, 0.035, 1.0, 0.94])

    frame = figure_to_rgb(fig)
    plt.close(fig)
    return frame


def _normalize_frames_even(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return frames

    h, w, _ = frames[0].shape
    nh = h - (h % 2)
    nw = w - (w % 2)

    if (nh, nw) == (h, w):
        return frames

    return [np.ascontiguousarray(frame[:nh, :nw], dtype=np.uint8) for frame in frames]


def _write_mp4_imageio_attempt(
    frames: list[np.ndarray],
    fps: int,
    config: dict[str, Any],
) -> bytes:
    if not IMAGEIO_AVAILABLE or imageio is None:
        raise RuntimeError(
            f"imageio is unavailable: {IMAGEIO_IMPORT_ERROR or 'unknown error'}"
        )

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    path = Path(tmp_name)
    writer = None

    try:
        writer = imageio.get_writer(
            str(path),
            mode="I",
            fps=int(fps),
            **config,
        )

        for frame in frames:
            writer.append_data(frame)

        writer.close()
        writer = None

        size = os.path.getsize(path)
        if size <= 0:
            raise RuntimeError("imageio produced an empty MP4 file")

        return path.read_bytes()
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def _write_mp4_imageio(frames: list[np.ndarray], fps: int) -> bytes:
    if not IMAGEIO_AVAILABLE or imageio is None:
        raise RuntimeError(
            f"imageio is unavailable: {IMAGEIO_IMPORT_ERROR or 'unknown error'}"
        )

    configs: list[dict[str, Any]] = [
        {
            "codec": "libx264",
            "pixelformat": "yuv420p",
            "macro_block_size": 1,
            "output_params": ["-movflags", "+faststart"],
        },
        {
            "codec": "libx264",
            "pixelformat": "yuv420p",
            "macro_block_size": 1,
        },
        {
            "codec": "libx264",
            "macro_block_size": 1,
        },
        {
            "codec": "mpeg4",
            "pixelformat": "yuv420p",
            "macro_block_size": 1,
        },
        {
            "codec": "mpeg4",
            "macro_block_size": 1,
        },
        {
            "macro_block_size": 1,
        },
    ]

    errors: list[str] = []

    for config in configs:
        try:
            return _write_mp4_imageio_attempt(frames, fps, config)
        except Exception as exc:
            errors.append(f"{config.get('codec', 'default')}: {type(exc).__name__}: {exc}")

    raise RuntimeError(" | ".join(errors))


def _write_mp4_opencv(frames: list[np.ndarray], fps: int) -> bytes:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install optional MP4 backend with: "
            "pip install opencv-python-headless"
        ) from exc

    if not frames:
        raise RuntimeError("No frames supplied to OpenCV MP4 writer")

    h, w, _ = frames[0].shape
    candidates = ["avc1", "H264", "mp4v"]
    errors: list[str] = []

    for fourcc_name in candidates:
        fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        path = Path(tmp_name)
        writer = None

        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(
                str(path),
                fourcc,
                float(fps),
                (int(w), int(h)),
                True,
            )

            if not writer.isOpened():
                raise RuntimeError(f"cv2.VideoWriter could not open codec {fourcc_name}")

            for frame in frames:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
                writer.write(bgr)

            writer.release()
            writer = None

            size = os.path.getsize(path)
            if size <= 0:
                raise RuntimeError(f"cv2 produced an empty MP4 file for codec {fourcc_name}")

            return path.read_bytes()
        except Exception as exc:
            errors.append(f"{fourcc_name}: {type(exc).__name__}: {exc}")
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    raise RuntimeError(" | ".join(errors))


def _write_gif(frames: list[np.ndarray], fps: int) -> bytes:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"Pillow is unavailable: {type(exc).__name__}: {exc}") from exc

    if not frames:
        raise RuntimeError("No frames supplied to GIF writer")

    duration_ms = max(1, int(round(1000.0 / float(max(1, fps)))))
    images = [Image.fromarray(frame) for frame in frames]

    buf = BytesIO()
    images[0].save(
        buf,
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    data = buf.getvalue()
    if not data:
        raise RuntimeError("GIF writer produced empty bytes")

    return data


def generate_demo_video(
    frames: list[np.ndarray],
    fps: int,
) -> tuple[bytes | None, bytes | None, str | None, str | None]:
    frames = _normalize_frames_even(frames)

    mp4_bytes: bytes | None = None
    mp4_errors: list[str] = []

    try:
        mp4_bytes = _write_mp4_imageio(frames, fps)
    except Exception as exc:
        mp4_errors.append(f"imageio: {exc}")

    if mp4_bytes is None:
        try:
            mp4_bytes = _write_mp4_opencv(frames, fps)
        except Exception as exc:
            mp4_errors.append(f"opencv: {exc}")

    if mp4_bytes is not None:
        mp4_error: str | None = None
    else:
        mp4_error = " || ".join(mp4_errors) if mp4_errors else "No MP4 backend available."

    gif_bytes: bytes | None = None
    gif_error: str | None = None

    if mp4_bytes is None:
        try:
            gif_bytes = _write_gif(frames, fps)
        except Exception as exc:
            gif_error = f"{type(exc).__name__}: {exc}"

    return mp4_bytes, gif_bytes, mp4_error, gif_error


@st.cache_data(show_spinner="Generating live demo video and baseline history…")
def build_demo_artifacts(
    settings_json: str,
) -> tuple[
    bytes | None,
    bytes | None,
    str | None,
    str | None,
    list[dict[str, Any]],
    str,
    str,
]:
    settings = AppSettings.model_validate_json(settings_json)
    total_frames = max(36, settings.video_seconds * settings.video_fps)

    history: list[QualitySnapshot] = []
    hasher = hashlib.sha256()

    for step in range(total_frames):
        faults = scheduled_faults(step)

        t0 = time.perf_counter()
        df, manifest, csv_bytes, window_end = generate_batch(step, settings, faults)
        process_seconds = max(1e-6, time.perf_counter() - t0)

        hasher.update(csv_bytes)

        snapshot = evaluate_batch(
            df=df,
            manifest=manifest,
            settings=settings,
            process_seconds=process_seconds,
            window_timestamp=window_end,
        )
        history.append(snapshot)

    fingerprint = hasher.hexdigest()

    frames = [
        make_video_frame(
            history=history,
            end_index=i + 1,
            total_frames=total_frames,
            settings=settings,
            fingerprint=fingerprint,
        )
        for i in range(total_frames)
    ]

    mp4_bytes, gif_bytes, mp4_error, gif_error = generate_demo_video(
        frames=frames,
        fps=settings.video_fps,
    )

    history_dicts = [s.model_dump(mode="json") for s in history]

    return (
        mp4_bytes,
        gif_bytes,
        mp4_error,
        gif_error,
        history_dicts,
        fingerprint,
        settings.config_hash(),
    )


def initialize_session(settings: AppSettings, history_dicts: list[dict[str, Any]]) -> None:
    if "history" not in st.session_state:
        st.session_state.step = len(history_dicts)
        st.session_state.history = deque(
            (QualitySnapshot.model_validate(h) for h in history_dicts),
            maxlen=settings.history_limit,
        )
        st.session_state.alerts = deque(maxlen=120)
        st.session_state.latest_alerts = []
        st.session_state.faults = default_faults()
        st.session_state.fault_active = False
        st.session_state.recovery = {
            "pending": False,
            "started_step": 0,
            "good_streak": 0,
            "success": 0,
            "total": 0,
            "last_result": "none",
        }
        st.session_state.alert_state = {}
        st.session_state.playing = True
        st.session_state.speed = 2.0

        df, manifest, _, window_end = generate_batch(
            step=int(st.session_state.step),
            settings=settings,
            faults=dict(st.session_state.faults),
        )
        st.session_state.latest_df = df
        st.session_state.latest_manifest = manifest
        st.session_state.latest_window_end = window_end
    else:
        if not isinstance(st.session_state.history, deque):
            raw_history = st.session_state.history
            st.session_state.history = deque(
                (
                    QualitySnapshot.model_validate(item)
                    if isinstance(item, dict)
                    else item
                    for item in raw_history
                ),
                maxlen=settings.history_limit,
            )

        if "alerts" not in st.session_state or not isinstance(st.session_state.alerts, deque):
            existing_alerts = st.session_state.get("alerts", [])
            if existing_alerts is None:
                existing_alerts = []
            st.session_state.alerts = deque(existing_alerts, maxlen=120)

        st.session_state.setdefault("step", len(st.session_state.history))
        st.session_state.setdefault("latest_alerts", [])
        st.session_state.setdefault("faults", default_faults())
        st.session_state.setdefault("fault_active", False)
        st.session_state.setdefault(
            "recovery",
            {
                "pending": False,
                "started_step": 0,
                "good_streak": 0,
                "success": 0,
                "total": 0,
                "last_result": "none",
            },
        )
        st.session_state.setdefault("alert_state", {})
        st.session_state.setdefault("playing", True)
        st.session_state.setdefault("speed", 2.0)

    # These keys belong to widgets. This function is called before widgets are
    # instantiated in main(), and reset_live_state() also calls it before widgets.
    st.session_state.setdefault("speed_slider", float(st.session_state.get("speed", 2.0)))
    st.session_state.setdefault("alert_ack_select", [])

    faults_for_defaults = st.session_state.get("faults")
    if not isinstance(faults_for_defaults, dict):
        faults_for_defaults = default_faults()

    for fault_key in FAULT_KEYS:
        st.session_state.setdefault(
            fault_widget_key(fault_key),
            bool(faults_for_defaults.get(fault_key, False)),
        )

    if (
        "latest_df" not in st.session_state
        or "latest_manifest" not in st.session_state
        or "latest_window_end" not in st.session_state
    ):
        step_for_latest = int(st.session_state.get("step", 0))
        df, manifest, _, window_end = generate_batch(
            step=step_for_latest,
            settings=settings,
            faults=dict(faults_for_defaults),
        )
        st.session_state.latest_df = df
        st.session_state.latest_manifest = manifest
        st.session_state.latest_window_end = window_end


def reset_live_state(settings: AppSettings, history_dicts: list[dict[str, Any]]) -> None:
    # Safe only when called before keyed widgets are instantiated in the script run.
    for key in RESET_KEYS:
        st.session_state.pop(key, None)

    for fault_key in FAULT_KEYS:
        st.session_state.pop(fault_widget_key(fault_key), None)

    st.session_state.pop("speed_slider", None)
    st.session_state.pop("alert_ack_select", None)

    initialize_session(settings, history_dicts)


def advance_step(settings: AppSettings) -> None:
    step = int(st.session_state.get("step", 0))

    faults_value = st.session_state.get("faults")
    faults = dict(faults_value) if isinstance(faults_value, dict) else default_faults()

    t0 = time.perf_counter()
    df, manifest, _, window_end = generate_batch(step, settings, faults)
    process_seconds = max(1e-6, time.perf_counter() - t0)

    snapshot = evaluate_batch(
        df=df,
        manifest=manifest,
        settings=settings,
        process_seconds=process_seconds,
        window_timestamp=window_end,
    )

    history = st.session_state.get("history")
    if not isinstance(history, deque):
        existing_history = history if history is not None else []
        history = deque(
            (
                QualitySnapshot.model_validate(item)
                if isinstance(item, dict)
                else item
                for item in existing_history
            ),
            maxlen=settings.history_limit,
        )
        st.session_state.history = history

    history.append(snapshot)

    baseline = compute_baseline(history, settings.ewma_alpha)

    alert_state = st.session_state.get("alert_state")
    if not isinstance(alert_state, dict):
        alert_state = {}
        st.session_state.alert_state = alert_state

    new_alerts = build_alerts(
        snapshot=snapshot,
        baseline=baseline,
        settings=settings,
        step=step,
        alert_state=alert_state,
    )

    alerts = st.session_state.get("alerts")
    if not isinstance(alerts, deque):
        existing_alerts = alerts if alerts is not None else []
        alerts = deque(existing_alerts, maxlen=120)
        st.session_state.alerts = alerts

    for alert in new_alerts:
        alerts.append(alert)

    st.session_state.latest_alerts = new_alerts

    update_recovery(snapshot, step)

    st.session_state.latest_df = df
    st.session_state.latest_manifest = manifest
    st.session_state.latest_window_end = window_end
    st.session_state.step = step + 1

    LOGGER.info(
        "Window processed.",
        extra={
            "window_id": manifest.window_id,
            "rows": snapshot.rows,
            "composite": composite_score(snapshot),
        },
    )


def render_top_metrics(
    latest: QualitySnapshot,
    recovery: dict[str, Any],
    playing: bool,
    step: int,
) -> None:
    cols = st.columns(8)
    cols[0].metric("Status", "Running" if playing else "Paused", delta=f"window {step}")
    cols[1].metric("Completeness", f"{latest.completeness:.1%}")
    cols[2].metric("Validity", f"{latest.validity:.1%}")
    cols[3].metric("Uniqueness", f"{latest.uniqueness:.1%}")
    cols[4].metric("Freshness", f"{latest.freshness:.1%}")
    cols[5].metric("Rows", f"{latest.rows:,}")
    cols[6].metric("Rows/s", f"{latest.throughput_rows_per_sec:.1f}")
    cols[7].metric("Schema drift", "Yes" if latest.schema_drift else "No")

    c1, c2, c3 = st.columns(3)
    c1.metric("Composite score", f"{composite_score(latest):.1%}")
    c2.metric("Recovery state", str(recovery.get("last_result", "none")))

    total = int(recovery.get("total", 0))
    success = int(recovery.get("success", 0))
    rate = f"{success / total:.0%}" if total else "n/a"
    c3.metric("Recovery success", rate)


def _render_gif_fallback(gif_bytes: bytes, mp4_error: str | None) -> None:
    st.info(
        "MP4 encoder unavailable on this machine. "
        "Showing animated GIF fallback so the demo is still live and moving."
    )

    try:
        st.image(gif_bytes, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not embed GIF preview: {exc}")

    st.download_button(
        "Download animated GIF demo",
        data=gif_bytes,
        file_name="dqms_live_demo.gif",
        mime="image/gif",
    )

    if mp4_error:
        with st.expander("MP4 encoder details"):
            st.code(mp4_error, language="text")
            st.caption(
                "To enable MP4, install one of these optional backends and then click "
                "Clear cache & reset demo: pip install imageio imageio-ffmpeg "
                "or pip install opencv-python-headless"
            )


def render_video(
    mp4_bytes: bytes | None,
    gif_bytes: bytes | None,
    mp4_error: str | None,
    gif_error: str | None,
) -> None:
    st.subheader("Auto-generated live demo video")

    if mp4_bytes:
        embedded = False
        try:
            st.video(
                mp4_bytes,
                format="video/mp4",
                loop=True,
                autoplay=True,
                muted=True,
            )
            embedded = True
        except Exception as exc:
            st.warning(f"MP4 bytes were generated, but Streamlit could not embed them: {exc}")

        st.download_button(
            "Download MP4 demo video",
            data=mp4_bytes,
            file_name="dqms_live_demo.mp4",
            mime="video/mp4",
        )

        if not embedded and gif_bytes:
            _render_gif_fallback(gif_bytes, mp4_error=None)

        return

    if gif_bytes:
        _render_gif_fallback(gif_bytes, mp4_error)
        return

    st.warning(
        "Neither MP4 nor GIF demo generation succeeded on this machine. "
        "The live dashboard below still runs and updates."
    )

    if mp4_error:
        with st.expander("MP4 encoder details"):
            st.code(mp4_error, language="text")

    if gif_error:
        with st.expander("GIF fallback details"):
            st.code(gif_error, language="text")

    st.caption(
        "Try installing an optional encoder backend, then click Clear cache & reset demo: "
        "pip install imageio imageio-ffmpeg or pip install opencv-python-headless"
    )


def render_charts(history: deque[QualitySnapshot]) -> None:
    recent = list(history)[-80:]

    if len(recent) < 2:
        st.info("Collecting at least two windows for charts…")
        return

    df = pd.DataFrame([s.model_dump(mode="json") for s in recent])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    for metric in COMPOSITE_FIELDS:
        ax.plot(df["timestamp"], df[metric], label=metric.replace("_", " "), linewidth=1.6)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Rolling data-quality scores")
    ax.legend(fontsize=7, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7.8, 3.0))
    ax1.plot(df["timestamp"], df["rows"], color="#2b7de9", label="rows/window")
    ax1.set_ylabel("rows/window")
    ax1.set_title("Volume and throughput")

    ax2 = ax1.twinx()
    ax2.plot(
        df["timestamp"],
        df["throughput_rows_per_sec"],
        color="#e67e22",
        label="rows/s",
    )
    ax2.set_ylabel("rows/s")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def build_change_set(
    latest: QualitySnapshot,
    latest_df: pd.DataFrame,
    latest_manifest: BatchManifest,
    current_alerts: Iterable[Alert],
    settings: AppSettings,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    alert_list = list(current_alerts)
    codes = set(latest.error_codes) | {alert.code for alert in alert_list}

    proposed: list[dict[str, Any]] = []

    if ErrorCode.DUPLICATE_KEY in codes:
        dup_count = (
            int(latest_df.duplicated(subset=["event_id"], keep=False).sum())
            if "event_id" in latest_df.columns
            else 0
        )
        proposed.append(
            {
                "action": "deduplicate_event_id",
                "mode": "dry_run",
                "destructive": False,
                "evidence": {"duplicate_rows_detected": dup_count},
                "derivative_path": f"artifacts/derivatives/{latest_manifest.window_id}/deduped.parquet",
                "undo_path": "delete derivative; raw input remains immutable",
            }
        )

    if ErrorCode.NULL_SPIKE in codes:
        proposed.append(
            {
                "action": "flag_missing_values",
                "mode": "dry_run",
                "destructive": False,
                "evidence": {"completeness": latest.completeness},
                "derivative_path": f"artifacts/derivatives/{latest_manifest.window_id}/missing_flags.parquet",
                "undo_path": "delete derivative; raw input remains immutable",
            }
        )

    if ErrorCode.OUT_OF_RANGE in codes:
        proposed.append(
            {
                "action": "quarantine_invalid_rows",
                "mode": "dry_run",
                "destructive": False,
                "evidence": {"validity": latest.validity},
                "derivative_path": f"artifacts/derivatives/{latest_manifest.window_id}/invalid_quarantine.parquet",
                "undo_path": "delete derivative; raw input remains immutable",
            }
        )

    if ErrorCode.SCHEMA_MISSING_FIELD in codes or ErrorCode.SCHEMA_UNKNOWN_FIELD in codes:
        proposed.append(
            {
                "action": "schema_review_change_set",
                "mode": "dry_run",
                "destructive": False,
                "evidence": {"notes": latest.notes},
                "derivative_path": f"artifacts/derivatives/{latest_manifest.window_id}/schema_review.json",
                "undo_path": "reject proposed schema rule change; no raw mutation",
            }
        )

    if ErrorCode.FRESHNESS_LAG in codes:
        proposed.append(
            {
                "action": "review_ingestion_latency",
                "mode": "dry_run",
                "destructive": False,
                "evidence": {
                    "freshness": latest.freshness,
                    "late_threshold_seconds": settings.late_threshold_seconds,
                },
                "derivative_path": f"artifacts/derivatives/{latest_manifest.window_id}/freshness_review.json",
                "undo_path": "no data mutation; review only",
            }
        )

    if ErrorCode.VOLUME_ANOMALY in codes:
        proposed.append(
            {
                "action": "review_volume_anomaly",
                "mode": "dry_run",
                "destructive": False,
                "evidence": {
                    "rows": latest.rows,
                    "throughput_rows_per_sec": latest.throughput_rows_per_sec,
                },
                "derivative_path": f"artifacts/derivatives/{latest_manifest.window_id}/volume_review.json",
                "undo_path": "no data mutation; review only",
            }
        )

    if ErrorCode.DISTRIBUTION_DRIFT in codes:
        proposed.append(
            {
                "action": "review_distribution_or_consistency_shift",
                "mode": "dry_run",
                "destructive": False,
                "evidence": {"consistency": latest.consistency},
                "derivative_path": f"artifacts/derivatives/{latest_manifest.window_id}/distribution_review.json",
                "undo_path": "no data mutation; review only",
            }
        )

    confidence = round(composite_score(latest), 3)

    return {
        "schema_version": settings.schema_version,
        "artifact_version": settings.artifact_version,
        "generated_at": utcnow().isoformat(),
        "applied": False,
        "human_approval_required": True,
        "input_window_id": latest_manifest.window_id,
        "input_sha256": latest_manifest.sha256,
        "dataset_sha256": dataset_fingerprint,
        "confidence": confidence,
        "evidence": [
            {
                "window_id": latest_manifest.window_id,
                "metrics": {
                    "completeness": latest.completeness,
                    "validity": latest.validity,
                    "uniqueness": latest.uniqueness,
                    "consistency": latest.consistency,
                    "freshness": latest.freshness,
                    "rows": latest.rows,
                    "throughput_rows_per_sec": latest.throughput_rows_per_sec,
                },
                "alert_codes": sorted(code.value for code in codes),
                "notes": latest.notes,
            }
        ],
        "proposed_changes": proposed,
        "rollback_policy": "Raw inputs are read-only and fingerprinted. All repairs are new derivatives.",
    }


def render_alerts(
    alerts: deque[Alert],
    current_alerts: list[Alert],
    latest: QualitySnapshot,
    latest_df: pd.DataFrame,
    latest_manifest: BatchManifest,
    recovery: dict[str, Any],
    settings: AppSettings,
    dataset_fingerprint: str,
) -> None:
    st.subheader("Alert manager")

    open_alerts = sum(1 for a in alerts if not a.acknowledged)
    critical_unacked = sum(
        1 for a in alerts if a.severity == "critical" and not a.acknowledged
    )

    total = int(recovery.get("total", 0))
    success = int(recovery.get("success", 0))
    recovery_rate = f"{success / total:.0%}" if total else "n/a"

    m1, m2, m3 = st.columns(3)
    m1.metric("Open alerts", open_alerts)
    m2.metric("Critical unacknowledged", critical_unacked)
    m3.metric("Recovery success", recovery_rate)

    if not alerts:
        # Safe to reset this widget key because the multiselect widget has not
        # been instantiated yet in this script run.
        st.session_state["alert_ack_select"] = []
        st.success("No alerts in the current alert window.")
        return

    alert_rows = []
    for alert in list(alerts)[::-1]:
        alert_rows.append(
            {
                "id": alert.id,
                "created_at": alert.created_at.isoformat(),
                "severity": alert.severity,
                "code": alert.code.value,
                "metric": alert.metric,
                "value": round(alert.value, 4),
                "baseline": None if alert.baseline is None else round(alert.baseline, 4),
                "z": None if alert.z_score is None else round(alert.z_score, 2),
                "acknowledged": alert.acknowledged,
                "title": alert.title,
                "detail": alert.detail,
            }
        )

    alert_df = pd.DataFrame(alert_rows)
    st.dataframe(alert_df, use_container_width=True, hide_index=True)

    option_ids = alert_df["id"].tolist()
    valid_ids = set(option_ids)

    # Sanitize stale selections before instantiating the multiselect widget.
    existing_selection = st.session_state.get("alert_ack_select", [])
    if isinstance(existing_selection, list):
        cleaned_selection = [alert_id for alert_id in existing_selection if alert_id in valid_ids]
        if cleaned_selection != existing_selection:
            st.session_state["alert_ack_select"] = cleaned_selection
    else:
        st.session_state["alert_ack_select"] = []

    selected = st.multiselect(
        "Select alerts to acknowledge",
        options=option_ids,
        key="alert_ack_select",
    )

    selected_valid = [alert_id for alert_id in selected if alert_id in valid_ids]

    if st.button("Acknowledge selected alerts"):
        selected_set = set(selected_valid)
        changed = 0
        for alert in alerts:
            if alert.id in selected_set and not alert.acknowledged:
                alert.acknowledged = True
                changed += 1
        st.success(f"Acknowledged {changed} alert(s).")
        st.rerun()

    st.divider()
    st.markdown("#### Human review gate")

    unacked_critical = [
        alert for alert in alerts if alert.severity == "critical" and not alert.acknowledged
    ]

    if unacked_critical:
        st.warning(
            "Critical alerts must be acknowledged before drafting a reviewable change-set. "
            "This prototype never auto-applies destructive repairs."
        )
        can_draft = False
    else:
        st.success(
            "No unacknowledged critical alerts. You may draft a non-destructive, "
            "reviewable change-set."
        )
        can_draft = True

    if can_draft and st.button("Draft reviewable change-set (dry-run)"):
        change_set = build_change_set(
            latest=latest,
            latest_df=latest_df,
            latest_manifest=latest_manifest,
            current_alerts=current_alerts,
            settings=settings,
            dataset_fingerprint=dataset_fingerprint,
        )
        st.json(change_set)
        st.download_button(
            "Download change-set JSON",
            data=json.dumps(change_set, indent=2, default=str).encode("utf-8"),
            file_name="dqms_change_set.json",
            mime="application/json",
        )


def render_raw_sample(
    df: pd.DataFrame,
    manifest: BatchManifest,
    dataset_fingerprint: str,
) -> None:
    st.subheader("Latest immutable evidence")

    left, right = st.columns([2, 1])

    with left:
        st.dataframe(df.head(25), use_container_width=True, hide_index=True)

    with right:
        st.json(manifest.model_dump(mode="json"))

    st.caption(
        f"Batch SHA-256: `{manifest.sha256}` • "
        f"Demo dataset SHA-256: `{dataset_fingerprint}`"
    )

    csv_bytes = dataframe_to_csv_bytes(df)
    st.download_button(
        "Download latest batch CSV sample",
        data=csv_bytes,
        file_name=f"{manifest.window_id}.csv",
        mime="text/csv",
    )


def render_config_lineage(
    settings: AppSettings,
    config_hash: str,
    dataset_fingerprint: str,
) -> None:
    st.subheader("Configuration and lineage")

    payload = {
        "app_name": settings.app_name,
        "artifact_version": settings.artifact_version,
        "schema_version": settings.schema_version,
        "random_seed": settings.random_seed,
        "window_seconds": settings.window_seconds,
        "base_rows": settings.base_rows,
        "max_rows": settings.max_rows,
        "late_threshold_seconds": settings.late_threshold_seconds,
        "ewma_alpha": settings.ewma_alpha,
        "z_alert": settings.z_alert,
        "video_seconds": settings.video_seconds,
        "video_fps": settings.video_fps,
        "history_limit": settings.history_limit,
        "config_hash": config_hash,
        "dataset_sha256": dataset_fingerprint,
        "expected_columns": EXPECTED_COLUMNS,
        "fault_keys": FAULT_KEYS,
        "imageio_available": IMAGEIO_AVAILABLE,
        "imageio_import_error": IMAGEIO_IMPORT_ERROR or None,
        "notes": [
            "Synthetic data only.",
            "No personal data.",
            "No paid API keys.",
            "Raw inputs are fingerprinted and treated as read-only evidence.",
            "Change-sets are dry-run only and require human review.",
            "MP4 generation tries imageio-ffmpeg, then OpenCV, then GIF fallback.",
        ],
    }

    st.json(payload)


def main() -> None:
    st.set_page_config(
        page_title="Real-Time Data Quality Monitor",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    settings = AppSettings.load()

    (
        mp4_bytes,
        gif_bytes,
        mp4_error,
        gif_error,
        history_dicts,
        dataset_fingerprint,
        config_hash,
    ) = build_demo_artifacts(settings.model_dump_json())

    initialize_session(settings, history_dicts)

    st.sidebar.title(settings.app_name)
    st.sidebar.caption(
        f"Artifact {settings.artifact_version} • "
        f"config {short_hash(config_hash)} • "
        f"data {short_hash(dataset_fingerprint)}"
    )

    # Cache/reset must happen before keyed widgets are instantiated.
    if st.sidebar.button("Clear cache & reset demo", use_container_width=True):
        st.cache_data.clear()
        reset_live_state(settings, history_dicts)
        st.rerun()

    if st.sidebar.button("Reset live state", use_container_width=True):
        reset_live_state(settings, history_dicts)
        st.rerun()

    playing = bool(st.session_state.get("playing", True))
    if st.sidebar.button(
        "Pause live simulation" if playing else "Resume live simulation",
        use_container_width=True,
    ):
        st.session_state.playing = not playing
        playing = bool(st.session_state.playing)

    speed = float(
        st.sidebar.slider(
            "Simulation speed (windows/sec)",
            min_value=0.5,
            max_value=8.0,
            step=0.5,
            key="speed_slider",
        )
    )
    st.session_state.speed = speed

    st.sidebar.header("Fault injection")
    st.sidebar.caption("Toggle faults to observe detector and alert behaviour.")

    current_faults: dict[str, bool] = {}
    for fault_key in FAULT_KEYS:
        label = fault_key.replace("_", " ").title()
        checked = bool(
            st.sidebar.checkbox(
                label,
                key=fault_widget_key(fault_key),
            )
        )
        current_faults[fault_key] = checked

    recovery = st.session_state.get("recovery")
    if not isinstance(recovery, dict):
        recovery = {
            "pending": False,
            "started_step": 0,
            "good_streak": 0,
            "success": 0,
            "total": 0,
            "last_result": "none",
        }
        st.session_state.recovery = recovery

    previous_active = bool(st.session_state.get("fault_active", False))
    current_any = any(current_faults.values())

    if current_any:
        if not previous_active:
            st.session_state.fault_active = True
            recovery["pending"] = False
        recovery["last_result"] = "fault-active"
    elif previous_active and not current_any:
        st.session_state.fault_active = False
        recovery.update(
            {
                "pending": True,
                "started_step": int(st.session_state.get("step", 0)),
                "good_streak": 0,
                "last_result": "pending",
            }
        )

    st.session_state.faults = current_faults

    st.sidebar.metric("Next window", int(st.session_state.get("step", 0)))
    st.sidebar.metric("Active faults", sum(current_faults.values()))

    if bool(st.session_state.get("playing", True)):
        advance_step(settings)

    history = st.session_state.get("history")
    if not isinstance(history, deque) or not history:
        st.error("History is empty. This should not happen in the demo.")
        st.stop()

    latest = history[-1]

    current_alerts = st.session_state.get("latest_alerts", [])
    if not isinstance(current_alerts, list):
        current_alerts = list(current_alerts)

    st.title("Real-Time Data Quality Monitoring System")
    st.caption(
        "CPU-only prototype • synthetic evidence • Streamlit UI • "
        "auto-generated demo video/GIF • no API keys • no destructive auto-repairs"
    )

    render_top_metrics(
        latest=latest,
        recovery=st.session_state.recovery,
        playing=bool(st.session_state.get("playing", True)),
        step=int(st.session_state.get("step", 0)),
    )

    render_video(mp4_bytes, gif_bytes, mp4_error, gif_error)

    tab_live, tab_alerts, tab_raw, tab_config = st.tabs(
        ["Live quality", "Alerts and review", "Raw evidence", "Config and lineage"]
    )

    with tab_live:
        render_charts(history)

    with tab_alerts:
        render_alerts(
            alerts=st.session_state.alerts,
            current_alerts=current_alerts,
            latest=latest,
            latest_df=st.session_state.latest_df,
            latest_manifest=st.session_state.latest_manifest,
            recovery=st.session_state.recovery,
            settings=settings,
            dataset_fingerprint=dataset_fingerprint,
        )

    with tab_raw:
        render_raw_sample(
            df=st.session_state.latest_df,
            manifest=st.session_state.latest_manifest,
            dataset_fingerprint=dataset_fingerprint,
        )

    with tab_config:
        render_config_lineage(
            settings=settings,
            config_hash=config_hash,
            dataset_fingerprint=dataset_fingerprint,
        )

    if bool(st.session_state.get("playing", True)):
        sleep_seconds = max(0.05, 1.0 / float(st.session_state.get("speed", 2.0)))
        time.sleep(sleep_seconds)
        st.rerun()


if __name__ == "__main__":
    main()