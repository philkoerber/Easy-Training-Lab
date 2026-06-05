#!/usr/bin/env python3
"""
Starter feature set used by both local training and QuantConnect inference.

The pandas path builds the training CSV. The streaming path is rendered into
tmp/qc_package/feature_starter_set.py so QC can compute the same columns live.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FEATURE_SET = "starter_set"
FEATURE_SET_VERSION = "2"

# Output column order (after timestamp). First = target for trainer.
OUTPUT_COLS = [
    "close",
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "log_return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "bar_range_pct",
    "relative_volume",
    "body_ratio",
    "rsi_14",
    "macd_line_pct",
    "macd_signal_pct",
]

MODEL_INPUT_COLS = [c for c in OUTPUT_COLS if c != "close"]

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ROLLING_VOL_WINDOWS = (5, 20)
RETURN_LAGS = 5
RELATIVE_VOLUME_WINDOW = 20

# First valid row is after 20 log returns. That requires 21 raw bars.
MIN_WARMUP_BARS = max(max(ROLLING_VOL_WINDOWS), RELATIVE_VOLUME_WINDOW)
QC_MODULE_NAME = "feature_starter_set.py"


def _parse_timestamp(series):
    """Parse timestamp column: Unix seconds (int/float) or numeric as seconds."""
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DatetimeIndex([])
    # If values are in seconds and large (e.g. > 1e9), treat as Unix.
    if s.iloc[0] > 1e9:
        return pd.to_datetime(s, unit="s", utc=True)
    return pd.to_datetime(s, unit="s", utc=True)


def _timestamp_parts(timestamp):
    if hasattr(timestamp, "to_pydatetime"):
        dt = timestamp.to_pydatetime()
    elif isinstance(timestamp, datetime):
        dt = timestamp
    elif isinstance(timestamp, (int, float)):
        dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    else:
        dt = timestamp

    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.astimezone(timezone.utc)
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    day_of_week = dt.weekday()
    return hour, day_of_week


def _cyclical_sin_cos(x, period: float):
    """Encode cyclical feature: sin and cos with period (max value in cycle)."""
    x = np.asarray(x, dtype=float)
    rad = 2 * np.pi * x / period
    return np.sin(rad), np.cos(rad)


def _ema_alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def _ema_next(prev, value: float, span: int) -> float:
    if prev is None:
        return value
    alpha = _ema_alpha(span)
    return alpha * value + (1 - alpha) * prev


def _rolling_std(values) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr, ddof=1))


def _ema(series, span: int):
    """Exponential moving average with span (alpha = 2/(span+1))."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(close, period: int = RSI_PERIOD):
    """RSI using Wilder smoothing (EMA of gain/loss). No future leakage."""
    import numpy as np

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd_line_and_signal(
    close,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
):
    """MACD line and signal line. No future leakage."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, signal)
    return macd_line, macd_signal


def build_features(df):
    """Compute starter-set features from OHLCV. Returns DataFrame with OUTPUT_COLS + timestamp."""
    import pandas as pd

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ts = _parse_timestamp(df["timestamp"])
    if len(ts) != len(df):
        ts = ts.reindex(df.index).ffill().bfill()
    if hasattr(ts, "hour"):
        hour = ts.hour + ts.minute / 60 + ts.second / 3600
        day_of_week = ts.dayofweek
    else:
        hour = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
        day_of_week = ts.dt.dayofweek

    time_sin, time_cos = _cyclical_sin_cos(hour, 24.0)
    dow_sin, dow_cos = _cyclical_sin_cos(day_of_week, 7.0)

    close = df["close"].astype(float)
    log_return = np.log(close / close.shift(1))

    return_lag_cols = {f"return_lag_{i}": log_return.shift(i).values for i in range(1, RETURN_LAGS + 1)}

    rolling_vol_5 = log_return.rolling(ROLLING_VOL_WINDOWS[0]).std()
    rolling_vol_20 = log_return.rolling(ROLLING_VOL_WINDOWS[1]).std()

    bar_range_pct = (df["high"] - df["low"]) / close.replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    range_hl = df["high"] - df["low"]
    body_ratio = body / range_hl.replace(0, np.nan)
    body_ratio = body_ratio.fillna(0)

    vol_ma = df["volume"].rolling(RELATIVE_VOLUME_WINDOW).mean()
    relative_volume = df["volume"] / vol_ma.replace(0, np.nan)

    rsi_14 = rsi(close, RSI_PERIOD)
    macd_line, macd_signal = macd_line_and_signal(close)
    macd_line_pct = macd_line / close.replace(0, np.nan)
    macd_signal_pct = macd_signal / close.replace(0, np.nan)

    out = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "close": close.values,
        "time_of_day_sin": time_sin,
        "time_of_day_cos": time_cos,
        "day_of_week_sin": dow_sin,
        "day_of_week_cos": dow_cos,
        "log_return": log_return.values,
        **return_lag_cols,
        "rolling_vol_5": rolling_vol_5.values,
        "rolling_vol_20": rolling_vol_20.values,
        "bar_range_pct": bar_range_pct.values,
        "relative_volume": relative_volume.values,
        "body_ratio": body_ratio.values,
        "rsi_14": rsi_14.values,
        "macd_line_pct": macd_line_pct.values,
        "macd_signal_pct": macd_signal_pct.values,
    })

    out = out.dropna().reset_index(drop=True)
    return out


# --- QC streaming (generated) ---
class FeatureComputer:
    def __init__(self):
        self.prev_close = None
        self.log_returns = deque(maxlen=max(ROLLING_VOL_WINDOWS))
        self.volumes = deque(maxlen=RELATIVE_VOLUME_WINDOW)
        self.avg_gain = None
        self.avg_loss = None
        self.ema_fast = None
        self.ema_slow = None
        self.macd_signal = None

    def update(self, timestamp, open_, high, low, close, volume):
        open_ = float(open_)
        high = float(high)
        low = float(low)
        close = float(close)
        volume = float(volume)

        hour, day_of_week = _timestamp_parts(timestamp)
        time_of_day_sin, time_of_day_cos = _cyclical_sin_cos(hour, 24.0)
        day_of_week_sin, day_of_week_cos = _cyclical_sin_cos(day_of_week, 7.0)

        if self.prev_close is None:
            log_return = float("nan")
            gain = 0.0
            loss = 0.0
        else:
            log_return = float(np.log(close / self.prev_close))
            delta = close - self.prev_close
            gain = delta if delta > 0 else 0.0
            loss = -delta if delta < 0 else 0.0

        self.avg_gain = gain if self.avg_gain is None else (gain / RSI_PERIOD) + (1 - 1 / RSI_PERIOD) * self.avg_gain
        self.avg_loss = loss if self.avg_loss is None else (loss / RSI_PERIOD) + (1 - 1 / RSI_PERIOD) * self.avg_loss
        if self.avg_loss == 0:
            rsi_14 = 50.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi_14 = 100 - (100 / (1 + rs))

        self.ema_fast = _ema_next(self.ema_fast, close, MACD_FAST)
        self.ema_slow = _ema_next(self.ema_slow, close, MACD_SLOW)
        macd_line = self.ema_fast - self.ema_slow
        self.macd_signal = _ema_next(self.macd_signal, macd_line, MACD_SIGNAL)

        self.volumes.append(volume)
        if not np.isnan(log_return):
            self.log_returns.append(log_return)

        self.prev_close = close

        if len(self.log_returns) < MIN_WARMUP_BARS or len(self.volumes) < RELATIVE_VOLUME_WINDOW:
            return None

        rolling_vol_5 = _rolling_std(list(self.log_returns)[-ROLLING_VOL_WINDOWS[0]:])
        rolling_vol_20 = _rolling_std(self.log_returns)
        vol_ma = sum(self.volumes) / len(self.volumes)
        relative_volume = volume / vol_ma if vol_ma != 0 else float("nan")
        bar_range_pct = (high - low) / close if close != 0 else float("nan")
        range_hl = high - low
        body_ratio = abs(close - open_) / range_hl if range_hl != 0 else 0.0
        returns = list(self.log_returns)
        macd_line_pct = macd_line / close if close != 0 else float("nan")
        macd_signal_pct = self.macd_signal / close if close != 0 else float("nan")

        row = [
            float(time_of_day_sin),
            float(time_of_day_cos),
            float(day_of_week_sin),
            float(day_of_week_cos),
            log_return,
            returns[-2],
            returns[-3],
            returns[-4],
            returns[-5],
            returns[-6],
            rolling_vol_5,
            rolling_vol_20,
            bar_range_pct,
            relative_volume,
            body_ratio,
            rsi_14,
            macd_line_pct,
            macd_signal_pct,
        ]
        if any(np.isnan(v) for v in row):
            return None
        return np.array(row, dtype=np.float64)
# --- end QC streaming ---


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_feature_spec(norm_params: dict) -> dict:
    return {
        "feature_set": FEATURE_SET,
        "feature_set_version": FEATURE_SET_VERSION,
        "source_hash": source_hash(),
        "feature_names": list(norm_params.get("feature_names", MODEL_INPUT_COLS)),
        "target_feature": norm_params.get("target_feature"),
        "output_cols": list(OUTPUT_COLS),
        "model_input_cols": list(MODEL_INPUT_COLS),
        "constants": {
            "rsi_period": RSI_PERIOD,
            "macd_fast": MACD_FAST,
            "macd_slow": MACD_SLOW,
            "macd_signal": MACD_SIGNAL,
            "rolling_vol_windows": list(ROLLING_VOL_WINDOWS),
            "return_lags": RETURN_LAGS,
            "relative_volume_window": RELATIVE_VOLUME_WINDOW,
        },
        "min_warmup_bars": MIN_WARMUP_BARS,
        "seq_len": norm_params.get("seq_len"),
        "pred_len": norm_params.get("pred_len"),
    }


def render_qc_module() -> str:
    imports = "\n".join([
        "from collections import deque",
        "from datetime import datetime, timezone",
        "",
        "import numpy as np",
        "",
    ])
    constants = {
        "FEATURE_SET": FEATURE_SET,
        "FEATURE_SET_VERSION": FEATURE_SET_VERSION,
        "SOURCE_HASH": source_hash(),
        "OUTPUT_COLS": OUTPUT_COLS,
        "MODEL_INPUT_COLS": MODEL_INPUT_COLS,
        "RSI_PERIOD": RSI_PERIOD,
        "MACD_FAST": MACD_FAST,
        "MACD_SLOW": MACD_SLOW,
        "MACD_SIGNAL": MACD_SIGNAL,
        "ROLLING_VOL_WINDOWS": ROLLING_VOL_WINDOWS,
        "RETURN_LAGS": RETURN_LAGS,
        "RELATIVE_VOLUME_WINDOW": RELATIVE_VOLUME_WINDOW,
        "MIN_WARMUP_BARS": MIN_WARMUP_BARS,
    }
    constant_lines = "\n".join(f"{key} = {value!r}" for key, value in constants.items())
    helper_sources = "\n\n".join([
        inspect.getsource(_timestamp_parts),
        inspect.getsource(_cyclical_sin_cos),
        inspect.getsource(_ema_alpha),
        inspect.getsource(_ema_next),
        inspect.getsource(_rolling_std),
        inspect.getsource(FeatureComputer),
    ])
    return "\n".join([
        "# Generated by scripts/wrap_up_for_qc.py. Do not edit by hand.",
        imports,
        constant_lines,
        "",
        helper_sources,
        "",
    ])


def write_feature_spec(path: Path, norm_params: dict) -> dict:
    spec = build_feature_spec(norm_params)
    path.write_text(json.dumps(spec, indent=2) + "\n")
    return spec
