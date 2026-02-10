#!/usr/bin/env python3
"""
Build a training-ready feature CSV from OHLCV data.

Reads OHLCV CSV (e.g. tmp/train_ohlcv.csv), computes the starter-set of features,
drops leading NaNs from rolling/lag burn-in, and writes tmp/train_features.csv.
Column order is fixed so the first feature column is close (prediction target).

Usage:
  python feature_engineering.py
  python feature_engineering.py -i tmp/train_ohlcv.csv -o tmp/train_features.csv
  python feature_engineering.py -t 5min
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    "bar_range",
    "relative_volume",
    "body_ratio",
    "rsi_14",
    "macd_line",
    "macd_signal",
]

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ROLLING_VOL_WINDOWS = (5, 20)
RETURN_LAGS = 5
RELATIVE_VOLUME_WINDOW = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build feature CSV from OHLCV for transformer training."
    )
    p.add_argument(
        "-i", "--input",
        type=Path,
        default=None,
        help="Input OHLCV CSV (default: tmp/train_ohlcv.csv)",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output feature CSV (default: tmp/train_features.csv)",
    )
    p.add_argument(
        "-t", "--timeframe",
        default=None,
        help="Timeframe for context (e.g. 1min, 5min). Read from train_meta.txt if not set.",
    )
    return p.parse_args()


def _parse_timestamp(series: pd.Series) -> pd.DatetimeIndex:
    """Parse timestamp column: Unix seconds (int/float) or numeric as seconds."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DatetimeIndex([])
    # If values are in seconds and large (e.g. > 1e9), treat as Unix
    if s.iloc[0] > 1e9:
        return pd.to_datetime(s, unit="s", utc=True)
    # Otherwise assume seconds since epoch
    return pd.to_datetime(s, unit="s", utc=True)


def _cyclical_sin_cos(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode cyclical feature: sin and cos with period (max value in cycle)."""
    x = np.asarray(x, dtype=float)
    rad = 2 * np.pi * x / period
    return np.sin(rad), np.cos(rad)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with span (alpha = 2/(span+1))."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI using Wilder smoothing (EMA of gain/loss). No future leakage."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd_line_and_signal(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series]:
    """MACD line and signal line. No future leakage."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = _ema(macd_line, signal)
    return macd_line, macd_signal


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute starter-set features from OHLCV. Returns DataFrame with OUTPUT_COLS + timestamp."""
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Ensure numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Time from timestamp
    ts = _parse_timestamp(df["timestamp"])
    if len(ts) != len(df):
        ts = ts.reindex(df.index).ffill().bfill()
    # Support both DatetimeIndex and Series
    if hasattr(ts, "hour"):
        hour = ts.hour + ts.minute / 60 + ts.second / 3600  # 0..24
        day_of_week = ts.dayofweek  # 0..6 Monday=0
    else:
        hour = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
        day_of_week = ts.dt.dayofweek

    time_sin, time_cos = _cyclical_sin_cos(hour, 24.0)
    dow_sin, dow_cos = _cyclical_sin_cos(day_of_week, 7.0)

    close = df["close"].astype(float)
    log_return = np.log(close / close.shift(1))

    # Lags 1..5
    return_lag_cols = {f"return_lag_{i}": log_return.shift(i).values for i in range(1, RETURN_LAGS + 1)}

    # Rolling volatility (std of log return)
    rolling_vol_5 = log_return.rolling(ROLLING_VOL_WINDOWS[0]).std()
    rolling_vol_20 = log_return.rolling(ROLLING_VOL_WINDOWS[1]).std()

    # Bar range and body ratio
    bar_range = (df["high"] - df["low"]) / close.replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    range_hl = df["high"] - df["low"]
    body_ratio = body / range_hl.replace(0, np.nan)
    body_ratio = body_ratio.fillna(0)

    # Relative volume
    vol_ma = df["volume"].rolling(RELATIVE_VOLUME_WINDOW).mean()
    relative_volume = df["volume"] / vol_ma.replace(0, np.nan)

    rsi_14 = rsi(close, RSI_PERIOD)
    macd_line, macd_signal = macd_line_and_signal(close)

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
        "bar_range": bar_range.values,
        "relative_volume": relative_volume.values,
        "body_ratio": body_ratio.values,
        "rsi_14": rsi_14.values,
        "macd_line": macd_line.values,
        "macd_signal": macd_signal.values,
    })

    # Drop leading rows with any NaN (burn-in from lags/rolling)
    out = out.dropna().reset_index(drop=True)
    return out


def read_timeframe_from_meta(meta_path: Path) -> str | None:
    """Read timeframe_minutes or timeframe from train_meta.txt if present."""
    if not meta_path.exists():
        return None
    for line in meta_path.read_text().strip().splitlines():
        if "=" in line:
            k, v = line.strip().split("=", 1)
            if k in ("timeframe", "timeframe_minutes"):
                return v.strip()
    return None


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    in_path = args.input or (root / "tmp" / "train_ohlcv.csv")
    out_path = args.output or (root / "tmp" / "train_features.csv")
    out_dir = out_path.parent

    if not in_path.exists():
        print(f"Error: not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Timeframe: only used for optional meta; feature formulas don't depend on it
    if args.timeframe is None:
        meta_path = in_path.parent / "train_meta.txt"
        args.timeframe = read_timeframe_from_meta(meta_path) or "1min"

    df = pd.read_csv(in_path, header=0, low_memory=False)
    out = build_features(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    # Optional: update train_meta.txt for traceability (replace existing features lines)
    meta_path = out_dir / "train_meta.txt"
    new_lines = f"features=starter_set\nfeature_rows={len(out)}\n"
    if meta_path.exists():
        lines = [ln for ln in meta_path.read_text().splitlines() if not ln.strip().startswith("features=") and not ln.strip().startswith("feature_rows=")]
        meta_path.write_text("\n".join(lines) + "\n" + new_lines)
    else:
        meta_path.write_text(new_lines)

    print(f"Features: {len(out):,} rows, {len(OUTPUT_COLS)} columns -> {out_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
