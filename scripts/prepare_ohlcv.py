 #!/usr/bin/env python3
"""
Prepare OHLCV data from Kraken master_q4 for transformer training.

Reads the requested instrument/timeframe from data/master_q4/, normalizes
the CSV (header + columns), and writes to tmp/train_ohlcv.csv so the
training script can always load from a single path.

Usage:
  python prepare_ohlcv.py
  python prepare_ohlcv.py --instrument ETHUSD --timeframe 5min
  python prepare_ohlcv.py -i BTCUSD -t 1h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Standard column names for Kraken OHLCV (minutes)
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trades"]

# Timeframe string -> minutes (Kraken file suffix)
TIMEFRAME_MINUTES = {
    "1min": 1,
    "1m": 1,
    "5min": 5,
    "5m": 5,
    "15min": 15,
    "15m": 15,
    "30min": 30,
    "30m": 30,
    "1h": 60,
    "60min": 60,
    "60m": 60,
    "4h": 240,
    "240min": 240,
    "12h": 720,
    "720min": 720,
    "1d": 1440,
    "1440min": 1440,
    "daily": 1440,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare OHLCV from Kraken master_q4 for training."
    )
    p.add_argument(
        "-i", "--instrument",
        default="BTCUSD",
        help="Instrument symbol (default: BTCUSD)",
    )
    p.add_argument(
        "-t", "--timeframe",
        default="1min",
        help="Timeframe: 1min, 5min, 15min, 30min, 1h, 4h, 12h, 1d (default: 1min)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root data dir containing master_q4/ (default: project data/)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write prepared CSV (default: project tmp/)",
    )
    return p.parse_args()


def normalize_instrument(s: str) -> str:
    """Kraken uses XBT for BTC in some pairs; master_q4 uses BTCUSD."""
    s = s.strip().upper()
    if s in ("XBTUSD", "XBT/USD"):
        return "BTCUSD"
    return s.replace("/", "")


def timeframe_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf in TIMEFRAME_MINUTES:
        return TIMEFRAME_MINUTES[tf]
    # Try integer minutes
    try:
        m = int(tf)
        if m > 0:
            return m
    except ValueError:
        pass
    raise SystemExit(f"Unknown timeframe: {tf}. Use e.g. 1min, 5min, 1h, 1d.")


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir or (root / "data")
    out_dir = args.out_dir or (root / "tmp")

    instrument = normalize_instrument(args.instrument)
    tf_min = timeframe_to_minutes(args.timeframe)

    # Kraken master_q4: {SYMBOL}_{MINUTES}.csv
    source_file = data_dir / "master_q4" / f"{instrument}_{tf_min}.csv"
    if not source_file.exists():
        print(f"Error: not found: {source_file}", file=sys.stderr)
        print("Available BTC timeframes: 1, 5, 15, 30, 60, 240, 720, 1440", file=sys.stderr)
        sys.exit(1)

    # Read CSV; some files have header, some don't
    df = pd.read_csv(source_file, header=None, low_memory=False)
    first_val = str(df.iloc[0, 0]).strip().lower()
    if first_val == "timestamp":
        # First row is header
        df = df.iloc[1:].reset_index(drop=True)
    df.columns = COLUMNS[: len(df.columns)]

    # Ensure numeric and sorted by time
    for c in ["timestamp", "open", "high", "low", "close", "volume", "trades"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "train_ohlcv.csv"
    df.to_csv(out_csv, index=False)

    # Optional: write metadata so trainer knows what it's loading
    meta_path = out_dir / "train_meta.txt"
    meta_path.write_text(
        f"instrument={instrument}\ntimeframe={args.timeframe}\ntimeframe_minutes={tf_min}\nrows={len(df)}\n"
    )

    print(f"Prepared {len(df):,} rows: {instrument} {args.timeframe} -> {out_csv}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
