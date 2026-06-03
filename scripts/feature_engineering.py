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

import pandas as pd

from feature_starter_set import OUTPUT_COLS, build_features


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
