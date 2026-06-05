#!/usr/bin/env python3
"""
Compare batch feature engineering with the streaming QuantConnect feature path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from feature_starter_set import FeatureComputer, MODEL_INPUT_COLS, build_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check batch vs streaming feature parity.")
    p.add_argument("--csv", type=Path, default=None, help="OHLCV CSV (default: tmp/train_ohlcv.csv)")
    p.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance (default: 1e-5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    csv_path = args.csv or (root / "tmp" / "train_ohlcv.csv")

    if not csv_path.exists():
        print(f"Error: not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path, low_memory=False)
    df = df.sort_values("timestamp").reset_index(drop=True)
    batch = build_features(df)

    computer = FeatureComputer()
    stream_rows = []
    for row in df.itertuples(index=False):
        feature_row = computer.update(
            getattr(row, "timestamp"),
            getattr(row, "open"),
            getattr(row, "high"),
            getattr(row, "low"),
            getattr(row, "close"),
            getattr(row, "volume"),
        )
        if feature_row is not None:
            stream_rows.append(feature_row)

    stream = np.asarray(stream_rows, dtype=np.float64)
    batch_values = batch[MODEL_INPUT_COLS].to_numpy(dtype=np.float64)

    if stream.shape != batch_values.shape:
        print(f"Error: shape mismatch batch={batch_values.shape} stream={stream.shape}", file=sys.stderr)
        sys.exit(1)

    if not np.allclose(batch_values, stream, atol=args.atol, rtol=0):
        diff = np.abs(batch_values - stream)
        row_idx, col_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
        print(
            "Error: feature mismatch "
            f"row={row_idx} column={MODEL_INPUT_COLS[col_idx]} "
            f"batch={batch_values[row_idx, col_idx]} stream={stream[row_idx, col_idx]} "
            f"abs_diff={diff[row_idx, col_idx]}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Feature parity OK: {len(stream):,} rows, {len(MODEL_INPUT_COLS)} model input columns")


if __name__ == "__main__":
    main()
