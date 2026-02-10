#!/usr/bin/env python3
"""
Train a small transformer on OHLCV: 50 candles in -> 10 out.
Uses tsai (TST); reads tmp/train_ohlcv.csv, saves model and norm_params.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# tsai
from tsai.data.core import get_ts_dls
from tsai.learner import ts_learner
from tsai.models.TST import TST

SEQ_LEN = 50
PRED_LEN = 10
VAL_PCT = 0.1


def parse_args():
    p = argparse.ArgumentParser(description="Train OHLCV transformer (tsai TST).")
    p.add_argument("--csv", type=Path, default=None, help="OHLCV CSV (default: tmp/train_ohlcv.csv)")
    p.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: tmp/)")
    p.add_argument("--epochs", type=int, default=3, help="Epochs for fit_one_cycle")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--stride", type=int, default=10, help="Sliding window stride (1 = max samples)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    csv_path = args.csv or (root / "tmp" / "train_ohlcv.csv")
    out_dir = args.out_dir or (root / "tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load and select features: all columns except timestamp, in CSV order (first = target)
    df = pd.read_csv(csv_path, low_memory=False)
    if "timestamp" in df.columns:
        feature_cols = [c for c in df.columns if c != "timestamp"]
    else:
        feature_cols = df.columns.tolist()
    if not feature_cols:
        print("Error: no feature columns found", file=sys.stderr)
        sys.exit(1)
    for c in feature_cols:
        if c not in df.columns:
            print(f"Error: missing column {c}", file=sys.stderr)
            sys.exit(1)
    data = df[feature_cols].astype(np.float64).values

    n_features = len(feature_cols)
    n_rows = len(data)
    need = SEQ_LEN + PRED_LEN
    if n_rows < need:
        print(f"Error: need at least {need} rows, got {n_rows}", file=sys.stderr)
        sys.exit(1)

    # Time-based split: use first (1 - VAL_PCT) for train stats and both splits
    n_val = int(n_rows * VAL_PCT)
    n_train = n_rows - n_val
    train_data = data[:n_train]

    # Z-score normalize using train stats
    mean = np.nanmean(train_data, axis=0)
    std = np.nanstd(train_data, axis=0)
    std[std == 0] = 1.0
    data_norm = (data - mean) / std

    # Sliding windows
    stride = max(1, args.stride)
    X_list, y_list = [], []
    for i in range(0, n_rows - need + 1, stride):
        X_list.append(data_norm[i : i + SEQ_LEN].T)  # (n_features, SEQ_LEN)
        y_list.append(data_norm[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN, 0])  # next 10 of first feature (close)

    X = np.stack(X_list, axis=0).astype(np.float32)   # (n_samples, n_features, SEQ_LEN)
    y = np.stack(y_list, axis=0).astype(np.float32)  # (n_samples, PRED_LEN)

    # Splits: time-based, last portion as val (by index in X)
    n_samples = X.shape[0]
    n_val_samples = max(1, int(n_samples * VAL_PCT))
    val_start = n_samples - n_val_samples
    splits = (np.arange(0, val_start), np.arange(val_start, n_samples))

    # Norm params for inference / QC
    norm_params = {
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "n_features": n_features,
        "feature_names": feature_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    norm_path = out_dir / "norm_params.json"
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved {norm_path}")

    # DataLoaders
    dls = get_ts_dls(X, y, splits=splits, batch_size=args.batch_size)

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Learner and train (instantiate TST ourselves so tsai doesn't inject custom_head -> Conv1d)
    model = TST(
        c_in=n_features,
        c_out=PRED_LEN,
        seq_len=SEQ_LEN,
    )
    learn = ts_learner(dls, model, device=device)
    learn.fit_one_cycle(args.epochs)

    # Export full learner (for wrap-up to extract state_dict)
    model_path = out_dir / "model.pth"
    learn.export(model_path)
    print(f"Exported {model_path}")
    print("Done.")


if __name__ == "__main__":
    main()
