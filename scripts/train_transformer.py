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
    p.add_argument("--epochs", type=int, default=20, help="Epochs for fit_one_cycle")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--stride", type=int, default=1, help="Sliding window stride (1 = max samples)")
    p.add_argument("--train-end", default=None, help="Last timestamp/date used for training stats and samples")
    p.add_argument("--val-start", default=None, help="First timestamp/date used for validation samples")
    p.add_argument("--val-end", default=None, help="Last timestamp/date used for validation samples")
    return p.parse_args()


def _parse_timestamps(df: pd.DataFrame) -> pd.Series | None:
    if "timestamp" not in df.columns:
        return None
    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts_num.notna().any():
        return pd.to_datetime(ts_num, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")


def _parse_date(value: str | None):
    if value is None:
        return None
    return pd.Timestamp(value, tz="UTC")


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
    timestamps = _parse_timestamps(df)
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

    train_end = _parse_date(args.train_end)
    val_start = _parse_date(args.val_start)
    val_end = _parse_date(args.val_end)

    # Time-based split: use first (1 - VAL_PCT) for train stats unless date flags are provided.
    if timestamps is not None and train_end is not None:
        train_rows = timestamps <= train_end
        if not train_rows.any():
            print(f"Error: --train-end leaves no training rows: {args.train_end}", file=sys.stderr)
            sys.exit(1)
        train_data = data[train_rows.to_numpy()]
    else:
        n_val = int(n_rows * VAL_PCT)
        n_train = n_rows - n_val
        train_data = data[:n_train]

    # Z-score normalize using train stats
    mean = np.nanmean(train_data, axis=0)
    std = np.nanstd(train_data, axis=0)
    std[std == 0] = 1.0
    data_norm = (data - mean) / std

    # Sliding windows: target is future close return from the current input-window close.
    stride = max(1, args.stride)
    X_list, y_list = [], []
    sample_time_list = []
    for i in range(0, n_rows - need + 1, stride):
        X_list.append(data_norm[i : i + SEQ_LEN].T)  # (n_features, SEQ_LEN)
        current_close = data[i + SEQ_LEN - 1, 0]
        future_close = data[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN, 0]
        y_list.append((future_close / current_close) - 1.0)
        if timestamps is not None:
            sample_time_list.append(timestamps.iloc[i + SEQ_LEN - 1])

    X = np.stack(X_list, axis=0).astype(np.float32)   # (n_samples, n_features, SEQ_LEN)
    y_raw = np.stack(y_list, axis=0).astype(np.float64)  # (n_samples, PRED_LEN)

    # Splits: time-based, last portion as val (by index in X) unless date flags are provided.
    n_samples = X.shape[0]
    if timestamps is not None and (train_end is not None or val_start is not None or val_end is not None):
        sample_times = pd.Series(sample_time_list)
        if val_start is not None or val_end is not None:
            val_mask = pd.Series([True] * n_samples)
            if val_start is not None:
                val_mask &= sample_times >= val_start
            if val_end is not None:
                val_mask &= sample_times <= val_end
            train_mask = ~val_mask
            if val_start is not None:
                train_mask &= sample_times < val_start
        else:
            train_mask = sample_times <= train_end
            val_mask = sample_times > train_end
        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())
        if len(train_idx) == 0 or len(val_idx) == 0:
            print("Error: date split produced empty train or validation samples", file=sys.stderr)
            sys.exit(1)
        splits = (train_idx, val_idx)
    else:
        n_val_samples = max(1, int(n_samples * VAL_PCT))
        val_start_idx = n_samples - n_val_samples
        splits = (np.arange(0, val_start_idx), np.arange(val_start_idx, n_samples))

    target_mean = np.nanmean(y_raw[splits[0]], axis=0)
    target_std = np.nanstd(y_raw[splits[0]], axis=0)
    target_std[target_std == 0] = 1.0
    y = ((y_raw - target_mean) / target_std).astype(np.float32)

    # Norm params for inference / QC
    norm_params = {
        "seq_len": SEQ_LEN,
        "pred_len": PRED_LEN,
        "n_features": n_features,
        "feature_names": feature_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "target_type": "future_return",
        "target_feature": feature_cols[0],
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "target_horizon_steps": PRED_LEN,
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
