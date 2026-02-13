#!/usr/bin/env python3
"""
Standalone script: load trained model + norm params, take a few sequences from the
original training CSV, run inference, and print predictions (and actuals for comparison).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsai.models.TST import TST


def parse_args():
    p = argparse.ArgumentParser(description="Test inference on a few sequences from training data.")
    p.add_argument("--csv", type=Path, default=None, help="Training CSV (default: tmp/train_ohlcv.csv)")
    p.add_argument("--norm", type=Path, default=None, help="norm_params.json (default: tmp/norm_params.json)")
    p.add_argument("--model", type=Path, default=None, help="model.pth or model_state.pt (default: tmp/model.pth)")
    p.add_argument("-n", "--num-sequences", type=int, default=5, help="Number of sequences to test (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    tmp = root / "tmp"
    csv_path = args.csv or (tmp / "train_ohlcv.csv")
    norm_path = args.norm or (tmp / "norm_params.json")
    model_path = args.model or (tmp / "model.pth")
    n_seq = max(1, args.num_sequences)

    if not norm_path.exists():
        print(f"Error: not found: {norm_path}", file=sys.stderr)
        sys.exit(1)
    norm_params = json.loads(norm_path.read_text())
    for key in ("seq_len", "pred_len", "n_features", "feature_names", "mean", "std"):
        if key not in norm_params:
            print(f"Error: norm_params missing key: {key}", file=sys.stderr)
            sys.exit(1)
    seq_len = int(norm_params["seq_len"])
    pred_len = int(norm_params["pred_len"])
    n_features = int(norm_params["n_features"])
    feature_names = norm_params["feature_names"]
    mean = np.array(norm_params["mean"], dtype=np.float64)
    std = np.array(norm_params["std"], dtype=np.float64)
    std[std == 0] = 1.0

    # Load model: .pth = full learner (matches training), .pt = state_dict only (e.g. QC package)
    model = None
    if model_path.suffix == ".pth" and model_path.exists():
        from fastai.learner import load_learner
        learn = load_learner(model_path, cpu=True)
        model = learn.model
        model.eval()
        print(f"Loaded model from {model_path} (learner)")
    elif model_path.suffix == ".pt" and model_path.exists():
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location="cpu")
        model = TST(c_in=n_features, c_out=pred_len, seq_len=seq_len)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"Loaded model from {model_path} (state_dict)")
    else:
        print(f"Error: no model found at {model_path}", file=sys.stderr)
        sys.exit(1)

    if not csv_path.exists():
        print(f"Error: not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path, low_memory=False)
    for c in feature_names:
        if c not in df.columns:
            print(f"Error: CSV missing column '{c}' (expected from norm_params)", file=sys.stderr)
            sys.exit(1)
    data = df[feature_names].astype(np.float64).values
    need = seq_len + pred_len
    if data.shape[0] < need:
        print(f"Error: need at least {need} rows, got {data.shape[0]}", file=sys.stderr)
        sys.exit(1)
    if n_seq > data.shape[0] - need + 1:
        n_seq = data.shape[0] - need + 1
        print(f"Using at most {n_seq} sequences (data length limit)", file=sys.stderr)

    data_norm = (data - mean) / std
    X_list = []
    y_actual_list = []
    for i in range(n_seq):
        X_list.append(data_norm[i : i + seq_len].T)
        y_actual_list.append(data_norm[i + seq_len : i + seq_len + pred_len, 0])
    X = np.stack(X_list, axis=0).astype(np.float32)
    y_actual = np.stack(y_actual_list, axis=0).astype(np.float32)
    x_t = torch.from_numpy(X)

    with torch.no_grad():
        pred = model(x_t)
    pred_np = pred.numpy()

    # Denormalize to price space (target = first feature, e.g. close)
    mean_target = mean[0]
    std_target = std[0]
    pred_price = pred_np * std_target + mean_target
    y_actual_price = y_actual * std_target + mean_target

    print(f"\nInference on {n_seq} sequence(s) (seq_len={seq_len}, pred_len={pred_len}):\n")
    for i in range(n_seq):
        print(f"  Sequence {i + 1}:")
        print(f"    pred (price):  {pred_price[i].tolist()}")
        print(f"    actual (price): {y_actual_price[i].tolist()}")
        mae_price = np.abs(pred_price[i] - y_actual_price[i]).mean()
        print(f"    MAE (price):   {mae_price:.6f}\n")
    print("Done.")


if __name__ == "__main__":
    main()
