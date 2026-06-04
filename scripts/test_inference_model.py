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
    p.add_argument("--start-date", default=None, help="First sample timestamp/date to evaluate")
    p.add_argument("--end-date", default=None, help="Last sample timestamp/date to evaluate")
    p.add_argument("--sample-step", type=int, default=1, help="Evaluate every Nth available sample")
    p.add_argument("--threshold", type=float, default=0.001, help="Entry threshold for summary metrics")
    p.add_argument("--exit-threshold", type=float, default=-0.0005, help="Exit threshold for summary metrics")
    p.add_argument("--horizon", type=int, default=-1, help="Prediction horizon index for summary metrics")
    p.add_argument("--summary-only", action="store_true", help="Only print aggregate diagnostics")
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


def _horizon_index(horizon: int, pred_len: int) -> int:
    idx = horizon if horizon >= 0 else pred_len + horizon
    if idx < 0 or idx >= pred_len:
        raise ValueError(f"horizon {horizon} out of range for pred_len={pred_len}")
    return idx


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
    target_type = norm_params.get("target_type", "price")
    mean = np.array(norm_params["mean"], dtype=np.float64)
    std = np.array(norm_params["std"], dtype=np.float64)
    std[std == 0] = 1.0
    horizon_idx = _horizon_index(args.horizon, pred_len)

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
        try:
            torch.set_default_device("cpu")
        except AttributeError:
            pass
        with torch.device("cpu"):
            model = TST(c_in=n_features, c_out=pred_len, seq_len=seq_len)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to("cpu")
        print(f"Loaded model from {model_path} (state_dict)")
    else:
        print(f"Error: no model found at {model_path}", file=sys.stderr)
        sys.exit(1)

    if not csv_path.exists():
        print(f"Error: not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(csv_path, low_memory=False)
    timestamps = _parse_timestamps(df)
    for c in feature_names:
        if c not in df.columns:
            print(f"Error: CSV missing column '{c}' (expected from norm_params)", file=sys.stderr)
            sys.exit(1)
    data = df[feature_names].astype(np.float64).values
    need = seq_len + pred_len
    if data.shape[0] < need:
        print(f"Error: need at least {need} rows, got {data.shape[0]}", file=sys.stderr)
        sys.exit(1)

    data_norm = (data - mean) / std
    starts = list(range(0, data.shape[0] - need + 1, max(1, args.sample_step)))
    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    if timestamps is not None and (start_date is not None or end_date is not None):
        filtered = []
        for i in starts:
            sample_time = timestamps.iloc[i + seq_len - 1]
            if pd.isna(sample_time):
                continue
            if start_date is not None and sample_time < start_date:
                continue
            if end_date is not None and sample_time > end_date:
                continue
            filtered.append(i)
        starts = filtered
    if not starts:
        print("Error: no sequences selected for evaluation", file=sys.stderr)
        sys.exit(1)

    X_list = []
    current_close_list = []
    future_close_list = []
    for i in starts:
        X_list.append(data_norm[i : i + seq_len].T)
        current_close_list.append(data[i + seq_len - 1, 0])
        future_close_list.append(data[i + seq_len : i + seq_len + pred_len, 0])
    X = np.stack(X_list, axis=0).astype(np.float32)
    current_close = np.array(current_close_list, dtype=np.float64)
    future_close = np.stack(future_close_list, axis=0).astype(np.float64)
    x_t = torch.from_numpy(X)

    with torch.no_grad():
        pred = model(x_t)
    pred_np = pred.detach().cpu().numpy()

    if target_type == "future_return":
        target_mean = np.array(norm_params["target_mean"], dtype=np.float64)
        target_std = np.array(norm_params["target_std"], dtype=np.float64)
        target_std[target_std == 0] = 1.0
        pred_return = pred_np * target_std + target_mean
        actual_return = (future_close / current_close[:, None]) - 1.0
        pred_price = current_close[:, None] * (1.0 + pred_return)
        y_actual_price = future_close
    else:
        # Backward compatible path for older models trained on normalized future close.
        mean_target = mean[0]
        std_target = std[0]
        pred_price = pred_np * std_target + mean_target
        y_actual_price = future_close
        pred_return = (pred_price / current_close[:, None]) - 1.0
        actual_return = (future_close / current_close[:, None]) - 1.0

    pred_h = pred_return[:, horizon_idx]
    actual_h = actual_return[:, horizon_idx]
    bias = pred_h - actual_h
    price_bias = (pred_price[:, horizon_idx] - current_close) / current_close
    direction_hits = np.sign(pred_h) == np.sign(actual_h)
    price_mae = np.abs(pred_price[:, horizon_idx] - y_actual_price[:, horizon_idx]).mean()

    print(f"\nInference summary on {len(starts)} sequence(s) (seq_len={seq_len}, pred_len={pred_len}, horizon={horizon_idx}):")
    print(f"  target_type:          {target_type}")
    print(f"  price MAE:            {price_mae:.6f}")
    print(f"  mean return bias:     {bias.mean():.6%}")
    print(f"  median return bias:   {np.median(bias):.6%}")
    print(f"  mean price bias:      {price_bias.mean():.6%}")
    print(f"  median price bias:    {np.median(price_bias):.6%}")
    print(f"  directional hit rate: {direction_hits.mean():.2%}")
    print(f"  pred > entry thresh:  {(pred_h > args.threshold).mean():.2%} (threshold={args.threshold:.4%})")
    print(f"  pred < exit thresh:   {(pred_h < args.exit_threshold).mean():.2%} (threshold={args.exit_threshold:.4%})")
    train_close_mean = float(mean[0])
    eval_close_mean = float(current_close.mean())
    if train_close_mean != 0:
        regime_gap = (eval_close_mean / train_close_mean) - 1.0
        print(f"  eval close mean:      {eval_close_mean:.6f} ({regime_gap:.2%} vs train mean close)")

    if args.summary_only:
        print("Done.")
        return

    n_seq = min(max(1, args.num_sequences), len(starts))
    print(f"\nFirst {n_seq} sequence(s):\n")
    for i in range(n_seq):
        print(f"  Sequence {i + 1}:")
        print(f"    pred (price):  {pred_price[i].tolist()}")
        print(f"    actual (price): {y_actual_price[i].tolist()}")
        mae_price = np.abs(pred_price[i] - y_actual_price[i]).mean()
        print(f"    MAE (price):   {mae_price:.6f}\n")
    print("Done.")


if __name__ == "__main__":
    main()
