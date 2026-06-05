---
name: smoke-test-iteration
description: >-
  Runs the standard Training Lab pipeline end-to-end with a fast 1-epoch train
  to verify setup. Use when the user asks for a smoke test or quick pipeline check.
---

# Smoke test iteration

Same pipeline as always (see README workflow). Only difference: **train fast** — 1 epoch and a larger stride so it finishes quickly. Overwriting `tmp/model.pth` and related outputs is fine.

Requires `.env` with `QC_USER_ID`, `QC_API_TOKEN`, `QC_ORGANIZATION_ID` for upload + QC backtest steps.

## Steps (repo root, in order)

```bash
python3 scripts/prepare_ohlcv.py
python3 scripts/feature_engineering.py
python3 scripts/train_transformer.py --csv tmp/train_features.csv --epochs 1 --stride 20 --batch-size 64
python3 scripts/test_inference_model.py --summary-only --csv tmp/train_features.csv --sample-step 5000
python3 scripts/wrap_up_for_qc.py --upload
```

Optional before train: `python3 scripts/test_feature_parity.py`

## QuantConnect smoke backtest

After upload, run a short QC backtest that loads the model from Object Store:

1. Sync `quantconnect/justLoading.py` (load-only) or `quantconnect/main.py` + `tmp/qc_package/feature_starter_set.py` into the QC project.
2. For a **fast** plumbing check, use `justLoading.py` as `main.py` with a ~1-week range (e.g. `set_start_date(2024, 11, 1)` + `set_end_date(2024, 11, 7)`).
3. Compile, backtest, confirm debug logs: `Using Object Store folder:`, `Loaded norm_params`, `Loaded model_state.pt` (or `Model ready` for full `main.py`).

Use QuantConnect MCP (`create_compile` → `read_compile` → `create_backtest` → `read_backtest`) or the QC API via `.env` if MCP list/read responses fail validation.

## Pass criteria

- Each script exits 0
- `tmp/norm_params.json` has `"target_type": "future_return"`
- `tmp/qc_package/` has `model_state.pt`, `norm_params.json`, `manifest.json`
- Upload prints `Uploaded: ohlcv_latest`
- QC backtest completes without runtime error; debug shows model loaded from latest Object Store folder

Weak metrics from 1 epoch are OK. Smoke checks plumbing, not strategy quality.

## Not in scope

- Separate output dirs (`tmp/smoke/`, etc.)
- Subagent orchestration
- Skipping prepare/features because files already exist
