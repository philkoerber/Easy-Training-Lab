# Training Lab

A minimal pipeline to **train a small transformer on OHLCV data** (Kraken master_q4), **package it for QuantConnect**, and run it on QC with the model loaded from Object Store.

- **Model:** 50 candles in → 10 candles out (e.g. last 50 minutes → next 10 minutes of close).
- **Stack:** [tsai](https://github.com/timeseriesAI/tsai) (TST), PyTorch, pandas. Training on Mac GPU (MPS) when available.
- **North star:** train locally → upload package to Object Store → run `quantconnect/main.py` with the same features, normalization, and model weights as training.

---

## Goal

1. **Clean load** — Resolve the latest package via `ohlcv_latest`; load `norm_params.json`, `feature_spec.json`, and `model_state.pt`; build inline TST from norm params. Fail loudly in debug if anything is missing.
2. **Train/QC parity** — Same 19 features, column order, and z-score normalization as training. `norm_params.json` is the runtime contract; `feature_starter_set.py` is the feature recipe.
3. **Observable runs** — QC backtests should show model loaded, buffer ready, prediction vs price, and enter/exit reason via `debug` logs.

---

## Setup

1. **Clone and enter the repo.**

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Data:** Put Kraken master_q4 CSVs under `data/master_q4/` (one file per instrument/timeframe, e.g. `BTCUSD_1.csv`, `ETHUSD_5.csv`).

5. **QuantConnect credentials** (for upload): copy `.env.example` to `.env` and set `QC_USER_ID`, `QC_API_TOKEN`, `QC_ORGANIZATION_ID`.

---

## Workflow

| Step | Script | What it does | Output |
|------|--------|--------------|--------|
| 1 | `prepare_ohlcv.py` | Load Kraken CSV, normalize columns | `tmp/train_ohlcv.csv`, `tmp/train_meta.txt` |
| 2 | `feature_engineering.py` | Compute 19 starter features from OHLCV | `tmp/train_features.csv` |
| 3 | `train_transformer.py` | Train TST (50 in → 10 out), save norm params | `tmp/model.pth`, `tmp/norm_params.json` |
| 4 | `test_inference_model.py` | Sanity-check model + norm params locally | stdout |
| 5 | `wrap_up_for_qc.py` | Build QC package, optionally upload | `tmp/qc_package/` |
| 6 | QC project | Copy algorithm files, backtest | — |

**Recommended path (matches QuantConnect):**
```bash
python3 scripts/prepare_ohlcv.py
python3 scripts/feature_engineering.py
python3 scripts/train_transformer.py --csv tmp/train_features.csv
python3 scripts/test_inference_model.py
python3 scripts/wrap_up_for_qc.py --upload
# Copy tmp/qc_package/feature_starter_set.py into your QC project alongside main.py
# Sync quantconnect/main.py into the QC project and backtest
```

**Minimal local path (raw OHLCV only — not for QC deploy):**
```bash
python3 scripts/prepare_ohlcv.py
python3 scripts/train_transformer.py
```

Step 3 uses the **first CSV column** (except `timestamp`) as the prediction target. With engineered features that is still `close`. The same column order must match at deploy time.

Feature formulas live in one place: [`scripts/feature_starter_set.py`](scripts/feature_starter_set.py). Training uses the batch path; QC uses a generated copy of the streaming path.

---

### 1. Prepare OHLCV

```bash
python3 scripts/prepare_ohlcv.py                    # default: SOLUSD 1min
python3 scripts/prepare_ohlcv.py -i ETHUSD -t 5min
```

Output: `tmp/train_ohlcv.csv`, `tmp/train_meta.txt`.

### 2. Feature engineering

```bash
python3 scripts/feature_engineering.py
```

Output: `tmp/train_features.csv` (19 features, `close` first). Constants and formulas come from `feature_starter_set.py`.

Optional: `python3 scripts/test_feature_parity.py` compares batch vs streaming features before upload.

### 3. Train

```bash
python3 scripts/train_transformer.py --csv tmp/train_features.csv
```

Output: `tmp/model.pth`, `tmp/norm_params.json` (`seq_len`, `pred_len`, `feature_names`, `mean`, `std`).

Options: `--epochs`, `--batch-size`, `--stride`, `--out-dir`.

### 4. Validate locally

```bash
python3 scripts/test_inference_model.py
```

Runs a few sequences from the training CSV through the model using the same norm params that QC will use.

### 5. Package and upload

```bash
python3 scripts/wrap_up_for_qc.py          # build only
python3 scripts/wrap_up_for_qc.py --upload # build + Object Store upload
```

**`tmp/qc_package/` contents:**

| File | Uploaded to Object Store? | Used by |
|------|---------------------------|---------|
| `model_state.pt` | yes | QC algorithm (Object Store) |
| `norm_params.json` | yes | QC algorithm (Object Store) |
| `feature_spec.json` | yes | QC algorithm (hash + constants check) |
| `manifest.json` | yes | package metadata |
| `feature_starter_set.py` | no — copy into QC project | QC algorithm (live features) |
| `qc_folder.txt` | yes, as `ohlcv_latest` | points algorithm at latest folder |

Upload uses `QC_USER_ID`, `QC_API_TOKEN`, `QC_ORGANIZATION_ID`. User ID and token: [My Account → Security](https://www.quantconnect.com/settings/). Organization ID: from your org URL (`.../organization/<id>`).

### 6. Deploy on QuantConnect

1. Upload with `wrap_up_for_qc.py --upload`.
2. Copy `tmp/qc_package/feature_starter_set.py` into your QC project.
3. Copy/sync [`quantconnect/main.py`](quantconnect/main.py) into the QC project.
4. Backtest. Check debug logs for model load, feature warmup (~20 bars), buffer full, and predictions.

The algorithm resolves the latest model folder from `ohlcv_latest`, loads weights and norm params from Object Store, computes 19 live features via `FeatureComputer`, then runs inference. [`quantconnect/justLoading.py`](quantconnect/justLoading.py) is a smaller script that only verifies Object Store load.

---

## Project layout

```
Training Lab/
├── data/master_q4/           # Kraken CSVs
├── quantconnect/
│   ├── main.py                 # QC algorithm: Object Store model + live features
│   └── justLoading.py          # load-only smoke test
├── scripts/
│   ├── prepare_ohlcv.py
│   ├── feature_starter_set.py  # single source of truth for feature formulas
│   ├── feature_engineering.py
│   ├── train_transformer.py
│   ├── test_inference_model.py
│   ├── test_feature_parity.py  # optional batch vs streaming check
│   └── wrap_up_for_qc.py
├── tmp/                        # data, model, qc_package (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- Kraken master_q4 data in `data/master_q4/`
- For upload: QuantConnect account and API credentials (see `.env.example`)
