# Training Lab

A minimal pipeline to **train a small transformer on OHLCV data** (Kraken master_q4), then **package it for QuantConnect** and optionally upload to their Object Store.

- **Model:** 50 candles in → 10 candles out (e.g. last 50 minutes → next 10 minutes of close).
- **Stack:** [tsai](https://github.com/timeseriesAI/tsai) (TST), PyTorch, pandas. Training on Mac GPU (MPS) when available.
- **Output:** Trained model + normalization params, plus a QC-ready bundle (`model_state.pt` + `norm_params.json`) for inference in QuantConnect.

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

---

## How to use

### 1. Prepare OHLCV data

Select instrument and timeframe; writes a single normalized CSV for training.

```bash
# Defaults: BTCUSD, 1min → tmp/train_ohlcv.csv
python3 scripts/prepare_ohlcv.py

# Custom
python3 scripts/prepare_ohlcv.py -i ETHUSD -t 5min
python3 scripts/prepare_ohlcv.py --instrument BTCUSD --timeframe 1h
```

- **Output:** `tmp/train_ohlcv.csv`, `tmp/train_meta.txt`.
- **Options:** `-i/--instrument`, `-t/--timeframe`, `--data-dir`, `--out-dir`. Timeframes: `1min`, `5min`, `15min`, `30min`, `1h`, `4h`, `12h`, `1d`.

### 2. Train the transformer

Train a TST model (50 steps in → 10 out). Uses `tmp/train_ohlcv.csv` and writes the model and normalization params to `tmp/`.

```bash
python3 scripts/train_transformer.py --epochs 3
```

- **Output:** `tmp/model.pth` (tsai/fastai export), `tmp/norm_params.json` (seq_len, pred_len, mean/std, etc.).
- **Options:** `--csv`, `--out-dir`, `--epochs`, `--batch-size`, `--stride`. Feature set is configurable in the script (`FEATURE_COLS`, default: close only).

### 3. Wrap up for QuantConnect

Build a QuantConnect-ready package and optionally upload it to the Object Store.

```bash
# Only build the package (tmp/qc_package/)
python3 scripts/wrap_up_for_qc.py

# Build and upload (requires QC API credentials in env)
python3 scripts/wrap_up_for_qc.py --upload
```

- **Package:** `tmp/qc_package/model_state.pt`, `norm_params.json`, `manifest.json`. QC algorithms load these (PyTorch `state_dict` + same preprocessing).
- **Upload:** Uses `QC_USER_ID`, `QC_API_TOKEN`, `QC_ORGANIZATION_ID`. Copy `.env.example` to `.env` and set values. User ID and API token: [My Account → Security](https://www.quantconnect.com/settings/). **Organization ID:** in the Algorithm Lab, click “Connected as: &lt;org name&gt;” in the top bar, open your organization, then copy the long hex string from the URL (e.g. `https://www.quantconnect.com/organization/5cad178b...` → the part after `/organization/`). Without valid credentials, the script only builds the package.

---

## Project layout

```
Training Lab/
├── data/
│   └── master_q4/          # Kraken CSVs (e.g. BTCUSD_1.csv)
├── quantconnect/
│   └── main.py             # Example QC algorithm: loads model from Object Store
├── scripts/
│   ├── prepare_ohlcv.py    # 1. Filter & normalize OHLCV → tmp/train_ohlcv.csv
│   ├── train_transformer.py # 2. Train TST → tmp/model.pth, tmp/norm_params.json
│   └── wrap_up_for_qc.py   # 3. Package + optional upload to QC Object Store
├── tmp/                    # Prepared data, model, and QC package (gitignored)
├── .env.example            # Template for QC API keys
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- Kraken master_q4 data in `data/master_q4/` (see above).
- For upload: QuantConnect account and API token/organization ID (see `.env.example`).
