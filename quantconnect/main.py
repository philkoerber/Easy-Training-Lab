# region imports
from AlgorithmImports import *
import json
import io
from collections import deque
# endregion

# -----------------------------------------------------------------------------
# Inline TST model (QuantConnect may not have tsai; matches tsai TST defaults)
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np


def _ifnone(a, b):
    return b if a is None else a


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k)
        scores = scores / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return context, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, mask=None):
        bs = Q.size(0)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        context, attn = _ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        return self.W_O(context), attn


class _Transpose12(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2).contiguous()


class _TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, dropout=0.1, activation="gelu"):
        super().__init__()
        d_k = _ifnone(d_k, d_model // n_heads)
        d_v = _ifnone(d_v, d_model // n_heads)
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.dropout_attn = nn.Dropout(dropout)
        self.batchnorm_attn = nn.Sequential(_Transpose12(), nn.BatchNorm1d(d_model), _Transpose12())
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), act_fn, nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.batchnorm_ffn = nn.Sequential(_Transpose12(), nn.BatchNorm1d(d_model), _Transpose12())

    def forward(self, src, mask=None):
        src2, _ = self.self_attn(src, src, src, mask=mask)
        src = src + self.dropout_attn(src2)
        src = self.batchnorm_attn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        src = self.batchnorm_ffn(src)
        return src


class _TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, dropout=0.1, activation="gelu", n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            _TSTEncoderLayer(d_model, n_heads, d_k, d_v, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class TST(nn.Module):
    """Minimal TST (Time Series Transformer) matching tsai TST(c_in, c_out, seq_len) defaults."""

    def __init__(self, c_in, c_out, seq_len, n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, dropout=0.1, act="gelu", fc_dropout=0.0):
        super().__init__()
        self.c_out, self.seq_len = c_out, seq_len
        self.W_P = nn.Linear(c_in, d_model)
        q_len = seq_len
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k, d_v, d_ff, dropout, act, n_layers)
        self.head_nf = q_len * d_model
        layers = [nn.GELU() if act == "gelu" else nn.ReLU(), nn.Flatten()]
        if fc_dropout:
            layers.append(nn.Dropout(fc_dropout))
        layers.append(nn.Linear(self.head_nf, c_out))
        self.head = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        u = self.W_P(x.transpose(2, 1))
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = z.transpose(2, 1).contiguous()
        return self.head(z)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Object Store: wrap_up_for_qc.py uploads to a folder like ohlcv_btcusd_1min_seq50_pred10_...
OHLCV_LATEST_KEY = "ohlcv_latest"
SEQ_LEN = 50
PRED_LEN = 10
THRESHOLD = 0.001


# -----------------------------------------------------------------------------
# Algorithm
# -----------------------------------------------------------------------------
class LogicalFluorescentOrangeGiraffe(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 11, 1)
        self.set_end_date(2025, 1, 31)
        self.set_cash(100000)
        self.symbol = self.add_crypto("BTCUSD", Resolution.MINUTE)
        self.model = None
        self.norm_params = None
        self.bar_buffer = deque(maxlen=SEQ_LEN)
        self.threshold = THRESHOLD

        self._load_model_from_object_store()

    # --- Model loading ---
    def _load_model_from_object_store(self):
        store = self.ObjectStore
        folder = "ohlcv_model"
        if store.contains_key(OHLCV_LATEST_KEY):
            raw = store.read_bytes(OHLCV_LATEST_KEY)
            folder = bytes(raw).decode("utf-8").strip()
            self.debug(f"Using Object Store folder: {folder}")
        norm_key = f"{folder}/norm_params.json"
        model_key = f"{folder}/model_state.pt"
        if not store.contains_key(norm_key):
            self.debug(f"Object Store: {norm_key} not found. Upload with: python3 scripts/wrap_up_for_qc.py --upload")
            return
        if not store.contains_key(model_key):
            self.debug(f"Object Store: {model_key} not found.")
            return

        norm_bytes = store.read_bytes(norm_key)
        self.norm_params = json.loads(bytes(norm_bytes).decode("utf-8"))
        self.debug(f"Loaded norm_params: seq_len={self.norm_params['seq_len']}, pred_len={self.norm_params['pred_len']}")

        model_bytes = store.read_bytes(model_key)
        try:
            state_dict = torch.load(io.BytesIO(bytes(model_bytes)), map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(io.BytesIO(bytes(model_bytes)), map_location="cpu")

        n_features = self.norm_params["n_features"]
        pred_len = self.norm_params["pred_len"]
        seq_len = self.norm_params["seq_len"]
        self.model = TST(c_in=n_features, c_out=pred_len, seq_len=seq_len)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.debug("Model loaded successfully.")

    # --- Data / inference ---
    def _bar_to_features(self, bar):
        """Extract feature tuple (open, high, low, close, volume, trades). QC has no trades; use 0."""
        o = float(bar.Open)
        h = float(bar.High)
        l = float(bar.Low)
        c = float(bar.Close)
        v = float(bar.Volume)
        trades = 0.0
        return (o, h, l, c, v, trades)

    def _run_inference(self):
        if self.model is None or self.norm_params is None or len(self.bar_buffer) < SEQ_LEN:
            return None
        np_params = self.norm_params
        mean = np.array(np_params["mean"], dtype=np.float64)
        std = np.array(np_params["std"], dtype=np.float64)
        std[std == 0] = 1.0
        feature_names = np_params["feature_names"]

        rows = []
        for bar in self.bar_buffer:
            row = self._bar_to_features(bar)
            rows.append(row)
        data = np.array(rows, dtype=np.float64)
        data_norm = (data - mean) / std
        X = data_norm.T
        X = np.expand_dims(X, axis=0).astype(np.float32)
        x_t = torch.from_numpy(X)

        with torch.no_grad():
            pred_norm = self.model(x_t)
        pred_norm_np = pred_norm.numpy().flatten()
        pred_price = pred_norm_np * std[0] + mean[0]
        return pred_price

    # --- Trading logic ---
    def on_data(self, data: Slice):
        if not data.Bars.ContainsKey(self.symbol):
            return
        bar = data.Bars[self.symbol]
        self.bar_buffer.append(bar)

        if len(self.bar_buffer) < SEQ_LEN:
            return

        pred_price = self._run_inference()
        if pred_price is None:
            return

        current_close = float(bar.Close)
        pred_close = float(pred_price[-1])
        threshold = self.threshold

        if pred_close > current_close * (1 + threshold):
            if not self.portfolio.invested:
                self.set_holdings(self.symbol, 1.0)
                self.debug(f"Inference: pred_close={pred_close:.2f}, current={current_close:.2f}, signal=long")
        else:
            if self.portfolio.invested:
                self.liquidate(self.symbol)
                self.debug(f"Inference: pred_close={pred_close:.2f}, current={current_close:.2f}, signal=flat")
