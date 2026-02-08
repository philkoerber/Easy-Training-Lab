# region imports
from AlgorithmImports import *
import json
import io
# endregion

# Object Store: wrap_up_for_qc.py uploads to a folder like ohlcv_btcusd_1min_seq50_pred10_20260208_203045
# and writes that folder name to "ohlcv_latest" so we can resolve the latest upload
OHLCV_LATEST_KEY = "ohlcv_latest"


class LogicalFluorescentOrangeGiraffe(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 8, 7)
        self.set_cash(100000)
        # Model is trained on BTCUSD 1min (prepare_ohlcv.py default)
        self.add_crypto("BTCUSD", Resolution.MINUTE)

        # Load model and norm params from Object Store (uploaded by wrap_up_for_qc.py)
        self.model_state = None
        self.norm_params = None
        self._load_model_from_object_store()

    def _load_model_from_object_store(self):
        # Resolve folder: use "ohlcv_latest" if present, else fall back to fixed "ohlcv_model"
        folder = "ohlcv_model"
        if self.ObjectStore.ContainsKey(OHLCV_LATEST_KEY):
            folder = self.ObjectStore.ReadBytes(OHLCV_LATEST_KEY).decode("utf-8").strip()
            self.Debug(f"Using Object Store folder: {folder}")
        norm_key = f"{folder}/norm_params.json"
        model_key = f"{folder}/model_state.pt"
        if not self.ObjectStore.ContainsKey(norm_key):
            self.Debug(f"Object Store: {norm_key} not found. Upload with: python3 scripts/wrap_up_for_qc.py --upload")
            return
        if not self.ObjectStore.ContainsKey(model_key):
            self.Debug(f"Object Store: {model_key} not found.")
            return

        # Load norm_params.json
        norm_bytes = self.ObjectStore.ReadBytes(norm_key)
        self.norm_params = json.loads(norm_bytes.decode("utf-8"))
        self.Debug(f"Loaded norm_params: seq_len={self.norm_params['seq_len']}, pred_len={self.norm_params['pred_len']}, n_features={self.norm_params['n_features']}")

        # Load PyTorch state_dict (verify only; no forward pass here)
        model_bytes = self.ObjectStore.ReadBytes(model_key)
        import torch
        try:
            state_dict = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(io.BytesIO(model_bytes), map_location="cpu")
        self.model_state = state_dict
        self.Debug(f"Loaded model_state.pt: {len(state_dict)} parameter tensors")
        self.Debug("Model loaded successfully.")

    def on_data(self, data: Slice):
        if not self.portfolio.invested:
            self.set_holdings("BTCUSD", 1.0)
