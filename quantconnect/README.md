# QuantConnect algorithm (load model from Object Store)

Copy `main.py` into a QuantConnect project (or create a new algorithm and paste the code).

**Requires:** The model and norm params to be in your org’s Object Store under:
- `ohlcv_model/model_state.pt`
- `ohlcv_model/norm_params.json`

Upload them from this repo with:
```bash
python3 scripts/wrap_up_for_qc.py --upload
```

**Behavior:** In `Initialize`, the algorithm reads both files from the Object Store, parses the JSON, and loads the PyTorch state_dict. It logs to the Debug console so you can confirm the model loaded. The existing SPY/BND/AAPL allocation logic is unchanged.

If QuantConnect’s Python API uses snake_case for Object Store, change `self.ObjectStore.ContainsKey` → `self.ObjectStore.contains_key` and `self.ObjectStore.ReadBytes` → `self.ObjectStore.read_bytes`, and `self.Debug` → `self.debug` if needed.
