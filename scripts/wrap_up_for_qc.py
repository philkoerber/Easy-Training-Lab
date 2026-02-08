#!/usr/bin/env python3
"""
Build a QuantConnect-ready package from trained model + norm params, and optionally
upload to QuantConnect Object Store via API. Uses env vars for API keys (see .env.example).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

QC_PREFIX = "ohlcv_model"
BASE_URL = "https://www.quantconnect.com/api/v2"


def parse_args():
    p = argparse.ArgumentParser(description="Package model for QuantConnect and optionally upload.")
    p.add_argument("--model", type=Path, default=None, help="Path to tsai/fastai export (default: tmp/model.pth)")
    p.add_argument("--norm-params", type=Path, default=None, help="Path to norm_params.json (default: tmp/norm_params.json)")
    p.add_argument("--out-dir", type=Path, default=None, help="Package output dir (default: tmp/qc_package)")
    p.add_argument("--upload", action="store_true", help="Upload package to QuantConnect Object Store")
    return p.parse_args()


def get_qc_headers():
    """Build QuantConnect API auth headers. Returns None if credentials missing or placeholder."""
    user_id = os.environ.get("QC_USER_ID", "")
    api_token = os.environ.get("QC_API_TOKEN", "")
    org_id = os.environ.get("QC_ORGANIZATION_ID", "")
    placeholders = ("", "0", "replace_me", "your_token", "your_org_id")
    if not user_id or not api_token or not org_id:
        return None, None
    if str(user_id).strip() in placeholders or api_token.strip() in placeholders or org_id.strip() in placeholders:
        return None, None
    import time
    timestamp = str(int(time.time()))
    time_stamped_token = f"{api_token}:{timestamp}".encode("utf-8")
    hashed = hashlib.sha256(time_stamped_token).hexdigest()
    auth_str = f"{user_id}:{hashed}".encode("utf-8")
    auth_b64 = base64.b64encode(auth_str).decode("ascii")
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Timestamp": timestamp,
    }
    return headers, org_id


def build_package(model_path: Path, norm_path: Path, out_dir: Path) -> list[Path]:
    """Produce model_state.pt and norm_params.json in out_dir. Returns list of created files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []

    # Norm params: copy or symlink
    if norm_path.exists():
        dest_norm = out_dir / "norm_params.json"
        dest_norm.write_text(norm_path.read_text())
        created.append(dest_norm)
    else:
        print(f"Warning: {norm_path} not found; norm_params.json will be missing.", file=sys.stderr)

    # Model: extract state_dict from tsai/fastai export
    if not model_path.exists():
        print(f"Error: {model_path} not found.", file=sys.stderr)
        return created

    try:
        from fastai.learner import load_learner
        import torch
    except ImportError as e:
        print(f"Error: need fastai and torch to extract model: {e}", file=sys.stderr)
        return created

    learn = load_learner(model_path, cpu=True)
    state = learn.model.state_dict()
    dest_model = out_dir / "model_state.pt"
    torch.save(state, dest_model)
    created.append(dest_model)

    # Optional manifest
    manifest = {
        "input_shape": [1, "n_features", "seq_len"],
        "output_steps": "pred_len",
        "files": ["model_state.pt", "norm_params.json"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    created.append(out_dir / "manifest.json")

    return created


def upload_file(headers: dict, org_id: str, key: str, file_path: Path) -> bool:
    """Upload one file to QuantConnect Object Store. Returns True on success."""
    url = f"{BASE_URL}/object/set"
    with open(file_path, "rb") as f:
        data = {"organizationId": org_id, "key": key}
        files = {"objectData": (file_path.name, f, "application/octet-stream")}
        # requests sends multipart; QC expects form with organizationId, key, and objectData file
        r = requests.post(url, headers=headers, data=data, files=files, timeout=60)
    if r.status_code != 200:
        print(f"Upload failed for {key}: {r.status_code} {r.text}", file=sys.stderr)
        return False
    try:
        j = r.json()
        if not j.get("success", True):
            print(f"Upload failed for {key}: {j}", file=sys.stderr)
            return False
    except Exception:
        pass
    return True


def main():
    if load_dotenv:
        load_dotenv()

    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    tmp = root / "tmp"
    model_path = args.model or (tmp / "model.pth")
    norm_path = args.norm_params or (tmp / "norm_params.json")
    out_dir = args.out_dir or (tmp / "qc_package")

    # Step 1: build package
    created = build_package(model_path, norm_path, out_dir)
    if not created:
        print("No files produced.", file=sys.stderr)
        sys.exit(1)
    print(f"Package built in {out_dir}:")
    for p in created:
        print(f"  {p.name}")

    # Step 2: optional upload
    if not args.upload:
        print("Skipping upload (use --upload to upload to QuantConnect Object Store).")
        return

    headers, org_id = get_qc_headers()
    if headers is None:
        print("Set QC_USER_ID, QC_API_TOKEN, QC_ORGANIZATION_ID to upload (see .env.example).")
        return

    for fpath in created:
        key = f"{QC_PREFIX}/{fpath.name}"
        if upload_file(headers, org_id, key, fpath):
            print(f"Uploaded: {key}")
        else:
            sys.exit(1)
    print("Upload done.")


if __name__ == "__main__":
    main()
