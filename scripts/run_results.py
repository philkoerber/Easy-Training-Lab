"""Write training and holdout benchmarks under results/ (gitignored)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def results_root(root: Path) -> Path:
    path = root / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def run_dir(root: Path, run_id: str) -> Path:
    path = results_root(root) / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def set_latest_run(root: Path, run_id: str) -> None:
    write_json(results_root(root) / "latest_run.json", {"run_id": run_id})


def load_latest_run_id(root: Path) -> str | None:
    path = results_root(root) / "latest_run.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("run_id")


def append_benchmark(root: Path, record: dict) -> None:
    record = {**record, "recorded_at": datetime.now(timezone.utc).isoformat()}
    line = json.dumps(record, separators=(",", ":"))
    with open(results_root(root) / "benchmarks.jsonl", "a") as f:
        f.write(line + "\n")


def read_train_meta(meta_path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not meta_path.exists():
        return meta
    for line in meta_path.read_text().splitlines():
        if "=" in line:
            key, value = line.strip().split("=", 1)
            meta[key.strip()] = value.strip()
    return meta
