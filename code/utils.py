"""Shared paths, constants, and small helpers for the notebook pipeline."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
FIGURES_DIR = CODE_DIR / "figures"

for d in (RAW_DIR, ARTIFACTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_amazon_csv() -> Path | None:
    """Locate the kritanjalijain Amazon Reviews train CSV under data/raw.

    Expected files (per Kaggle dataset): train.csv, test.csv with columns
    [polarity (1=neg, 2=pos), title, text]. Returns the train CSV path or None.
    """
    candidates = [
        RAW_DIR / "train.csv",
        RAW_DIR / "amazon_reviews" / "train.csv",
        RAW_DIR / "amazon-reviews" / "train.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(RAW_DIR.rglob("train.csv"))
    return matches[0] if matches else None


def save_metrics(name: str, payload: dict) -> Path:
    out = ARTIFACTS_DIR / f"metrics_{name}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out


def load_all_metrics() -> dict[str, dict]:
    return {
        p.stem.replace("metrics_", ""): json.loads(p.read_text())
        for p in sorted(ARTIFACTS_DIR.glob("metrics_*.json"))
    }
