"""Shared fixtures for QML-GA tests.

Synthetic data is used everywhere so tests are self-contained — no dependency on
real CSVs in `data/`. Real-data smoke tests live in test_smoke.py and skip if
the expected files are missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make src/ importable without poetry install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def synthetic_xy():
    """Tiny linearly separable binary dataset, 4 features, 32 samples.

    Returns (X, y) with y in {-1, +1}.
    """
    rng = np.random.default_rng(seed=42)
    n = 32
    X = rng.uniform(-1.0, 1.0, size=(n, 4))
    # Label = sign of first feature (trivially learnable)
    y = np.where(X[:, 0] >= 0, 1.0, -1.0)
    return X, y


@pytest.fixture
def synthetic_dataset_csv(tmp_path, synthetic_xy):
    """Write the synthetic dataset to a CSV in tmp_path and return the path."""
    import pandas as pd

    X, y = synthetic_xy
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target"] = y
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_dataset_cfg(synthetic_dataset_csv):
    """Dict shaped like configs/datasets/*.yaml's `dataset:` block."""
    return {
        "name": "synthetic",
        "path": str(synthetic_dataset_csv),
        "target": "target",
        "features": {"use": "all", "first_k": None, "names": []},
        "kfold": {"n_splits": 2, "shuffle": True, "stratified": True, "seed": 42},
    }


@pytest.fixture
def amplitude_fm_cfg():
    """Clean amplitude feature_map block (no noise)."""
    return {"type": "amplitude", "params": {"normalize": True}}


@pytest.fixture
def zz_fm_cfg():
    """Clean ZZ feature_map block (no noise)."""
    return {"type": "zz", "params": {"reps": 2, "entanglement": "full"}}


@pytest.fixture
def ansatz1_d2_cfg():
    """Shallow ansatz_1 depth=2 — small enough to keep tests fast."""
    return {"type": "ansatz_1", "params": {"depth": 2}}
