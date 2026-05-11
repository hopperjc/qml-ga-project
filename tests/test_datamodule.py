"""Tests for DataModule — wire computation, label remapping, error paths."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qml_ga.data.datamodule import DataModule


# --------------------------- required_wires ---------------------------


@pytest.mark.parametrize(
    "n_features, expected",
    [
        (1, 0),    # log2(1) = 0
        (2, 1),
        (3, 2),    # ceil(log2(3)) = 2
        (4, 2),
        (5, 3),
        (8, 3),
        (9, 4),
        (16, 4),
        (30, 5),   # WDBC has 30 features → 5 qubits
        (60, 6),   # Sonar has 60 features → 6 qubits
    ],
)
def test_required_wires_amplitude(n_features, expected):
    assert DataModule.required_wires("amplitude", n_features) == expected


@pytest.mark.parametrize("n_features", [1, 4, 30, 60])
def test_required_wires_zz_is_identity(n_features):
    """ZZ Feature Map uses one qubit per feature (1:1)."""
    assert DataModule.required_wires("zz", n_features) == n_features


def test_required_wires_unknown_raises():
    with pytest.raises(ValueError, match="desconhecido"):
        DataModule.required_wires("unknown_fm", 4)


@pytest.mark.parametrize("fm_type", ["amplitude", "AMPLITUDE", "Amplitude"])
def test_required_wires_case_insensitive(fm_type):
    """fm_type comparison should be case-insensitive."""
    assert DataModule.required_wires(fm_type, 4) == 2


# ----------------------- ensure_wires_compatible -----------------------


def test_ensure_wires_compatible_ok():
    """Should not raise when wires match the computed requirement."""
    DataModule.ensure_wires_compatible("amplitude", 4, wires=2)
    DataModule.ensure_wires_compatible("zz", 4, wires=4)


def test_ensure_wires_compatible_mismatch_raises():
    with pytest.raises(ValueError, match="Incompatibilidade"):
        DataModule.ensure_wires_compatible("amplitude", 4, wires=3)


# ------------------------------ load_all ------------------------------


@pytest.fixture
def csv_binary_01(tmp_path):
    """CSV with labels in {0, 1} — must be remapped to {-1, +1}."""
    df = pd.DataFrame({
        "f0": [0.1, 0.2, 0.3, 0.4],
        "f1": [0.5, 0.6, 0.7, 0.8],
        "target": [0, 1, 0, 1],
    })
    p = tmp_path / "binary01.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def csv_binary_pm1(tmp_path):
    """CSV with labels already in {-1, +1}."""
    df = pd.DataFrame({
        "f0": [0.1, 0.2, 0.3, 0.4],
        "f1": [0.5, 0.6, 0.7, 0.8],
        "target": [-1, 1, -1, 1],
    })
    p = tmp_path / "binary_pm1.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def csv_binary_strings(tmp_path):
    """CSV with string labels ('M' / 'B' like WDBC) — must be remapped."""
    df = pd.DataFrame({
        "f0": [0.1, 0.2, 0.3, 0.4],
        "f1": [0.5, 0.6, 0.7, 0.8],
        "target": ["M", "B", "M", "B"],
    })
    p = tmp_path / "binary_str.csv"
    df.to_csv(p, index=False)
    return p


def test_load_all_remaps_01_to_pm1(csv_binary_01):
    dm = DataModule({"path": str(csv_binary_01), "target": "target"})
    X, y, feats = dm.load_all()
    assert set(np.unique(y)) == {-1.0, 1.0}
    assert X.shape == (4, 2)
    assert feats == ["f0", "f1"]


def test_load_all_preserves_pm1(csv_binary_pm1):
    dm = DataModule({"path": str(csv_binary_pm1), "target": "target"})
    X, y, feats = dm.load_all()
    assert set(np.unique(y)) == {-1.0, 1.0}


def test_load_all_remaps_strings(csv_binary_strings):
    dm = DataModule({"path": str(csv_binary_strings), "target": "target"})
    X, y, feats = dm.load_all()
    assert set(np.unique(y)) == {-1.0, 1.0}
    # The mapping should be deterministic (sorted alphabetically): 'B' < 'M'
    # so 'B' → -1, 'M' → +1


def test_load_all_missing_file_raises(tmp_path):
    dm = DataModule({"path": str(tmp_path / "does_not_exist.csv"), "target": "y"})
    with pytest.raises(FileNotFoundError):
        dm.load_all()


def test_load_all_missing_target_raises(csv_binary_01):
    dm = DataModule({"path": str(csv_binary_01), "target": "nonexistent_col"})
    with pytest.raises(ValueError, match="target"):
        dm.load_all()


def test_load_all_three_classes_raises(tmp_path):
    """Non-binary string labels should raise (binary-only design)."""
    df = pd.DataFrame({
        "f0": [0.1, 0.2, 0.3],
        "target": ["A", "B", "C"],
    })
    p = tmp_path / "multi.csv"
    df.to_csv(p, index=False)
    dm = DataModule({"path": str(p), "target": "target"})
    with pytest.raises(ValueError, match="bin"):
        dm.load_all()
