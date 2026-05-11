"""Tests for qml_ga.metrics.classification — binarize_logits and all_metrics."""
from __future__ import annotations

import numpy as np
import pytest

from qml_ga.metrics.classification import all_metrics, binarize_logits


# --------------------------- binarize_logits ---------------------------


def test_binarize_logits_positive_is_plus_one():
    assert np.array_equal(binarize_logits([0.1, 1.0, 0.9]), [1.0, 1.0, 1.0])


def test_binarize_logits_negative_is_minus_one():
    assert np.array_equal(binarize_logits([-0.1, -1.0, -0.5]), [-1.0, -1.0, -1.0])


def test_binarize_logits_zero_is_plus_one():
    """Threshold is >=0, so exactly zero goes to +1."""
    assert np.array_equal(binarize_logits([0.0]), [1.0])


def test_binarize_logits_handles_array_shape():
    """Multidim input is raveled."""
    out = binarize_logits([[0.1, -0.1], [0.2, -0.2]])
    assert out.shape == (4,)
    assert np.array_equal(out, [1.0, -1.0, 1.0, -1.0])


# ----------------------------- all_metrics -----------------------------


def test_all_metrics_returns_correct_keys():
    """Result is a dict with the four expected metric keys."""
    y_true = [1, -1, 1, -1]
    logits = [0.5, -0.5, 0.5, -0.5]
    out = all_metrics(y_true, logits)
    assert set(out.keys()) == {"accuracy", "precision", "recall", "f1"}


def test_all_metrics_perfect_prediction_is_one():
    """Perfectly aligned logits should yield 1.0 for all metrics."""
    y_true = [1, -1, 1, -1, 1, -1]
    logits = [10, -10, 10, -10, 10, -10]
    out = all_metrics(y_true, logits)
    assert out["accuracy"] == 1.0
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["f1"] == 1.0


def test_all_metrics_all_wrong_is_zero():
    """Inverted predictions yield 0 across the board."""
    y_true = [1, -1, 1, -1]
    logits = [-10, 10, -10, 10]
    out = all_metrics(y_true, logits)
    assert out["accuracy"] == 0.0
    assert out["recall"] == 0.0
    # precision is 0 because no TP and we asked zero_division=0


def test_all_metrics_balanced_50pct():
    """Half right, half wrong → accuracy=0.5."""
    y_true = [1, 1, -1, -1]
    logits = [10, -10, -10, 10]  # correct on 1st and 3rd
    out = all_metrics(y_true, logits)
    assert out["accuracy"] == 0.5


def test_all_metrics_imbalanced_high_accuracy_low_recall():
    """All predictions are negative; with 1 positive in y_true,
    accuracy is high (3/4) but recall is 0."""
    y_true = [1, -1, -1, -1]
    logits = [-1, -1, -1, -1]
    out = all_metrics(y_true, logits)
    assert out["accuracy"] == 0.75
    assert out["recall"] == 0.0


def test_all_metrics_returns_floats():
    """All metric values are Python floats (JSON-serializable)."""
    out = all_metrics([1, -1, 1], [0.5, -0.5, 0.5])
    for k, v in out.items():
        assert isinstance(v, float), f"{k} is {type(v).__name__}, expected float"


def test_all_metrics_f1_consistency():
    """F1 = 2*P*R/(P+R) within numerical tolerance."""
    y_true = [1, 1, 1, -1, -1, -1]
    logits = [10, 10, -10, 10, -10, -10]  # TP=2, FN=1, FP=1, TN=2
    out = all_metrics(y_true, logits)
    p, r, f1 = out["precision"], out["recall"], out["f1"]
    expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    assert abs(f1 - expected_f1) < 1e-9
