"""End-to-end smoke tests: 1 run per representative configuration.

These are slow (each runs 2-fold k-fold training over 5 epochs / 5 generations).
Marked `@pytest.mark.smoke` so the fast suite (`pytest -m "not smoke"`) skips them.
Run with `pytest -m smoke -v`.
"""
from __future__ import annotations

import json
import os

import pytest


pytestmark = pytest.mark.smoke


def _make_cfg(synthetic_dataset_cfg, fm_cfg, ansatz_cfg, optimizer_cfg, n_qubits, k_folds=2):
    return {
        "dataset": synthetic_dataset_cfg,
        "feature_map": fm_cfg,
        "ansatz": ansatz_cfg,
        "device": {"name": "auto", "wires": n_qubits, "shots": None},
        "optimizer": optimizer_cfg,
        "k_folds": k_folds,
        "seed": 42,
        "objective_name": "accuracy",
        "mid_checkpoint_epoch": 9999,  # disable mid-checkpoint for smoke
    }


@pytest.mark.parametrize(
    "fm_name,fm_cfg_name,n_qubits,noise",
    [
        ("amplitude_clean", "amplitude_fm_cfg", 2, None),
        ("amplitude_noisy", "amplitude_fm_cfg", 2,
         {"after_feature_map": {"type": "amplitude_damping", "gamma": 0.05}}),
        ("zz_clean", "zz_fm_cfg", 4, None),
        ("zz_noisy", "zz_fm_cfg", 4,
         {"after_feature_map": {"type": "phase_damping", "gamma": 0.05}}),
    ],
)
def test_smoke_classical(tmp_path, request, synthetic_dataset_cfg, ansatz1_d2_cfg,
                         fm_name, fm_cfg_name, n_qubits, noise):
    fm_cfg = dict(request.getfixturevalue(fm_cfg_name))
    if noise is not None:
        fm_cfg = {**fm_cfg, "noise": noise}
    cfg = _make_cfg(
        synthetic_dataset_cfg, fm_cfg, ansatz1_d2_cfg,
        optimizer_cfg={"type": "adam", "classical": {"lr": 0.05, "epochs": 5, "batch_size": 8}},
        n_qubits=n_qubits,
    )

    from qml_ga.experiments.kfold_train import run_kfold_experiment
    run_dir = tmp_path / f"run_{fm_name}"
    rep_dir = tmp_path / f"rep_{fm_name}"
    summary = run_kfold_experiment(cfg, str(run_dir), report_dir=str(rep_dir),
                                   progress=False, resume=False)

    assert isinstance(summary, dict)
    assert os.path.exists(rep_dir / "summary.json")
    assert os.path.exists(rep_dir / "folds.csv")


def test_smoke_ga(tmp_path, synthetic_dataset_cfg, amplitude_fm_cfg, ansatz1_d2_cfg):
    cfg = _make_cfg(
        synthetic_dataset_cfg, amplitude_fm_cfg, ansatz1_d2_cfg,
        optimizer_cfg={"type": "ga", "ga": {
            "population_size": 8,
            "num_generations": 5,
            "selection_type": "tournament",
            "crossover_type": "single_point",
            "mutation_type": "random",
            "mutation_percent_genes": 10,
            "elitism": 2,
            "init_range_low": -3.14,
            "init_range_high": 3.14,
        }},
        n_qubits=2,
    )

    from qml_ga.experiments.kfold_train import run_kfold_experiment
    run_dir = tmp_path / "run_ga"
    rep_dir = tmp_path / "rep_ga"
    summary = run_kfold_experiment(cfg, str(run_dir), report_dir=str(rep_dir),
                                   progress=False, resume=False)

    assert isinstance(summary, dict)
    assert os.path.exists(rep_dir / "summary.json")


def test_smoke_runner_execute_one_combo(tmp_path, synthetic_dataset_cfg,
                                        amplitude_fm_cfg, ansatz1_d2_cfg):
    """Exercise the runner's per-task entry point used by ProcessPoolExecutor."""
    cfg = _make_cfg(
        synthetic_dataset_cfg, amplitude_fm_cfg, ansatz1_d2_cfg,
        optimizer_cfg={"type": "adam", "classical": {"lr": 0.05, "epochs": 3, "batch_size": 8}},
        n_qubits=2,
    )

    from qml_ga.sweep.runner import execute_one_combo
    res = execute_one_combo(
        cfg=cfg, tag="smoke_runner",
        reports_dir=str(tmp_path / "reports"),
        runs_dir=str(tmp_path / "runs"),
        resume=False,
    )

    assert res["status"] == "ok"
    assert res["wall_time_seconds"] is not None and res["wall_time_seconds"] > 0
    assert res["peak_memory_mb"] is not None and res["peak_memory_mb"] > 0

    # Re-running with resume=True must short-circuit
    res2 = execute_one_combo(
        cfg=cfg, tag="smoke_runner",
        reports_dir=str(tmp_path / "reports"),
        runs_dir=str(tmp_path / "runs"),
        resume=True,
    )
    assert res2["status"] == "skip"
