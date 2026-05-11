"""Tests for qml_ga.sweep.runner internals — _tag_is_done and execute_one_combo
resume path. End-to-end runner is covered by test_smoke.py (marker `smoke`)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qml_ga.sweep.runner import _tag_is_done, execute_one_combo


# ----------------------------- _tag_is_done -----------------------------


def test_tag_is_done_empty_dir(tmp_path):
    """No artifacts → not done."""
    assert _tag_is_done(str(tmp_path)) is False


def test_tag_is_done_summary_present(tmp_path):
    """summary.json existing → considered done."""
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    assert _tag_is_done(str(tmp_path)) is True


def test_tag_is_done_status_done_true(tmp_path):
    """status.json with done=True → done."""
    status = {"done": True, "started_at": "2026-05-08 10:00:00"}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    assert _tag_is_done(str(tmp_path)) is True


def test_tag_is_done_status_done_false(tmp_path):
    """status.json with done=False → NOT done (e.g., interrupted run)."""
    status = {"done": False, "failed": False}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    assert _tag_is_done(str(tmp_path)) is False


def test_tag_is_done_status_failed(tmp_path):
    """A failed status (done=False, failed=True) is also not 'done'."""
    status = {"done": False, "failed": True, "last_error": "OOM"}
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    assert _tag_is_done(str(tmp_path)) is False


def test_tag_is_done_corrupted_status_returns_false(tmp_path):
    """Malformed status.json is treated as 'not done' (graceful degradation)."""
    (tmp_path / "status.json").write_text("not valid json{{{", encoding="utf-8")
    assert _tag_is_done(str(tmp_path)) is False


def test_tag_is_done_summary_takes_precedence(tmp_path):
    """summary.json is the strongest 'done' signal — checked before status.json."""
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "status.json").write_text(
        json.dumps({"done": False, "failed": True}), encoding="utf-8",
    )
    assert _tag_is_done(str(tmp_path)) is True


def test_tag_is_done_nonexistent_dir(tmp_path):
    """A path that doesn't exist is not done (no exception)."""
    bogus = tmp_path / "never_created"
    assert _tag_is_done(str(bogus)) is False


# ------------------------ execute_one_combo skip path ------------------------


def test_execute_one_combo_resume_skip(tmp_path):
    """When resume=True and the report dir already has summary.json,
    execute_one_combo returns status='skip' immediately, without running
    a kfold experiment."""
    reports_dir = tmp_path / "reports"
    runs_dir = tmp_path / "runs"
    tag = "existing_combo"
    rep_dir = reports_dir / tag
    rep_dir.mkdir(parents=True)
    # Pre-populate summary so the combo is considered done
    (rep_dir / "summary.json").write_text(
        json.dumps({"val_mean": {"accuracy": 0.9}, "objective_name": "accuracy"}),
        encoding="utf-8",
    )

    # cfg is irrelevant since execute should short-circuit before reading it
    cfg = {"dataset": {"name": "fake"}, "feature_map": {"type": "amplitude"}}
    res = execute_one_combo(
        cfg=cfg, tag=tag, reports_dir=str(reports_dir), runs_dir=str(runs_dir),
        resume=True,
    )
    assert res["status"] == "skip"
    assert res["tag"] == tag
    # The run_dir should be empty string when skipped (nothing was written)
    assert res["run_dir"] == ""
    # No timing recorded because no execution happened
    assert res["wall_time_seconds"] is None
    assert res["peak_memory_mb"] is None


def test_execute_one_combo_resume_false_does_not_skip(tmp_path):
    """When resume=False, an existing summary.json should NOT cause a skip;
    instead the combo would attempt to re-run. We don't run a real kfold
    here — just verify it does NOT return 'skip' immediately."""
    reports_dir = tmp_path / "reports"
    runs_dir = tmp_path / "runs"
    tag = "existing_combo"
    rep_dir = reports_dir / tag
    rep_dir.mkdir(parents=True)
    (rep_dir / "summary.json").write_text("{}", encoding="utf-8")

    # cfg here is intentionally minimal/broken so a real run would fail; we
    # just want to assert the code path is NOT 'skip'.
    cfg = {
        "dataset": {"name": "fake", "path": str(tmp_path / "missing.csv"),
                    "target": "y"},
        "feature_map": {"type": "amplitude"},
        "ansatz": {"type": "ansatz_1", "params": {"depth": 1}},
        "device": {"wires": 2, "shots": None},
        "optimizer": {"type": "adam",
                      "classical": {"lr": 0.1, "epochs": 1, "batch_size": 4}},
    }
    res = execute_one_combo(
        cfg=cfg, tag=tag, reports_dir=str(reports_dir), runs_dir=str(runs_dir),
        resume=False,
    )
    # Real run failed (no real data file) — status should be "fail", not "skip"
    assert res["status"] == "fail"
    # And wall_time should have been measured (we did try to execute)
    assert res["wall_time_seconds"] is not None
    assert res["wall_time_seconds"] >= 0.0
