"""Tests for qml_ga.utils.io and qml_ga.utils.logger."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import pytest

from qml_ga.utils.io import (
    ensure_dirs,
    load_yaml,
    make_run_dir,
    make_run_id,
    write_yaml,
)
from qml_ga.utils.logger import append_csv_row, make_text_logger


# ----------------------------- make_run_id -----------------------------


def test_make_run_id_format():
    """Run ID should follow YYYYMMDD_HHMMSS_<6 hex> pattern."""
    rid = make_run_id()
    assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{6}$", rid), f"unexpected format: {rid}"


def test_make_run_id_unique_on_rapid_calls():
    """Two consecutive calls produce distinct IDs (uuid suffix prevents collisions)."""
    a = make_run_id()
    b = make_run_id()
    assert a != b


def test_make_run_id_uuid_collision_resistance():
    """In 100 rapid calls, no duplicates."""
    ids = {make_run_id() for _ in range(100)}
    assert len(ids) == 100


# ------------------------------ ensure_dirs ------------------------------


def test_ensure_dirs_creates_single(tmp_path):
    target = tmp_path / "newdir" / "nested"
    ensure_dirs(str(target))
    assert target.is_dir()


def test_ensure_dirs_idempotent(tmp_path):
    """Calling twice on the same path is a no-op."""
    target = tmp_path / "dir"
    ensure_dirs(str(target))
    ensure_dirs(str(target))  # should not raise
    assert target.is_dir()


def test_ensure_dirs_multiple_at_once(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c" / "deep"
    ensure_dirs(str(a), str(b), str(c))
    for p in (a, b, c):
        assert p.is_dir()


# ------------------------------ make_run_dir ------------------------------


def test_make_run_dir_creates_timestamped_subfolder(tmp_path):
    run_dir = make_run_dir(str(tmp_path))
    p = Path(run_dir)
    assert p.is_dir()
    # Folder name follows YYYYMMDD_HHMMSS_run pattern
    assert re.match(r"^\d{8}_\d{6}_run$", p.name)


def test_make_run_dir_creates_base_if_missing(tmp_path):
    base = tmp_path / "new_base"
    run_dir = make_run_dir(str(base))
    assert Path(run_dir).is_dir()
    assert base.is_dir()


# --------------------------- yaml roundtrip ---------------------------


def test_yaml_roundtrip_simple(tmp_path):
    """write → load returns equivalent dict."""
    data = {"a": 1, "b": "two", "c": [1, 2, 3], "d": {"nested": True}}
    p = tmp_path / "out.yaml"
    write_yaml(data, str(p))
    loaded = load_yaml(str(p))
    assert loaded == data


def test_yaml_roundtrip_preserves_unicode(tmp_path):
    """YAML must preserve non-ASCII (Portuguese accents, special chars)."""
    data = {"título": "Avaliação à varredura", "γ": "gamma"}
    p = tmp_path / "out.yaml"
    write_yaml(data, str(p))
    loaded = load_yaml(str(p))
    assert loaded == data


def test_yaml_write_creates_parent_dirs(tmp_path):
    """write_yaml should create any missing parent directories."""
    p = tmp_path / "deep" / "nested" / "file.yaml"
    write_yaml({"key": "value"}, str(p))
    assert p.is_file()


# ------------------------------ append_csv_row ------------------------------


def test_append_csv_row_creates_with_header(tmp_path):
    p = tmp_path / "log.csv"
    append_csv_row(str(p), {"a": 1, "b": "x"})
    df = pd.read_csv(p)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1
    assert df.iloc[0]["a"] == 1
    assert df.iloc[0]["b"] == "x"


def test_append_csv_row_appends_without_duplicate_header(tmp_path):
    p = tmp_path / "log.csv"
    append_csv_row(str(p), {"a": 1, "b": "x"})
    append_csv_row(str(p), {"a": 2, "b": "y"})
    append_csv_row(str(p), {"a": 3, "b": "z"})
    df = pd.read_csv(p)
    assert len(df) == 3
    assert list(df["a"]) == [1, 2, 3]
    assert list(df["b"]) == ["x", "y", "z"]
    # Header appears exactly once — pd.read_csv would fail if header were duplicated as data row


# ------------------------------ make_text_logger ------------------------------


def test_make_text_logger_writes_file(tmp_path):
    log_path = tmp_path / "test.log"
    logger = make_text_logger(str(log_path), prefix="[test] ")
    logger("first message")
    logger("second message")
    content = log_path.read_text(encoding="utf-8")
    assert "first message" in content
    assert "second message" in content
    assert "[test]" in content


def test_make_text_logger_timestamps_present(tmp_path):
    log_path = tmp_path / "test.log"
    logger = make_text_logger(str(log_path))
    logger("hello")
    content = log_path.read_text(encoding="utf-8")
    # Format: [YYYY-MM-DD HH:MM:SS] ...
    assert re.search(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]", content)
