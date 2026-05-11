"""Parallel runner for QML-GA sweeps using ProcessPoolExecutor.

Each task = one (dataset × feature_map × ansatz × optimizer × hyperparams)
combination. Tasks are independent: each creates its own run_dir, writes a
config snapshot, calls run_kfold_experiment, and appends wall-time + peak
memory to the resulting summary.json.

The sweep CLI builds the list of tasks and dispatches them here either serially
(workers <= 1) or in parallel (workers > 1).
"""
from __future__ import annotations

import json
import os
import time
import traceback
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from qml_ga.utils.io import ensure_dirs, make_run_dir, write_yaml
from qml_ga.experiments.kfold_train import run_kfold_experiment


Combo = Tuple[Dict[str, Any], str, str, Dict[str, Any]]
"""(cfg, tag, opt_type, opt_params) — opt_type and opt_params are passed
through to on_result for index logging."""


def _tag_is_done(rep_dir: str) -> bool:
    if os.path.exists(os.path.join(rep_dir, "summary.json")):
        return True
    sst = os.path.join(rep_dir, "status.json")
    if os.path.exists(sst):
        try:
            with open(sst, "r", encoding="utf-8") as f:
                return bool(json.load(f).get("done", False))
        except Exception:
            return False
    return False


def execute_one_combo(
    cfg: Dict[str, Any],
    tag: str,
    reports_dir: str,
    runs_dir: str,
    resume: bool = True,
    progress_prefix: str = "",
) -> Dict[str, Any]:
    """Run one combination end-to-end with timing + memory instrumentation.

    Picklable / safe for ProcessPoolExecutor: all state derives from arguments
    and is persisted to disk. Returns a dict with status in {"ok", "skip", "fail"}.
    """
    rep_dir = os.path.join(reports_dir, tag)
    if resume and _tag_is_done(rep_dir):
        return {
            "status": "skip", "tag": tag, "run_dir": "",
            "summary": None, "error": None,
            "wall_time_seconds": None, "peak_memory_mb": None,
        }

    run_dir = make_run_dir(runs_dir)
    write_yaml(cfg, os.path.join(run_dir, "config_snapshot.yaml"))
    ensure_dirs(rep_dir)
    write_yaml(cfg, os.path.join(rep_dir, "config.yaml"))

    tracemalloc.start()
    t0 = time.perf_counter()
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status = "ok"
    try:
        summary = run_kfold_experiment(
            cfg, run_dir,
            report_dir=rep_dir,
            progress=False,  # parallel path: per-fold progress is muted
            progress_prefix=progress_prefix,
            resume=resume,
        )
    except Exception as e:
        status = "fail"
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        try:
            with open(os.path.join(rep_dir, "status.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"done": False, "failed": True, "last_error": str(e)},
                    f, ensure_ascii=False, indent=2,
                )
        except Exception:
            pass
    finally:
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)

    if summary is not None:
        summary["wall_time_seconds"] = wall
        summary["peak_memory_mb"] = peak_mb
        try:
            with open(os.path.join(rep_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

    return {
        "status": status, "tag": tag, "run_dir": run_dir,
        "summary": summary, "error": error,
        "wall_time_seconds": wall, "peak_memory_mb": peak_mb,
    }


def run_combos(
    combos: List[Combo],
    reports_dir: str,
    runs_dir: str,
    workers: int = 1,
    resume: bool = True,
    on_result: Optional[Callable[[int, Combo, Dict[str, Any]], None]] = None,
    abort_on_fail: bool = False,
) -> None:
    """Dispatch combos serially (workers<=1) or in parallel (workers>1).

    `on_result(idx, combo, result)` runs in the orchestrator process and is the
    point where the caller appends rows to the index CSV / updates sweep status.
    """
    total = len(combos)
    if workers <= 1:
        for idx, combo in enumerate(combos, start=1):
            cfg, tag, _opt_type, _opt_params = combo
            prefix = f"[{idx}/{total}] {tag}"
            print(f"{prefix} [START]", flush=True)
            res = execute_one_combo(cfg, tag, reports_dir, runs_dir, resume=resume,
                                    progress_prefix=prefix)
            _print_result(prefix, res)
            if on_result:
                on_result(idx, combo, res)
            if abort_on_fail and res["status"] == "fail":
                raise RuntimeError(f"abort_on_fail: combo {tag} failed")
        return

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for idx, combo in enumerate(combos, start=1):
            cfg, tag, _opt_type, _opt_params = combo
            prefix = f"[{idx}/{total}] {tag}"
            fut = ex.submit(execute_one_combo, cfg, tag, reports_dir, runs_dir, resume, prefix)
            futures[fut] = (idx, combo, prefix)

        for fut in as_completed(futures):
            idx, combo, prefix = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "status": "fail", "tag": combo[1], "run_dir": "",
                    "summary": None, "error": str(e),
                    "wall_time_seconds": None, "peak_memory_mb": None,
                }
            _print_result(prefix, res)
            if on_result:
                on_result(idx, combo, res)
            if abort_on_fail and res["status"] == "fail":
                ex.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"abort_on_fail: combo {combo[1]} failed")


def _print_result(prefix: str, res: Dict[str, Any]) -> None:
    status = res["status"]
    if status == "ok":
        s = res.get("summary") or {}
        acc: Optional[float] = None
        if isinstance(s.get("val_mean"), dict):
            acc = s["val_mean"].get("accuracy")
        elif s.get("mean_accuracy") is not None:
            acc = s.get("mean_accuracy")
        try:
            acc_str = f"{float(acc):.4f}" if acc is not None else "nan"
        except Exception:
            acc_str = "nan"
        wall = res.get("wall_time_seconds") or 0.0
        mem = res.get("peak_memory_mb") or 0.0
        print(f"{prefix} [OK] val_mean_acc={acc_str} time={wall:.1f}s mem={mem:.0f}MB", flush=True)
    elif status == "skip":
        print(f"{prefix} [SKIP DONE]", flush=True)
    elif status == "fail":
        first_line = (res.get("error") or "?").splitlines()[0]
        print(f"{prefix} [FAIL] {first_line}", flush=True)
