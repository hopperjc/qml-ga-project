import os, glob, json, argparse, time
from typing import Dict, List, Optional

from qml_ga.utils.io import (
    load_yaml, ensure_dirs, setup_cwd_to_repo_root, make_run_id,
)
from qml_ga.utils.logger import append_csv_row
from qml_ga.data.datamodule import DataModule
from qml_ga.sweep.runner import Combo, run_combos


def _list_yaml(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.yaml")))


def _load(path: str) -> Dict:
    d = load_yaml(path); d["_file"] = path; return d


def _derive_device(dataset_block: Dict, feature_map_block: Dict, device_name="default.qubit") -> Dict:
    dm = DataModule(dataset_block["dataset"])
    X, y, feats = dm.load_all()
    fm_type = feature_map_block["feature_map"]["type"]
    wires = DataModule.required_wires(fm_type, len(feats))
    return {"device": {"name": device_name, "wires": wires, "shots": None}}


def _tag(cfg: Dict, ga_params: Dict = None, classical_params: Dict = None) -> str:
    ds = cfg["dataset"]["name"]; fm = cfg["feature_map"]["type"]
    at = cfg["ansatz"]["type"]; d = cfg["ansatz"].get("params", {}).get("depth", "")
    opt = cfg["optimizer"]["type"]; w = cfg["device"]["wires"]
    extras = []
    if ga_params:
        for k in ["population_size","selection_type","crossover_type","mutation_type","num_generations"]:
            if k in ga_params: extras.append(f"{k[:3]}={ga_params[k]}")
    if classical_params:
        if "epochs" in classical_params: extras.append(f"ep={classical_params['epochs']}")
        if "lr" in classical_params: extras.append(f"lr={classical_params['lr']}")
    return f"{ds}__{fm}__{at}_d{d}__{opt}__w{w}" + (("__" + "_".join(extras)) if extras else "")


def _iter_classical(base_cfg: Dict, opt_type: str, grid: Dict):
    import itertools, json as _json
    keys = sorted(grid.keys()); vals = [grid[k] for k in keys]
    for comb in itertools.product(*vals):
        params = dict(zip(keys, comb))
        cfg = _json.loads(_json.dumps(base_cfg))
        cfg["optimizer"] = {"type": opt_type, "classical": params}
        yield cfg, params


def _iter_ga(base_cfg: Dict, grid: Dict):
    import itertools, json as _json
    keys = sorted(grid.keys()); vals = [grid[k] for k in keys]
    for comb in itertools.product(*vals):
        params = dict(zip(keys, comb))
        cfg = _json.loads(_json.dumps(base_cfg))
        cfg["optimizer"] = {"type": "ga", "ga": params}
        yield cfg, params


def _belongs_to_shard(idx_1based: int, shard_index: Optional[int], shard_total: Optional[int]) -> bool:
    if shard_total is None or shard_total <= 1 or shard_index is None:
        return True
    return ((idx_1based - 1) % shard_total) == shard_index


def _build_combos(
    DS: List[Dict], FM: List[Dict], AN: List[Dict],
    GA_GRID: Dict, CL_GRID: Dict,
    runs_dir: str, max_wires: int,
) -> List[Combo]:
    """Build the full list of combinations (cfg, tag, opt_type, opt_params)."""
    combos: List[Combo] = []
    for ds in DS:
        for fm in FM:
            try:
                dev = _derive_device(ds, fm)
            except Exception as e:
                print(f"[SKIP] derive_device ({ds['_file']},{fm['_file']}): {e}", flush=True)
                continue
            if dev["device"]["wires"] > max_wires:
                print(f"[SKIP] wires={dev['device']['wires']}>max={max_wires}", flush=True)
                continue
            for an in AN:
                base_cfg = {
                    "dataset": ds["dataset"], "feature_map": fm["feature_map"],
                    "ansatz": an["ansatz"], "device": dev["device"],
                    "seed": 42, "output": {"base_dir": runs_dir, "save_specs": False},
                }
                for opt_type, grid in [("adam", CL_GRID["adam"]), ("nesterov", CL_GRID["nesterov"])]:
                    for cfgC, params in _iter_classical(base_cfg, opt_type, grid):
                        combos.append((cfgC, _tag(cfgC, classical_params=params), opt_type, params))
                for cfgG, ga_params in _iter_ga(base_cfg, GA_GRID):
                    combos.append((cfgG, _tag(cfgG, ga_params=ga_params), "ga", ga_params))
    return combos


def _make_index_appender(reports_dir: str, run_id: str, total_combos: int):
    """Return a closure suitable for runner.on_result that appends index rows
    and updates the sweep status JSON."""
    parts_dir = os.path.join(reports_dir, "_index_parts")
    ensure_dirs(parts_dir)
    part_path = os.path.join(parts_dir, f"index.{run_id}.csv")
    sweep_status_path = os.path.join(reports_dir, f"sweep_status.{run_id}.json")
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    state = {"done": 0, "skipped": 0, "failed": 0}

    def _write_status(last_tag: Optional[str]):
        with open(sweep_status_path, "w", encoding="utf-8") as f:
            json.dump({
                "started_at": started_at, "total_combos": total_combos,
                **state, "last_tag": last_tag,
                "last_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, ensure_ascii=False, indent=2)

    def _on_result(idx: int, combo: Combo, res: Dict):
        cfg, tag, opt_type, opt_params = combo
        if res["status"] == "skip":
            state["skipped"] += 1
            _write_status(tag)
            return
        if res["status"] == "fail":
            state["failed"] += 1
            _write_status(tag)
            return

        # status == "ok"
        state["done"] += 1
        s = res.get("summary") or {}
        mean_acc = None
        std_acc = None
        if isinstance(s.get("val_mean"), dict):
            mean_acc = s["val_mean"].get("accuracy")
        elif s.get("mean_accuracy") is not None:
            mean_acc = s.get("mean_accuracy")
        if isinstance(s.get("val_std"), dict):
            std_acc = s["val_std"].get("accuracy")
        elif s.get("std_accuracy") is not None:
            std_acc = s.get("std_accuracy")

        row = {
            "tag": tag, "run_dir": res.get("run_dir", ""),
            "dataset": cfg["dataset"]["name"], "feature_map": cfg["feature_map"]["type"],
            "ansatz": cfg["ansatz"]["type"], "depth": cfg["ansatz"].get("params", {}).get("depth", ""),
            "optimizer": opt_type, "params": json.dumps(opt_params),
            "device_wires": cfg["device"]["wires"],
            "mean_accuracy": mean_acc, "std_accuracy": std_acc,
            "objective_name": s.get("objective_name", ""),
            "wall_time_seconds": res.get("wall_time_seconds"),
            "peak_memory_mb": res.get("peak_memory_mb"),
        }
        append_csv_row(part_path, row)
        _write_status(tag)

    _write_status(None)
    return _on_result


def main(
    datasets_dir="configs/datasets", feature_maps_dir="configs/feature_maps",
    ansatz_dir="configs/ansatz", ga_grid_path="configs/hypergrids/ga.yaml",
    classical_grid_path="configs/hypergrids/classical.yaml",
    reports_dir="reports", runs_dir="runs",
    include_feature_maps="amplitude,zz", max_wires=12, dry_run=False,
    abort_on_fail=False, limit: int = 0,
    shard_index: Optional[int] = None, shard_total: Optional[int] = None,
    print_total_only: bool = False, resume: bool = True,
    workers: int = 1,
):
    setup_cwd_to_repo_root()
    ensure_dirs(reports_dir, runs_dir)

    DS = [_load(p) for p in _list_yaml(datasets_dir)]
    FM_filter = {s.strip() for s in include_feature_maps.split(",")}
    FM = [_load(p) for p in _list_yaml(feature_maps_dir) if load_yaml(p)["feature_map"]["type"] in FM_filter]
    AN = [_load(p) for p in _list_yaml(ansatz_dir)]
    GA_GRID = load_yaml(ga_grid_path)["ga"]
    CL_GRID = load_yaml(classical_grid_path)["classical"]

    all_combos = _build_combos(DS, FM, AN, GA_GRID, CL_GRID, runs_dir, max_wires)
    total_combos = len(all_combos)

    if print_total_only:
        print(total_combos)
        return

    # Apply sharding + limit
    selected: List[Combo] = []
    for idx, combo in enumerate(all_combos, start=1):
        if limit and idx > limit:
            break
        if not _belongs_to_shard(idx, shard_index, shard_total):
            continue
        selected.append(combo)

    run_id = make_run_id()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SWEEP] run_id={run_id} start={started_at} total_combos={total_combos} selected={len(selected)}", flush=True)
    print(f"[SWEEP] include_feature_maps={include_feature_maps} max_wires={max_wires} dry_run={dry_run} "
          f"abort_on_fail={abort_on_fail} limit={limit} resume={resume} workers={workers}", flush=True)
    print(f"[SWEEP] shards: index={shard_index} total={shard_total}", flush=True)

    if dry_run:
        for idx, (_cfg, tag, _opt_type, _params) in enumerate(selected, start=1):
            print(f"[{idx}/{len(selected)}] {tag} [DRY]", flush=True)
        return

    on_result = _make_index_appender(reports_dir, run_id, total_combos)
    run_combos(
        selected, reports_dir=reports_dir, runs_dir=runs_dir,
        workers=workers, resume=resume, on_result=on_result,
        abort_on_fail=abort_on_fail,
    )


def entrypoint():
    ap = argparse.ArgumentParser(description="Sweep com sharding manual opcional, resume e gravação incremental.")
    ap.add_argument("--datasets_dir", default="configs/datasets")
    ap.add_argument("--feature_maps_dir", default="configs/feature_maps")
    ap.add_argument("--ansatz_dir", default="configs/ansatz")
    ap.add_argument("--ga_grid_path", default="configs/hypergrids/ga.yaml")
    ap.add_argument("--classical_grid_path", default="configs/hypergrids/classical.yaml")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--include_feature_maps", default="amplitude,zz")
    ap.add_argument("--max_wires", type=int, default=12)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--abort_on_fail", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard_index", type=int, default=None)
    ap.add_argument("--shard_total", type=int, default=None)
    ap.add_argument("--print_total_only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="ProcessPoolExecutor workers (1 = serial)")
    args = ap.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    entrypoint()
