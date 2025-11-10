import os, glob, json, argparse, traceback, time
from typing import Dict, List, Optional
import pandas as pd

from qml_ga.utils.io import load_yaml, write_yaml, make_run_dir, ensure_dirs, setup_cwd_to_repo_root
from qml_ga.data.datamodule import DataModule
from qml_ga.experiments.kfold_train import run_kfold_experiment

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

def _append_index(reports_dir: str, row: Dict):
    parts_dir = os.path.join(reports_dir, "_index_parts")
    ensure_dirs(parts_dir)
    jid = os.getenv("SLURM_JOB_ID", "nojid")
    aid = os.getenv("SLURM_ARRAY_TASK_ID", "noaid")
    part_path = os.path.join(parts_dir, f"index.{jid}.{aid}.csv")
    hdr = not os.path.exists(part_path)
    pd.DataFrame([row]).to_csv(part_path, mode="a", header=hdr, index=False)

def _product_size(grid: Dict[str, List]) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n

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

def _extract_val_mean_accuracy(summary: Dict) -> Optional[float]:
    if isinstance(summary.get("val_mean"), dict) and "accuracy" in summary["val_mean"]:
        return float(summary["val_mean"]["accuracy"])
    if "mean_accuracy" in summary and summary["mean_accuracy"] is not None:
        return float(summary["mean_accuracy"])
    return None

def _extract_val_std_accuracy(summary: Dict) -> Optional[float]:
    if isinstance(summary.get("val_std"), dict) and "accuracy" in summary["val_std"]:
        return float(summary["val_std"]["accuracy"])
    if "std_accuracy" in summary and summary["std_accuracy"] is not None:
        return float(summary["std_accuracy"])
    return None

def _fmt_acc(val: Optional[float]) -> str:
    try:
        return f"{float(val):.4f}"
    except Exception:
        return "nan"

def _auto_shard_from_env(shard_index: Optional[int], shard_total: Optional[int]):
    if shard_index is None:
        sid = os.getenv("SLURM_ARRAY_TASK_ID")
        if sid is not None:
            shard_index = int(sid)
    if shard_total is None:
        stc = os.getenv("SLURM_ARRAY_TASK_COUNT")
        if stc is not None:
            shard_total = int(stc)
    return shard_index, shard_total

def _belongs_to_shard(combo_idx_1based: int, shard_index: Optional[int], shard_total: Optional[int]) -> bool:
    if shard_total is None or shard_total <= 1 or shard_index is None:
        return True
    return ((combo_idx_1based - 1) % shard_total) == shard_index

def main(
    datasets_dir="configs/datasets", feature_maps_dir="configs/feature_maps",
    ansatz_dir="configs/ansatz", ga_grid_path="configs/hypergrids/ga.yaml",
    classical_grid_path="configs/hypergrids/classical.yaml",
    reports_dir="reports", runs_dir="runs",
    include_feature_maps="amplitude,zz", max_wires=12, dry_run=False,
    trace=False, abort_on_fail=False, limit: int = 0,
    shard_index: Optional[int] = None, shard_total: Optional[int] = None,
    print_total_only: bool = False,
):
    setup_cwd_to_repo_root()
    ensure_dirs(reports_dir, runs_dir)

    DS = [_load(p) for p in _list_yaml(datasets_dir)]
    FM = [_load(p) for p in _list_yaml(feature_maps_dir) if load_yaml(p)["feature_map"]["type"] in [s.strip() for s in include_feature_maps.split(",")]]
    AN = [_load(p) for p in _list_yaml(ansatz_dir)]
    GA_GRID = load_yaml(ga_grid_path)["ga"]
    CL_GRID = load_yaml(classical_grid_path)["classical"]

    total_combos = 0
    for ds in DS:
        for fm in FM:
            try:
                dev = _derive_device(ds, fm)
            except Exception:
                continue
            if dev["device"]["wires"] > max_wires:
                continue
            for _ in AN:
                total_combos += _product_size(CL_GRID["adam"])
                total_combos += _product_size(CL_GRID["nesterov"])
                total_combos += _product_size(GA_GRID)

    if print_total_only:
        print(total_combos)
        return

    shard_index, shard_total = _auto_shard_from_env(shard_index, shard_total)

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[SWEEP] start={started_at} total_combos={total_combos}", flush=True)
    print(f"[SWEEP] include_feature_maps={include_feature_maps} max_wires={max_wires} dry_run={dry_run} abort_on_fail={abort_on_fail} trace={trace} limit={limit}", flush=True)
    print(f"[SWEEP] shards: index={shard_index} total={shard_total}", flush=True)

    combo_idx = 0
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
                        combo_idx += 1
                        if limit and combo_idx > limit:
                            return
                        if not _belongs_to_shard(combo_idx, shard_index, shard_total):
                            continue

                        tag = _tag(cfgC, classical_params=params)
                        prefix = f"[{combo_idx}/{total_combos}] {tag}"
                        if dry_run:
                            print(f"{prefix} [DRY]", flush=True)
                            continue

                        run_dir = make_run_dir(runs_dir)
                        write_yaml(cfgC, os.path.join(run_dir, "config_snapshot.yaml"))

                        rep_dir = os.path.join(reports_dir, tag)
                        ensure_dirs(rep_dir)
                        write_yaml(cfgC, os.path.join(rep_dir, "config.yaml"))

                        print(f"{prefix} [START]", flush=True)
                        try:
                            summary = run_kfold_experiment(
                                cfgC, run_dir,
                                report_dir=rep_dir,
                                progress=True,
                                progress_prefix=prefix
                            )
                        except Exception as e:
                            print(f"{prefix} [FAIL] {e}", flush=True)
                            if trace:
                                print(traceback.format_exc(), flush=True)
                            try:
                                with open(os.path.join(rep_dir, "status.json"), "r+", encoding="utf-8") as f:
                                    st = json.load(f)
                                    st["failed"] = True
                                    f.seek(0); json.dump(st, f, ensure_ascii=False, indent=2); f.truncate()
                            except Exception:
                                pass
                            if abort_on_fail:
                                raise
                            continue

                        mean_acc = _extract_val_mean_accuracy(summary)
                        std_acc = _extract_val_std_accuracy(summary)
                        print(f"{prefix} [OK] val_mean_acc={_fmt_acc(mean_acc)}", flush=True)

                        _append_index(reports_dir, {
                            "tag": tag, "run_dir": run_dir,
                            "dataset": cfgC["dataset"]["name"], "feature_map": cfgC["feature_map"]["type"],
                            "ansatz": cfgC["ansatz"]["type"], "depth": cfgC["ansatz"].get("params",{}).get("depth",""),
                            "optimizer": opt_type, "params": json.dumps(params),
                            "device_wires": cfgC["device"]["wires"],
                            "mean_accuracy": mean_acc,
                            "std_accuracy": std_acc,
                            "objective_name": summary.get("objective_name",""),
                        })

                for cfgG, ga_params in _iter_ga(base_cfg, GA_GRID):
                    combo_idx += 1
                    if limit and combo_idx > limit:
                        return
                    if not _belongs_to_shard(combo_idx, shard_index, shard_total):
                        continue

                    tag = _tag(cfgG, ga_params=ga_params)
                    prefix = f"[{combo_idx}/{total_combos}] {tag}"
                    if dry_run:
                        print(f"{prefix} [DRY]", flush=True)
                        continue

                    run_dir = make_run_dir(runs_dir)
                    write_yaml(cfgG, os.path.join(run_dir, "config_snapshot.yaml"))

                    rep_dir = os.path.join(reports_dir, tag)
                    ensure_dirs(rep_dir)
                    write_yaml(cfgG, os.path.join(rep_dir, "config.yaml"))

                    print(f"{prefix} [START]", flush=True)
                    try:
                        summary = run_kfold_experiment(
                            cfgG, run_dir,
                            report_dir=rep_dir,
                            progress=True,
                            progress_prefix=prefix
                        )
                    except Exception as e:
                        print(f"{prefix} [FAIL] {e}", flush=True)
                        if trace:
                            print(traceback.format_exc(), flush=True)
                        try:
                            with open(os.path.join(rep_dir, "status.json"), "r+", encoding="utf-8") as f:
                                st = json.load(f)
                                st["failed"] = True
                                f.seek(0); json.dump(st, f, ensure_ascii=False, indent=2); f.truncate()
                        except Exception:
                            pass
                        if abort_on_fail:
                            raise
                        continue

                    mean_acc = _extract_val_mean_accuracy(summary)
                    std_acc = _extract_val_std_accuracy(summary)
                    print(f"{prefix} [OK] val_mean_acc={_fmt_acc(mean_acc)}", flush=True)

                    _append_index(reports_dir, {
                        "tag": tag, "run_dir": run_dir,
                        "dataset": cfgG["dataset"]["name"], "feature_map": cfgG["feature_map"]["type"],
                        "ansatz": cfgG["ansatz"]["type"], "depth": cfgG["ansatz"].get("params",{}).get("depth",""),
                        "optimizer": "ga", "params": json.dumps(ga_params),
                        "device_wires": cfgG["device"]["wires"],
                        "mean_accuracy": mean_acc,
                        "std_accuracy": std_acc,
                        "objective_name": summary.get("objective_name",""),
                    })

def entrypoint():
    ap = argparse.ArgumentParser(description="Sweep com sharding SLURM e gravação incremental.")
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
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--abort_on_fail", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard_index", type=int, default=None)
    ap.add_argument("--shard_total", type=int, default=None)
    ap.add_argument("--print_total_only", action="store_true")
    args = ap.parse_args()
    main(**vars(args))

if __name__ == "__main__":
    entrypoint()
