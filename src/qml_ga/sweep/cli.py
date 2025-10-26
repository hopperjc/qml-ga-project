import os, glob, json, argparse, itertools, traceback
from typing import Dict, List
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
    return f"{ds}__{fm}__{at}_d{d}__{opt}__w{w}" + (("__" + "_".join(extras)) if extras else ""

)

def _save_report(rep_dir: str, cfg: Dict, summary: Dict):
    ensure_dirs(rep_dir)
    with open(os.path.join(rep_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame(summary.get("folds", [])).to_csv(os.path.join(rep_dir, "folds.csv"), index=False)
    write_yaml(cfg, os.path.join(rep_dir, "config.yaml"))

def _append_index(reports_dir: str, row: Dict):
    index_csv = os.path.join(reports_dir, "index.csv")
    hdr = not os.path.exists(index_csv)
    pd.DataFrame([row]).to_csv(index_csv, mode="a", header=hdr, index=False)

def _iter_classical(base_cfg: Dict, opt_type: str, grid: Dict):
    keys = sorted(grid.keys()); vals = [grid[k] for k in keys]
    for comb in itertools.product(*vals):
        params = dict(zip(keys, comb))
        cfg = json.loads(json.dumps(base_cfg))
        cfg["optimizer"] = {"type": opt_type, "classical": params}
        yield cfg, params

def _iter_ga(base_cfg: Dict, grid: Dict):
    keys = sorted(grid.keys()); vals = [grid[k] for k in keys]
    for comb in itertools.product(*vals):
        params = dict(zip(keys, comb))
        cfg = json.loads(json.dumps(base_cfg))
        cfg["optimizer"] = {"type": "ga", "ga": params}
        yield cfg, params

def main(
    datasets_dir="configs/datasets", feature_maps_dir="configs/feature_maps",
    ansatz_dir="configs/ansatz", ga_grid_path="configs/hypergrids/ga.yaml",
    classical_grid_path="configs/hypergrids/classical.yaml",
    reports_dir="reports", runs_dir="runs",
    include_feature_maps="amplitude,zz", max_wires=12, dry_run=False,
    trace=False, abort_on_fail=False, limit: int = 0,
):
    setup_cwd_to_repo_root()
    ensure_dirs(reports_dir, runs_dir)

    DS = [_load(p) for p in _list_yaml(datasets_dir)]
    FM = [_load(p) for p in _list_yaml(feature_maps_dir) if load_yaml(p)["feature_map"]["type"] in [s.strip() for s in include_feature_maps.split(",")]]
    AN = [_load(p) for p in _list_yaml(ansatz_dir)]
    GA_GRID = load_yaml(ga_grid_path)["ga"]
    CL_GRID = load_yaml(classical_grid_path)["classical"]

    count = 0
    for ds in DS:
        for fm in FM:
            try:
                dev = _derive_device(ds, fm)
            except Exception as e:
                print(f"[SKIP] derive_device ({ds['_file']},{fm['_file']}): {e}"); continue
            if dev["device"]["wires"] > max_wires:
                print(f"[SKIP] wires={dev['device']['wires']}>max={max_wires}"); continue

            for an in AN:
                base_cfg = {
                    "dataset": ds["dataset"], "feature_map": fm["feature_map"],
                    "ansatz": an["ansatz"], "device": dev["device"],
                    "seed": 42, "output": {"base_dir": runs_dir, "save_specs": False},
                }

                # Clássicos
                for opt_type, grid in [("adam", CL_GRID["adam"]), ("nesterov", CL_GRID["nesterov"])]:
                    for cfgC, params in _iter_classical(base_cfg, opt_type, grid):
                        tag = _tag(cfgC, classical_params=params)
                        if dry_run:
                            print("[DRY]", tag)
                            count += 1
                            if limit and count >= limit: return
                            continue
                        run_dir = make_run_dir(runs_dir); write_yaml(cfgC, os.path.join(run_dir, "config_snapshot.yaml"))
                        try:
                            summary = run_kfold_experiment(cfgC, run_dir)
                        except Exception as e:
                            print(f"[FAIL] {tag}: {e}")
                            if trace: print(traceback.format_exc())
                            if abort_on_fail: raise
                            count += 1
                            if limit and count >= limit: return
                            continue
                        rep_dir = os.path.join(reports_dir, tag); _save_report(rep_dir, cfgC, summary)
                        _append_index(reports_dir, {
                            "tag": tag, "run_dir": run_dir,
                            "dataset": cfgC["dataset"]["name"], "feature_map": cfgC["feature_map"]["type"],
                            "ansatz": cfgC["ansatz"]["type"], "depth": cfgC["ansatz"].get("params",{}).get("depth",""),
                            "optimizer": opt_type, "params": json.dumps(params),
                            "device_wires": cfgC["device"]["wires"],
                            "mean_accuracy": summary.get("mean_accuracy"),
                            "std_accuracy": summary.get("std_accuracy"),
                            "objective_name": summary.get("objective_name",""),
                        })
                        print(f"[OK] {tag} | mean_acc={summary.get('mean_accuracy'):.4f}")
                        count += 1
                        if limit and count >= limit: return

                # GA
                for cfgG, ga_params in _iter_ga(base_cfg, GA_GRID):
                    tag = _tag(cfgG, ga_params=ga_params)
                    if dry_run:
                        print("[DRY]", tag)
                        count += 1
                        if limit and count >= limit: return
                        continue
                    run_dir = make_run_dir(runs_dir); write_yaml(cfgG, os.path.join(run_dir, "config_snapshot.yaml"))
                    try:
                        summary = run_kfold_experiment(cfgG, run_dir)
                    except Exception as e:
                        print(f"[FAIL] {tag}: {e}")
                        if trace: print(traceback.format_exc())
                        if abort_on_fail: raise
                        count += 1
                        if limit and count >= limit: return
                        continue
                    rep_dir = os.path.join(reports_dir, tag); _save_report(rep_dir, cfgG, summary)
                    _append_index(reports_dir, {
                        "tag": tag, "run_dir": run_dir,
                        "dataset": cfgG["dataset"]["name"], "feature_map": cfgG["feature_map"]["type"],
                        "ansatz": cfgG["ansatz"]["type"], "depth": cfgG["ansatz"].get("params",{}).get("depth",""),
                        "optimizer": "ga", "params": json.dumps(ga_params),
                        "device_wires": cfgG["device"]["wires"],
                        "mean_accuracy": summary.get("mean_accuracy"),
                        "std_accuracy": summary.get("std_accuracy"),
                        "objective_name": summary.get("objective_name",""),
                    })
                    print(f"[OK] {tag} | mean_acc={summary.get('mean_accuracy'):.4f}")
                    count += 1
                    if limit and count >= limit: return

def entrypoint():
    ap = argparse.ArgumentParser(description="Varredura de todas as combinações + grade do GA e clássicos.")
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
    args = ap.parse_args()
    main(**vars(args))

if __name__ == "__main__":
    entrypoint()
