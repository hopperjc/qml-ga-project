#!/usr/bin/env python3
import os, sys, glob, argparse, math, yaml, csv
from pathlib import Path

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_yaml(dirpath):
    return sorted(glob.glob(os.path.join(dirpath, "*.yaml")))

def count_features_from_dataset_yaml(ds_yaml):
    ds = load_yaml(ds_yaml)
    csv_path = ds["dataset"]["path"]
    target = ds["dataset"]["target"]
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=5)
    return len([c for c in df.columns if c != target])

def infer_wires(fm_type, n_features):
    if fm_type.lower() == "amplitude":
        return int(math.ceil(math.log2(max(1, n_features))))
    return int(n_features)

def ansatz_meta_from_yaml(path):
    y = load_yaml(path)
    t = y["ansatz"]["type"]
    d = int(y["ansatz"]["params"]["depth"])
    return t, d

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--outdir", default="local/plan")
    ap.add_argument("--include_feature_maps", nargs="*", default=[])
    ap.add_argument("--max_wires", type=int, default=12)
    ap.add_argument("--max_total", type=int, default=0)
    # smoke e overrides
    ap.add_argument("--smoke", action="store_true", help="1 fold; 5 epocas/geracoes; checkpoint na epoca 3")
    ap.add_argument("--k_folds", type=int, default=None)
    ap.add_argument("--classical_epochs", type=int, default=None)
    ap.add_argument("--ga_generations", type=int, default=None)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    ds_dir = repo / "configs" / "datasets"
    fm_dir = repo / "configs" / "feature_maps"
    az_dir = repo / "configs" / "ansatz"
    hg_dir = repo / "configs" / "hypergrids"

    ds_list = list_yaml(str(ds_dir))
    fm_list = list_yaml(str(fm_dir))
    az_list = list_yaml(str(az_dir))
    if args.include_feature_maps:
        fm_list = [p for p in fm_list if Path(p).stem in args.include_feature_maps]

    hg_class = load_yaml(str(hg_dir / "classical.yaml"))["classical"]
    hg_ga = load_yaml(str(hg_dir / "ga.yaml"))["ga"]

    outdir = Path(args.outdir).resolve()
    cfgdir = outdir / "configs"
    ensure_dir(outdir); ensure_dir(cfgdir)

    runlist_path = outdir / "runlist.txt"
    plan_csv = outdir / "plan.csv"

    kfolds = 1 if args.smoke else 5
    if args.k_folds is not None: kfolds = args.k_folds
    cls_epochs_default = 5 if args.smoke else None
    ga_gens_default = 5 if args.smoke else None

    total = 0; idx = 0
    with open(runlist_path, "w", encoding="utf-8") as rl, open(plan_csv, "w", newline="", encoding="utf-8") as pcsv:
        w = csv.writer(pcsv); w.writerow(["idx", "tag", "config_yaml"])

        for ds_yaml in ds_list:
            n_feats = count_features_from_dataset_yaml(ds_yaml)
            ds_name = Path(ds_yaml).stem

            for fm_yaml in fm_list:
                fm = load_yaml(fm_yaml)["feature_map"]
                fm_type = fm["type"]
                wires = infer_wires(fm_type, n_features=n_feats)
                if wires > args.max_wires: continue

                for az_yaml in az_list:
                    az_type, depth = ansatz_meta_from_yaml(az_yaml)

                    # CLÁSSICOS
                    for opt_name, opt_grid in hg_class.items():
                        lrs = opt_grid.get("lr", [0.05])
                        eps = opt_grid.get("epochs", [200])
                        bss = opt_grid.get("batch_size", [16])

                        if cls_epochs_default is not None: eps = [cls_epochs_default]
                        if args.classical_epochs is not None: eps = [args.classical_epochs]

                        for lr in lrs:
                            for epochs in eps:
                                for bs in bss:
                                    idx += 1
                                    tag = f"{ds_name}__{fm_type}__{az_type}_d{depth}__{opt_name}__w{wires}__ep={epochs}_lr={lr}"
                                    merged = {
                                        "dataset": load_yaml(ds_yaml)["dataset"],
                                        "feature_map": load_yaml(fm_yaml)["feature_map"],
                                        "ansatz": load_yaml(az_yaml)["ansatz"],
                                        "device": {"name": "default.qubit", "wires": wires, "shots": None},
                                        "optimizer": {
                                            "type": opt_name,
                                            "classical": {"lr": float(lr), "epochs": int(epochs), "batch_size": int(bss[0])}
                                        },
                                        "k_folds": int(kfolds),
                                        "objective_name": "accuracy",
                                        "mid_checkpoint_epoch": 3 if args.smoke else 100,
                                    }
                                    cfg_path = cfgdir / f"{idx:06d}.yaml"
                                    with open(cfg_path, "w", encoding="utf-8") as f:
                                        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)
                                    cmd = f"qmlga-single --config {cfg_path} --resume --progress"
                                    rl.write(cmd + "\n")
                                    w.writerow([idx, tag, str(cfg_path)])
                                    total += 1
                                    if args.max_total and total >= args.max_total: print(total); return

                    # GA
                    from itertools import product
                    grid = {
                        "population_size": hg_ga.get("population_size", [50]),
                        "num_generations": hg_ga.get("num_generations", [100]),
                        "selection_type": hg_ga.get("selection_type", ["tournament"]),
                        "crossover_type": hg_ga.get("crossover_type", ["single_point"]),
                        "mutation_type": hg_ga.get("mutation_type", ["random"]),
                        "mutation_percent_genes": hg_ga.get("mutation_percent_genes", [10]),
                        "elitism": hg_ga.get("elitism", [2]),
                        "init_range_low": hg_ga.get("init_range_low", [-3.14]),
                        "init_range_high": hg_ga.get("init_range_high", [3.14]),
                    }
                    if ga_gens_default is not None:
                        grid["num_generations"] = [ga_gens_default]
                    if args.ga_generations is not None:
                        grid["num_generations"] = [int(args.ga_generations)]

                    keys = list(grid.keys())
                    for values in product(*[grid[k] for k in keys]):
                        params = dict(zip(keys, values))
                        idx += 1
                        tag = (f"{ds_name}__{fm_type}__{az_type}_d{depth}__ga__w{wires}"
                               f"__pop={params['population_size']}_sel={params['selection_type']}"
                               f"_cro={params['crossover_type']}_mut={params['mutation_type']}"
                               f"_num={params['num_generations']}")
                        merged = {
                            "dataset": load_yaml(ds_yaml)["dataset"],
                            "feature_map": load_yaml(fm_yaml)["feature_map"],
                            "ansatz": load_yaml(az_yaml)["ansatz"],
                            "device": {"name": "default.qubit", "wires": wires, "shots": None},
                            "optimizer": {"type": "ga", "ga": params},
                            "k_folds": int(kfolds),
                            "objective_name": "accuracy",
                            "mid_checkpoint_epoch": 3 if args.smoke else 100,
                        }
                        cfg_path = cfgdir / f"{idx:06d}.yaml"
                        with open(cfg_path, "w", encoding="utf-8") as f:
                            yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)
                        cmd = f"qmlga-single --config {cfg_path} --resume --progress"
                        rl.write(cmd + "\n")
                        w.writerow([idx, tag, str(cfg_path)])
                        total += 1
                        if args.max_total and total >= args.max_total: print(total); return

    print(total)

if __name__ == "__main__":
    main()
