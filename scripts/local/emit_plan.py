#!/usr/bin/env python3
import os, sys, glob, argparse, math, csv, shlex, yaml
from pathlib import Path
import pandas as pd
from itertools import product

def q(p: Path) -> str:
    return shlex.quote(str(p))

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_yaml(dirpath):
    return sorted(glob.glob(os.path.join(dirpath, "*.yaml")))

def count_features_from_dataset_yaml(ds_yaml):
    ds = load_yaml(ds_yaml)
    csv_path = ds["dataset"]["path"]
    target = ds["dataset"]["target"]
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="Raiz do repositório")
    ap.add_argument("--outdir", default="local/plan_full_all", help="Saída do plano")
    ap.add_argument("--include_feature_maps", nargs="*", default=[], help="Vazio = todos")
    ap.add_argument("--max_wires", type=int, default=64, help="Descarta combinações com wires maiores")
    ap.add_argument("--max_total", type=int, default=0, help="Limite total (0=sem limite)")
    ap.add_argument("--smoke", action="store_true", help="Clássicos: epochs=5; GA: generations=5")
    ap.add_argument("--val_size", type=float, default=0.2, help="Proporção de validação para qmlga-single")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    ds_dir = repo / "configs" / "datasets"
    fm_dir = repo / "configs" / "feature_maps"
    az_dir = repo / "configs" / "ansatz"
    hg_dir = repo / "configs" / "hypergrids"

    ds_list = [Path(p).resolve() for p in list_yaml(str(ds_dir))]
    fm_list = [Path(p).resolve() for p in list_yaml(str(fm_dir))]
    az_list = [Path(p).resolve() for p in list_yaml(str(az_dir))]

    if args.include_feature_maps:
        fm_list = [p for p in fm_list if p.stem in args.include_feature_maps]

    hg_class = load_yaml(hg_dir / "classical.yaml")["classical"]
    hg_ga = load_yaml(hg_dir / "ga.yaml")["ga"]

    outdir = Path(args.outdir).resolve()
    cfgdir = outdir / "configs"
    cfgdir.mkdir(parents=True, exist_ok=True)  # apenas para manter compatibilidade visual
    runlist_path = outdir / "runlist.txt"
    plan_csv = outdir / "plan.csv"
    outdir.mkdir(parents=True, exist_ok=True)

    total = 0
    idx = 0

    with open(runlist_path, "w", encoding="utf-8") as rl, open(plan_csv, "w", newline="", encoding="utf-8") as pcsv:
        w = csv.writer(pcsv)
        w.writerow(["idx", "tag", "dataset_yaml", "feature_map_yaml", "ansatz_yaml", "optimizer", "extra"])

        for ds_yaml in ds_list:
            n_feats = count_features_from_dataset_yaml(ds_yaml)
            ds_name = ds_yaml.stem

            for fm_yaml in fm_list:
                fm = load_yaml(fm_yaml)["feature_map"]
                fm_type = fm["type"]
                wires = infer_wires(fm_type, n_feats)
                if wires > args.max_wires:
                    continue

                for az_yaml in az_list:
                    az_type, depth = ansatz_meta_from_yaml(az_yaml)

                    # -------------------------
                    # Otimizadores clássicos
                    # -------------------------
                    for opt_name, opt_grid in hg_class.items():
                        lrs = opt_grid.get("lr", [0.05])
                        eps = opt_grid.get("epochs", [200])
                        bss = opt_grid.get("batch_size", [16])

                        if args.smoke:
                            eps = [5]  # smoke

                        for lr, epochs, bs in product(lrs, eps, bss):
                            idx += 1
                            tag = f"{ds_name}__{fm_type}__{az_type}_d{depth}__{opt_name}__w{wires}__ep={epochs}_lr={lr}"
                            cmd = (
                                f"qmlga-single "
                                f"--dataset_yaml {q(ds_yaml)} "
                                f"--feature_map_yaml {q(fm_yaml)} "
                                f"--ansatz_yaml {q(az_yaml)} "
                                f"--optimizer {opt_name} "
                                f"--device default.qubit --wires {wires} --shots 0 "
                                f"--val_size {args.val_size} "
                                f"--epochs {int(epochs)} --lr {float(lr)} --batch_size {int(bs)} "
                                f"--reports_dir reports --runs_dir runs "
                                f"--log_every 1 --progress --resume"
                            )
                            rl.write(cmd + "\n")
                            w.writerow([idx, tag, str(ds_yaml), str(fm_yaml), str(az_yaml), opt_name, f"epochs={epochs},lr={lr},bs={bs}"])
                            total += 1
                            if args.max_total and total >= args.max_total:
                                print(total)
                                return

                    # -------------------------
                    # Algoritmo Genético (GA)
                    # -------------------------
                    grid = {
                        "population_size": hg_ga.get("population_size", [50, 100]),
                        "num_generations": hg_ga.get("num_generations", [100]),
                        "selection_type": hg_ga.get("selection_type", ["tournament", "sss"]),
                        "crossover_type": hg_ga.get("crossover_type", ["single_point", "two_points"]),
                        "mutation_type": hg_ga.get("mutation_type", ["random", "adaptative"]),
                        "mutation_percent_genes": hg_ga.get("mutation_percent_genes", [10]),
                        "elitism": hg_ga.get("elitism", [2]),
                        "init_range_low": hg_ga.get("init_range_low", [-3.14]),
                        "init_range_high": hg_ga.get("init_range_high", [3.14]),
                    }
                    if args.smoke:
                        grid["num_generations"] = [5]  # smoke

                    keys = list(grid.keys())
                    for values in product(*[grid[k] for k in keys]):
                        params = dict(zip(keys, values))
                        idx += 1
                        tag = (
                            f"{ds_name}__{fm_type}__{az_type}_d{depth}__ga__w{wires}"
                            f"__pop={params['population_size']}_sel={params['selection_type']}"
                            f"_cro={params['crossover_type']}_mut={params['mutation_type']}"
                            f"_num={params['num_generations']}"
                        )
                        cmd = (
                            f"qmlga-single "
                            f"--dataset_yaml {q(ds_yaml)} "
                            f"--feature_map_yaml {q(fm_yaml)} "
                            f"--ansatz_yaml {q(az_yaml)} "
                            f"--optimizer ga "
                            f"--device default.qubit --wires {wires} --shots 0 "
                            f"--val_size {args.val_size} "
                            f"--population_size {int(params['population_size'])} "
                            f"--num_generations {int(params['num_generations'])} "
                            f"--selection_type {params['selection_type']} "
                            f"--crossover_type {params['crossover_type']} "
                            f"--mutation_type {params['mutation_type']} "
                            f"--mutation_percent_genes {int(params['mutation_percent_genes'])} "
                            f"--elitism {int(params['elitism'])} "
                            f"--init_range_low {float(params['init_range_low'])} "
                            f"--init_range_high {float(params['init_range_high'])} "
                            f"--reports_dir reports --runs_dir runs "
                            f"--log_every 1 --progress --resume"
                        )
                        rl.write(cmd + "\n")
                        w.writerow([idx, tag, str(ds_yaml), str(fm_yaml), str(az_yaml), "ga", str(params)])
                        total += 1
                        if args.max_total and total >= args.max_total:
                            print(total)
                            return

    print(total)

if __name__ == "__main__":
    main()
