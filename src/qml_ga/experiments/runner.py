import os
from qml_ga.utils.io import load_yaml, merge_includes, make_run_dir, write_yaml, setup_cwd_to_repo_root, ensure_dirs
from qml_ga.experiments.kfold_train import run_kfold_experiment

class ExperimentRunner:
    def run_from_yaml(self, path: str):
        setup_cwd_to_repo_root()
        base_dir = os.path.dirname(path)
        root_cfg = load_yaml(path)
        cfg = merge_includes(root_cfg, base_dir)

        out_base = cfg.get("output", {}).get("base_dir", "runs")
        ensure_dirs(out_base)
        run_dir = make_run_dir(out_base)

        write_yaml(cfg, os.path.join(run_dir, "config_snapshot.yaml"))
        summary = run_kfold_experiment(cfg, run_dir)

        print("Execução concluída.")
        print("Run dir:", run_dir)
        print("Resumo:", summary)
