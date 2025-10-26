import os
import json
import time
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def merge_includes(root_cfg: dict, base_dir: str) -> dict:
    merged = {}
    includes = root_cfg.get("includes", [])
    for inc in includes:
        inc_path = os.path.join(base_dir, inc)
        merged.update(load_yaml(inc_path))
    merged.update({k: v for k, v in root_cfg.items() if k != "includes"})
    return merged

def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def make_run_dir(base: str) -> str:
    os.makedirs(base, exist_ok=True)
    name = f"{timestamp()}_run"
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path

def write_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_yaml(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def setup_cwd_to_repo_root():
    cur = os.path.abspath(os.getcwd())
    markers = {"pyproject.toml", ".git", "configs"}
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in markers):
            os.chdir(cur)
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.getcwd()

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)
