import os, time, yaml

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(obj, path: str):
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def make_run_dir(base="runs"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(base, f"{ts}_run")
    ensure_dirs(d)
    return d

def setup_cwd_to_repo_root():
    here = os.getcwd()
    probe = here
    while True:
        if os.path.exists(os.path.join(probe, "pyproject.toml")):
            os.chdir(probe)
            return
        parent = os.path.dirname(probe)
        if parent == probe:
            os.chdir(here)
            return
        probe = parent
