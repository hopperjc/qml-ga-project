# src/qml_ga/utils/logger.py
import os
import time
import pandas as pd
from qml_ga.utils.io import ensure_dirs

def make_text_logger(path: str, prefix: str = ""):
    """Cria um logger que imprime no console (flush) e grava no arquivo."""
    ensure_dirs(os.path.dirname(path))
    def _log(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {prefix}{msg}"
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return _log

def append_csv_row(path: str, row: dict):
    """Acrescenta uma linha em CSV, criando cabeçalho se necessário."""
    ensure_dirs(os.path.dirname(path))
    exists = os.path.exists(path)
    pd.DataFrame([row]).to_csv(
        path,
        mode=("a" if exists else "w"),
        header=(not exists),
        index=False,
    )
