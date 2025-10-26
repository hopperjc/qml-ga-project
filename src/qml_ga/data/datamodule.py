# src/qml_ga/data/datamodule.py
from typing import Dict, Tuple, List
import os
import pandas as pd
import numpy as np
from math import ceil, log2

class DataModule:
    def __init__(self, dataset_cfg: Dict):
        self.cfg = dataset_cfg

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        path = self.cfg["path"]
        target = self.cfg["target"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV não encontrado: {path}")
        df = pd.read_csv(path)
        if target not in df.columns:
            raise ValueError(f"Coluna target '{target}' não encontrada em {path}")
        y = df[target].values
        X = df.drop(columns=[target]).values

        # mapear rótulos para {-1,+1} se não numéricos
        if not np.issubdtype(y.dtype, np.number):
            classes = sorted(pd.unique(y))
            if len(classes) != 2:
                raise ValueError(f"Target não-binário: {classes}")
            mapping = {classes[0]: -1.0, classes[1]: +1.0}
            y = np.array([mapping[v] for v in y], dtype=float)
        else:
            # normaliza para {-1,+1} se vier {0,1}
            uy = np.unique(y)
            if set(uy) == {0, 1}:
                y = np.where(y > 0, 1.0, -1.0).astype(float)
            else:
                y = y.astype(float)

        feats = list(df.drop(columns=[target]).columns)
        return X.astype(float), y, feats

    @staticmethod
    def required_wires(fm_type: str, n_features: int) -> int:
        fm = (fm_type or "").lower()
        if fm == "amplitude":
            return int(ceil(log2(max(1, n_features))))
        elif fm == "zz":
            return int(n_features)
        else:
            raise ValueError(f"Feature map desconhecido: {fm_type}")

    @staticmethod
    def ensure_wires_compatible(fm_type: str, n_features: int, wires: int):
        req = DataModule.required_wires(fm_type, n_features)
        if wires != req:
            raise ValueError(f"Incompatibilidade de wires: requerido={req}, recebido={wires}")
