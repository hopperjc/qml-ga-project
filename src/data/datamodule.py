import os
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class DataModule:
    def __init__(self, dataset_cfg: Dict):
        if "path" not in dataset_cfg:
            raise ValueError("dataset.path é obrigatório")
        self.path: str = dataset_cfg["path"]
        self.target: str = dataset_cfg.get("target", "target")

        fcfg: Dict = dataset_cfg.get("features", {}) or {}
        self.features_use: str = fcfg.get("use", "all")
        self.features_first_k: Optional[int] = fcfg.get("first_k", None)
        self.features_names: List[str] = list(fcfg.get("names", []) or [])

        kfold = dataset_cfg.get("kfold", {}) or {}
        self.n_splits: int = int(kfold.get("n_splits", 5))
        self.shuffle: bool = bool(kfold.get("shuffle", True))
        self.stratified: bool = bool(kfold.get("stratified", True))
        self.seed: int = int(kfold.get("seed", 42))

        self._df_cache: Optional[pd.DataFrame] = None
        self.selected_feature_names_: List[str] = []

    def _load_df(self) -> pd.DataFrame:
        if self._df_cache is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"CSV não encontrado em {self.path}")
            df = pd.read_csv(self.path)
            if self.target not in df.columns:
                raise ValueError(f"Coluna alvo '{self.target}' não encontrada")
            self._df_cache = df
        return self._df_cache

    @staticmethod
    def _feature_columns(df: pd.DataFrame, target: str) -> List[str]:
        return [c for c in df.columns if c != target]

    def _apply_feature_selection(self, df: pd.DataFrame) -> List[str]:
        cand = self._feature_columns(df, self.target)
        if self.features_use == "all":
            chosen = cand
        elif self.features_use == "first_k":
            if self.features_first_k is None or self.features_first_k < 1:
                raise ValueError("features.first_k inválido")
            if self.features_first_k > len(cand):
                raise ValueError("features.first_k maior que o número de colunas")
            chosen = cand[: self.features_first_k]
        elif self.features_use == "names":
            if not self.features_names:
                raise ValueError("features.names vazio")
            missing = [c for c in self.features_names if c not in cand]
            if missing:
                raise ValueError(f"Colunas não existem no CSV: {missing}")
            chosen = list(self.features_names)
        else:
            raise ValueError(f"features.use inválido: {self.features_use}")

        if self.target in chosen:
            raise ValueError("target não pode estar nas features")

        # remove duplicatas preservando ordem
        seen, unique = set(), []
        for c in chosen:
            if c not in seen:
                unique.append(c)
                seen.add(c)
        return unique

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = self._load_df()
        feats = self._apply_feature_selection(df)
        self.selected_feature_names_ = feats
        X = df[feats].to_numpy()
        y = df[self.target].to_numpy()
        return X, y, feats

    def folds(self, X: np.ndarray, y: np.ndarray):
        if self.stratified:
            skf = StratifiedKFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed
            )
            for tr, te in skf.split(X, y):
                yield tr, te
        else:
            n = len(y)
            idx = np.arange(n)
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                rng.shuffle(idx)
            parts = np.array_split(idx, self.n_splits)
            for i in range(self.n_splits):
                te = parts[i]
                tr = np.concatenate([parts[j] for j in range(self.n_splits) if j != i])
                yield tr, te

    @staticmethod
    def required_wires(feature_map_type: str, n_features: int) -> int:
        fm = feature_map_type.lower()
        if fm == "amplitude":
            # ceil(log2(n_features))
            k, wires = 1, 0
            while k < n_features:
                k <<= 1
            while k > 1:
                k >>= 1
                wires += 1
            return wires
        if fm == "zz":
            return int(n_features)
        raise ValueError(f"feature_map_type não suportado: {feature_map_type}")

    @staticmethod
    def ensure_wires_compatible(feature_map_type: str, n_features: int, device_wires: int) -> None:
        req = DataModule.required_wires(feature_map_type, n_features)
        if device_wires != req:
            raise ValueError(
                f"Incompatibilidade device.wires={device_wires} com Feature Map '{feature_map_type}'. "
                f"Para n_features={n_features}, wires correto = {req}."
            )
