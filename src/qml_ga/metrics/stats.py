import os
import glob
import json
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats


# -----------------------------------------------------------------------------
# 1) Coleta e organização dos resultados
# -----------------------------------------------------------------------------

def _read_metrics_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_config_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def collect_runs(runs_dir: str, metric: str = "accuracy") -> pd.DataFrame:
    """
    Vasculha runs/*_run/ e coleta:
      - metrics.json  (lista 'folds' com métricas por fold)
      - config_snapshot.yaml (metadados: dataset/optimizer/feature_map/ansatz)

    Retorna DataFrame "long" com uma linha por fold:
      ['run_dir','dataset','optimizer','feature_map','ansatz','depth','fold',
       'accuracy','f1','precision','recall','objective_name','train_objective']
    """
    rows = []
    run_dirs = sorted(glob.glob(os.path.join(runs_dir, "*_run")))
    for rd in run_dirs:
        mpath = os.path.join(rd, "metrics.json")
        cpath = os.path.join(rd, "config_snapshot.yaml")
        if not (os.path.exists(mpath) and os.path.exists(cpath)):
            continue

        mj = _read_metrics_json(mpath)
        cfg = _read_config_yaml(cpath)

        dataset = cfg.get("dataset", {}).get("name", "unknown")
        optimizer = cfg.get("optimizer", {}).get("type", "unknown")
        feature_map = cfg.get("feature_map", {}).get("type", "unknown")
        ansatz = cfg.get("ansatz", {}).get("type", "unknown")
        depth = cfg.get("ansatz", {}).get("params", {}).get("depth", None)

        folds = mj.get("folds", [])
        for fd in folds:
            rows.append({
                "run_dir": rd,
                "dataset": dataset,
                "optimizer": optimizer,
                "feature_map": feature_map,
                "ansatz": ansatz,
                "depth": depth,
                "fold": int(fd.get("fold", -1)),
                "accuracy": float(fd.get("accuracy", np.nan)),
                "f1": float(fd.get("f1", np.nan)),
                "precision": float(fd.get("precision", np.nan)),
                "recall": float(fd.get("recall", np.nan)),
                "objective_name": fd.get("objective_name", mj.get("objective_name", "")),
                "train_objective": float(fd.get("train_objective", np.nan)),
            })

    if not rows:
        raise FileNotFoundError(
            f"Nenhum run encontrado em {runs_dir}. "
            f"Certifique-se de ter executado os experimentos e que os artefatos foram salvos."
        )

    df = pd.DataFrame(rows)

    # sanity: remover entradas sem a métrica escolhida
    if metric not in df.columns:
        raise ValueError(f"Métrica '{metric}' não existe nas colunas: {list(df.columns)}")

    df = df.dropna(subset=[metric]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# 2) Preparação para Friedman (matriz D×K)
# -----------------------------------------------------------------------------

def _make_algo_id(df: pd.DataFrame, algo_cols: Iterable[str]) -> pd.Series:
    cols = list(algo_cols)
    return df[cols].astype(str).agg(" | ".join, axis=1)

def pivot_for_friedman(
    df: pd.DataFrame,
    metric: str = "accuracy",
    algo_cols: Tuple[str, ...] = ("optimizer", "feature_map", "ansatz"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói a matriz D×K exigida pelo Friedman.
    - Agrega por média do 'metric' sobre os folds de cada (dataset, algoritmo).
    - 'algoritmo' = concatenação das colunas em 'algo_cols'.
    - Remove colunas (algoritmos) que não tenham valores em TODOS os datasets.

    Retorna:
      - df_avg: DataFrame long com média por (dataset, algoritmo)
      - mat:    DataFrame pivotado (index=dataset, columns=algoritmo), shape (D,K)
    """
    algo_id = _make_algo_id(df, algo_cols)
    tmp = df.assign(algorithm=algo_id)
    df_avg = (
        tmp.groupby(["dataset", "algorithm"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: f"{metric}_mean"})
    )

    # pivot
    mat = df_avg.pivot(index="dataset", columns="algorithm", values=f"{metric}_mean")

    # mantém apenas algoritmos completos (sem NaN em nenhuma linha)
    complete_cols = mat.columns[mat.notna().all(axis=0)]
    mat = mat[complete_cols].dropna(axis=0, how="any")

    if mat.shape[0] < 2 or mat.shape[1] < 2:
        raise ValueError(
            "Matriz para Friedman precisa de pelo menos 2 datasets e 2 algoritmos completos."
        )

    return df_avg, mat


# -----------------------------------------------------------------------------
# 3) Friedman + Nemenyi (pós-hoc)
# -----------------------------------------------------------------------------

def friedman_test_from_df(
    df: pd.DataFrame,
    metric: str = "accuracy",
    algo_cols: Tuple[str, ...] = ("optimizer", "feature_map", "ansatz"),
    alpha: float = 0.05,
) -> Dict:
    """
    Executa o teste de Friedman na matriz D×K (médias por dataset/algoritmo) e
    aplica Nemenyi (CD) para pós-hoc.

    Retorno:
      {
        "statistic": ...,
        "pvalue": ...,
        "avg_ranks": {algorithm: rank_medio},
        "CD": ...,
        "significant_pairs": [(alg_i, alg_j), ...],
        "matrix": DataFrame D×K usado no teste
      }
    """
    df_avg, mat = pivot_for_friedman(df, metric=metric, algo_cols=algo_cols)
    data = mat.to_numpy()  # shape (D, K)
    D, K = data.shape

    # Friedman (SciPy espera K arrays, cada um com D observações)
    stat, p = stats.friedmanchisquare(*[data[:, j] for j in range(K)])

    # ranks por dataset (1 = melhor ⇒ ordenar por maior métrica)
    ranks = np.vstack([stats.rankdata(-row, method="average") for row in data])  # (D,K)
    avg_ranks = ranks.mean(axis=0)  # (K,)

    # Nemenyi (CD)
    try:
        q_alpha = stats.studentized_range.ppf(1 - alpha, K, np.inf)
    except Exception:
        # caso SciPy muito antigo (improvável); fallback aproximado
        # nota: valores de q_alpha padrão (K=2..10, alpha=0.05) poderiam ser tabulados aqui
        raise RuntimeError(
            "Não foi possível obter q_alpha da distribuição 'studentized_range'. "
            "Atualize o SciPy para usar o pós-hoc de Nemenyi."
        )
    CD = float(q_alpha * np.sqrt(K * (K + 1) / (6.0 * D)))

    # pares significativos: diferença de ranks médios maior que CD
    algs = list(mat.columns)
    sig_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            if abs(avg_ranks[i] - avg_ranks[j]) > CD:
                sig_pairs.append((algs[i], algs[j]))

    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "avg_ranks": dict(zip(algs, avg_ranks)),
        "CD": CD,
        "significant_pairs": sig_pairs,
        "matrix": mat,
    }

def nemenyi_from_ranks(
    avg_ranks: Dict[str, float],
    D: int,
    alpha: float = 0.05,
) -> Dict:
    """
    Calcula CD e pares significativos a partir de ranks médios já obtidos.
    Útil para reaproveitar resultados.
    """
    algs = list(avg_ranks.keys())
    ranks = np.array([avg_ranks[a] for a in algs])
    K = len(algs)
    q_alpha = stats.studentized_range.ppf(1 - alpha, K, np.inf)
    CD = float(q_alpha * np.sqrt(K * (K + 1) / (6.0 * D)))
    sig_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            if abs(ranks[i] - ranks[j]) > CD:
                sig_pairs.append((algs[i], algs[j]))
    return {"CD": CD, "significant_pairs": sig_pairs}


# -----------------------------------------------------------------------------
# 4) Normalidade por grupo (Shapiro–Wilk)
# -----------------------------------------------------------------------------

def shapiro_by(
    df: pd.DataFrame,
    metric: str = "accuracy",
    by: Tuple[str, ...] = ("dataset", "optimizer"),
) -> pd.DataFrame:
    """
    Roda Shapiro–Wilk por grupo (ex.: por dataset e optimizer) usando os valores por fold.
    Retorna DataFrame com colunas: list(by) + ['n','W','pvalue'].
    """
    groups = df.groupby(list(by))
    rows = []
    for key, g in groups:
        x = g[metric].to_numpy()
        if len(x) < 3:
            # Shapiro exige n>=3
            rows.append((key if isinstance(key, tuple) else (key,), len(x), np.nan, np.nan))
            continue
        W, p = stats.shapiro(x)
        rows.append((key if isinstance(key, tuple) else (key,), len(x), float(W), float(p)))
    cols = list(by) + ["n", "W", "pvalue"]
    return pd.DataFrame(rows, columns=cols).sort_values(by=list(by)).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 5) ANOVA de dois fatores (dataset e optimizer)
# -----------------------------------------------------------------------------

def anova_two_way(
    df: pd.DataFrame,
    metric: str = "accuracy",
) -> Optional[pd.DataFrame]:
    """
    ANOVA de dois fatores com interação:
      metric ~ C(dataset) + C(optimizer) + C(dataset):C(optimizer)

    Requer 'statsmodels'. Retorna a tabela ANOVA (Type II). Se o pacote não estiver
    disponível, retorna None e imprime instruções de instalação.
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
    except Exception:
        print(
            "[ANOVA] 'statsmodels' não está instalado. "
            "Instale com: 'poetry add statsmodels' ou 'pip install statsmodels'."
        )
        return None

    # modelagem: usamos os valores por fold (mais amostras)
    df2 = df.copy()
    df2["dataset"] = df2["dataset"].astype("category")
    df2["optimizer"] = df2["optimizer"].astype("category")

    formula = f"{metric} ~ C(dataset) + C(optimizer) + C(dataset):C(optimizer)"
    model = smf.ols(formula, data=df2).fit()
    table = anova_lm(model, typ=2)
    return table


# -----------------------------------------------------------------------------
# 6) Relatório rápido (pipeline)
# -----------------------------------------------------------------------------

def quick_report(
    runs_dir: str,
    metric: str = "accuracy",
    algo_cols: Tuple[str, ...] = ("optimizer", "feature_map", "ansatz"),
    alpha: float = 0.05,
) -> Dict:
    """
    Executa coleta, Friedman+Nemenyi, Shapiro e ANOVA-2F e imprime um resumo.
    Retorna um dicionário com todos os resultados intermediários.
    """
    df = collect_runs(runs_dir, metric=metric)

    # Friedman + Nemenyi (por algoritmo = combinação de cols)
    fr = friedman_test_from_df(df, metric=metric, algo_cols=algo_cols, alpha=alpha)

    print("\n=== Resumo dos Dados ===")
    print(df.groupby(["dataset", "optimizer"]).size().rename("n_folds").reset_index())
    print(f"\nMatriz D×K usada no Friedman ({fr['matrix'].shape[0]} datasets × {fr['matrix'].shape[1]} algoritmos):")
    print(fr["matrix"])

    print("\n=== Friedman ===")
    print(f"chi2 = {fr['statistic']:.4f}, p = {fr['pvalue']:.6f}")
    print("Ranks médios (1 = melhor):")
    for alg, r in sorted(fr["avg_ranks"].items(), key=lambda kv: kv[1]):
        print(f"  {alg}: {r:.3f}")

    print("\n=== Nemenyi (pós-hoc) ===")
    print(f"CD (@ α={alpha}) = {fr['CD']:.4f}")
    if fr["significant_pairs"]:
        print("Pares com diferença significativa de ranks:")
        for a, b in fr["significant_pairs"]:
            print(f"  {a} vs {b}")
    else:
        print("Nenhuma diferença significativa detectada.")

    # Shapiro por (dataset, optimizer)
    sh = shapiro_by(df, metric=metric, by=("dataset", "optimizer"))
    print("\n=== Shapiro–Wilk (por dataset, optimizer) ===")
    print(sh)

    # ANOVA 2 fatores
    anova_tbl = anova_two_way(df, metric=metric)
    if anova_tbl is not None:
        print("\n=== ANOVA (dois fatores com interação) ===")
        print(anova_tbl)

    return {
        "df": df,
        "friedman": fr,
        "shapiro": sh,
        "anova2": anova_tbl,
    }
