import os, json, argparse, pandas as pd, numpy as np
from qml_ga.metrics.stats import friedman_test_from_df, shapiro_by, anova_two_way

def load_index(reports_dir: str) -> pd.DataFrame:
    idx = os.path.join(reports_dir, "index.csv")
    if not os.path.exists(idx):
        raise FileNotFoundError(f"Não encontrei {idx}. Rode o sweep primeiro.")
    df = pd.read_csv(idx)
    # garantir colunas
    exp_cols = ["tag","dataset","feature_map","ansatz","depth","optimizer","device_wires","mean_accuracy","std_accuracy","run_dir"]
    for c in exp_cols:
        if c not in df.columns:
            raise ValueError(f"index.csv não possui a coluna obrigatória: {c}")
    return df

def arch_key(df):
    return (df["dataset"].astype(str) + "|" + df["feature_map"].astype(str) +
            "|" + df["ansatz"].astype(str) + "|d" + df["depth"].astype(str) +
            "|w" + df["device_wires"].astype(str))

def pick_winners(df: pd.DataFrame, metric="mean_accuracy"):
    df = df.copy()
    df["arch"] = arch_key(df)

    # separa GA vs Gradiente
    is_ga = df["optimizer"].str.lower().eq("ga")
    ga = df[is_ga]
    grad = df[~is_ga]   # adam/nesterov

    # Melhor GA por arquitetura
    ga_best = ga.sort_values([ "arch", metric ], ascending=[True, False]).groupby("arch", as_index=False).first()

    # Melhor Gradiente por arquitetura (pega o melhor entre Adam/Nesterov)
    grad_best = grad.sort_values([ "arch", metric ], ascending=[True, False]).groupby("arch", as_index=False).first()

    # Junta apenas onde temos os dois (comparação justa)
    merged = pd.merge(ga_best, grad_best, on="arch", suffixes=("_ga","_grad"))
    return merged

def build_fold_df_for_stats(winners: pd.DataFrame) -> pd.DataFrame:
    """
    Monta um DF 'long' por fold com colunas:
    ['arch','dataset','optimizer','accuracy', ...] apenas com os VENCEDORES.
    """
    rows = []
    for _, r in winners.iterrows():
        # GA winner
        for tag, opt in [(r["tag_ga"], "ga"), (r["tag_grad"], "grad")]:
            rep_dir = os.path.join("reports", tag)
            folds_csv = os.path.join(rep_dir, "folds.csv")
            if not os.path.exists(folds_csv):
                continue
            df_f = pd.read_csv(folds_csv)
            for _, ff in df_f.iterrows():
                rows.append({
                    "arch": r["arch"],
                    "dataset": r["dataset_ga"],  # = dataset_grad
                    "optimizer": opt,
                    "accuracy": ff.get("accuracy", np.nan),
                    "f1": ff.get("f1", np.nan),
                    "precision": ff.get("precision", np.nan),
                    "recall": ff.get("recall", np.nan),
                })
    return pd.DataFrame(rows)

def wilcoxon_signed(df_arch_mean: pd.DataFrame, metric="accuracy"):
    """Wilcoxon entre GA e Gradiente nas médias por arquitetura (pareado)."""
    from scipy.stats import wilcoxon
    pivot = df_arch_mean.pivot(index="arch", columns="optimizer", values=metric).dropna()
    if set(pivot.columns) >= {"ga", "grad"} and len(pivot) >= 1:
        stat, p = wilcoxon(pivot["ga"], pivot["grad"], zero_method="wilcox", alternative="greater")
        return {"statistic": float(stat), "pvalue": float(p), "n": int(len(pivot))}
    return {"statistic": None, "pvalue": None, "n": 0}

def main(reports_dir="reports", metric="accuracy", alpha=0.05):
    df = load_index(reports_dir)
    winners = pick_winners(df, metric="mean_accuracy")
    os.makedirs(os.path.join(reports_dir, "best"), exist_ok=True)
    winners.to_csv(os.path.join(reports_dir, "best", "winners_by_architecture.csv"), index=False)

    # Dados por fold apenas dos vencedores
    folds = build_fold_df_for_stats(winners)
    folds.to_csv(os.path.join(reports_dir, "best", "winners_folds_long.csv"), index=False)

    # Estatística 1: Wilcoxon (GA > Grad) em médias por arquitetura (robusto para n pares)
    arch_mean = folds.groupby(["arch","optimizer"], as_index=False)[metric].mean()
    w = wilcoxon_signed(arch_mean, metric=metric)

    # Estatística 2: Friedman+Nemenyi entre {GA, Grad} sobre as arquiteturas (K=2) — menos informativo; manter por completude
    fr = None
    try:
        fr = friedman_test_from_df(
            arch_mean.rename(columns={metric: f"{metric}"}), metric=metric, algo_cols=("optimizer",), alpha=alpha
        )
    except Exception:
        pass

    # Estatística 3: Shapiro e ANOVA (opcional; requer normalidade)
    sh = shapiro_by(folds, metric=metric, by=("arch","optimizer"))
    anova = anova_two_way(folds.rename(columns={"optimizer":"group"}), metric=metric)  # dataset vs group

    # Relatório
    summary = {
        "wilcoxon_signed": w,
        "friedman": None if fr is None else {"chi2": fr["statistic"], "pvalue": fr["pvalue"], "CD": fr["CD"]},
        "n_architectures": int(arch_mean["arch"].nunique()),
    }
    with open(os.path.join(reports_dir, "best", "stats_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    sh.to_csv(os.path.join(reports_dir, "best", "shapiro_by_arch_optimizer.csv"), index=False)
    if anova is not None:
        anova.to_csv(os.path.join(reports_dir, "best", "anova_two_way.csv"))

    print("\n=== RESULTADOS (configs vencedoras) ===")
    print(f"Arquiteturas comparadas: {summary['n_architectures']}")
    print(f"Wilcoxon (GA > Grad): stat={w['statistic']} p={w['pvalue']} n={w['n']}")
    if fr is not None:
        print(f"Friedman (K=2): chi2={summary['friedman']['chi2']:.4f} p={summary['friedman']['pvalue']:.6f}  CD={summary['friedman']['CD']:.4f}")
    print(f"[OK] Artefatos em {os.path.join(reports_dir, 'best')}")
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    main(**vars(args))
