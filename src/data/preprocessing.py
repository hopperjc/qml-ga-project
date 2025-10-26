import pandas as pd

def load_csv(path: str, target: str = "target"):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Coluna alvo {target} não encontrada em {path}")
    X = df.drop(columns=[target]).values
    y = df[target].values
    return X, y, df.columns.tolist()
