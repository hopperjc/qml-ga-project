import os, glob, argparse
import pandas as pd

def main(reports_dir: str = "reports"):
    parts = sorted(glob.glob(os.path.join(reports_dir, "_index_parts", "index.*.csv")))
    if not parts:
        print("No parts found.")
        return
    dfs = [pd.read_csv(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["tag", "run_dir"], keep="last")
    out = os.path.join(reports_dir, "index.csv")
    df.to_csv(out, index=False)
    print(f"merged {len(parts)} parts -> {out} ({len(df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()
    main(**vars(args))
