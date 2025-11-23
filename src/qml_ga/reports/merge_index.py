import os, glob, argparse, json, time
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

    # também mescla sweep_status.* em um JSON com snapshot por shard
    statuses = sorted(glob.glob(os.path.join(reports_dir, "sweep_status.*.json")))
    merged = []
    for p in statuses:
        try:
            with open(p, "r", encoding="utf-8") as f:
                merged.append(json.load(f))
        except Exception:
            pass
    with open(os.path.join(reports_dir, "sweep_status_merged.json"), "w", encoding="utf-8") as f:
        json.dump({"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "shards": merged}, f, ensure_ascii=False, indent=2)

    print(f"merged {len(parts)} parts -> {out} ({len(df)} rows)")
