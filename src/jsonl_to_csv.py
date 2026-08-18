import argparse
import json
from pathlib import Path

import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []

    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            rows.append({
                "id": r.get("id", r.get("patient_id", "")),
                "patient_id": r.get("patient_id", r.get("id", "")),
                "split": r.get("split", ""),
                "source": r.get("source", ""),
                "dataset_source": r.get("dataset_source", ""),
                "original_split": r.get("original_split", ""),
                "original_id": r.get("original_id", ""),
                "target": r.get("target", ""),
                "prediction": r.get("prediction", r.get("pred", r.get("output", ""))),
            })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print("[SAVED]", out)
    print("rows:", len(rows))

if __name__ == "__main__":
    main()
