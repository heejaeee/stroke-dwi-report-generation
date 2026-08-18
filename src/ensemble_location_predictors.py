import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def split_locs(x):
    if pd.isna(x): return set()
    return {s.strip() for s in str(x).split("|") if s.strip()}

def prob_cols(df):
    return [c for c in df.columns if c.startswith("prob_")]

def loc_metrics(y_sets, p_sets):
    f1s, exacts = [], []
    for y,p in zip(y_sets,p_sets):
        inter = len(y&p)
        prec = inter / len(p) if p else 0
        rec = inter / len(y) if y else 0
        f1s.append(2*prec*rec/(prec+rec) if prec+rec else 0)
        exacts.append(float(y == p))
    return float(np.mean(f1s)), float(np.mean(exacts))

def predict(probs, labels, threshold):
    out = []
    for row in probs:
        locs = {labels[i] for i,v in enumerate(row) if v >= threshold}
        if not locs:
            locs = {labels[int(np.argmax(row))]}
        out.append(locs)
    return out

def tune_threshold(val_df, labels, probs):
    y = [split_locs(x) for x in val_df["true_locations"]]
    best = (-1, 0.5, None)
    for t in np.arange(0.05, 0.96, 0.05):
        p = predict(probs, labels, t)
        f1, exact = loc_metrics(y, p)
        score = f1 + 0.25*exact
        if score > best[0]:
            best = (score, float(t), (f1, exact))
    return best

def build(split_paths, laterality_csv, out_csv, threshold=None):
    dfs = [pd.read_csv(p).fillna("") for p in split_paths]
    cols = prob_cols(dfs[0])
    labels = [c.replace("prob_", "") for c in cols]
    probs = np.mean([df[cols].to_numpy(float) for df in dfs], axis=0)

    base = dfs[0][["case_id", "true_locations"]].copy()
    pred = predict(probs, labels, threshold)
    base["pred_locations_ensemble_global"] = ["|".join(sorted(x)) for x in pred]
    base["pred_locations_ensemble_perclass"] = base["pred_locations_ensemble_global"]
    base["pred_location_top1"] = [labels[int(np.argmax(r))] for r in probs]

    lat = pd.read_csv(laterality_csv).fillna("")
    keep = [c for c in ["case_id", "true_laterality", "pred_laterality_ensemble", "pred_laterality"] if c in lat.columns]
    lat = lat[keep].drop_duplicates("case_id")
    base = base.merge(lat, on="case_id", how="left")
    if "pred_laterality_ensemble" not in base.columns and "pred_laterality" in base.columns:
        base["pred_laterality_ensemble"] = base["pred_laterality"]

    for j, lab in enumerate(labels):
        base[f"ens_prob_{lab}"] = probs[:, j]
        base[f"prob_{lab}"] = probs[:, j]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_csv, index=False)
    return base, labels, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csvs", nargs="+", required=True)
    ap.add_argument("--test_csvs", nargs="+", required=True)
    ap.add_argument("--laterality_val_csv", required=True)
    ap.add_argument("--laterality_test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    val0 = pd.read_csv(args.val_csvs[0]).fillna("")
    cols = prob_cols(val0)
    labels = [c.replace("prob_", "") for c in cols]
    val_probs = np.mean([pd.read_csv(p).fillna("")[cols].to_numpy(float) for p in args.val_csvs], axis=0)
    score, threshold, val_m = tune_threshold(val0, labels, val_probs)
    print("[BEST]", {"threshold": threshold, "val_f1": val_m[0], "val_exact": val_m[1], "score": score})

    build(args.val_csvs, args.laterality_val_csv, out/"val_predictions_location_ensemble.csv", threshold)
    test_df, _, _ = build(args.test_csvs, args.laterality_test_csv, out/"test_predictions_location_ensemble.csv", threshold)

    y = [split_locs(x) for x in test_df["true_locations"]]
    p = [split_locs(x) for x in test_df["pred_locations_ensemble_global"]]
    f1, exact = loc_metrics(y, p)
    print("[TEST]", {"location_f1": f1, "location_exact": exact})
    print("[SAVED]", out)

if __name__ == "__main__":
    main()
