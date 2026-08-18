import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_mod(path):
    spec = importlib.util.spec_from_file_location("rerank94", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidate_features(c, attr, loc_labels, source_labels):
    locs = c["locs"]
    lat = c["lat"]
    attr_locs = attr.get("locs", set())
    attr_lat = attr.get("lat", "unknown")

    overlap = len(locs & attr_locs)
    extra = len(locs - attr_locs)
    missing = len(attr_locs - locs)

    f = {
        "n_locs": len(locs),
        "pred_len": c["pred_len"],
        "attr_n_locs": len(attr_locs),
        "loc_overlap": overlap,
        "loc_extra": extra,
        "loc_missing": missing,
        "loc_jaccard": overlap / max(1, len(locs | attr_locs)),
        "lat_match": int(lat == attr_lat and lat != "unknown"),
        "lat_unknown": int(lat == "unknown"),
        "lat_left": int(lat == "left"),
        "lat_right": int(lat == "right"),
        "lat_bilateral": int(lat == "bilateral"),
        "attr_lat_left": int(attr_lat == "left"),
        "attr_lat_right": int(attr_lat == "right"),
        "attr_lat_bilateral": int(attr_lat == "bilateral"),
    }

    for s in source_labels:
        f[f"source__{s}"] = int(c["source"] == s)
    for loc in loc_labels:
        f[f"cand_loc__{loc}"] = int(loc in locs)
        f[f"attr_loc__{loc}"] = int(loc in attr_locs)

    for mode in [
        "balanced",
        "precision",
        "recall",
        "hybrid_preferred",
        "calibrated_v1",
        "calibrated_v2",
        "calibrated_v3",
    ]:
        try:
            f[f"score__{mode}"] = float(MOD.score_candidate(c, attr, mode))
        except Exception:
            f[f"score__{mode}"] = 0.0
    return f


def collect_labels(*by_id_maps):
    locs = set()
    sources = set()
    for by_id in by_id_maps:
        for cands in by_id.values():
            for c in cands:
                locs |= set(c["locs"])
                sources.add(c["source"])
    return sorted(locs), sorted(sources)


def make_table(by_id, targets, attr_map, loc_labels, source_labels, include_oracle):
    rows = []
    for pid, cands in by_id.items():
        target = targets.get(pid, cands[0].get("target", ""))
        oracle = MOD.choose_oracle(cands, target) if include_oracle else None
        for i, c in enumerate(cands):
            attr = attr_map.get(pid, attr_map.get(c.get("patient_id", pid), {"lat": "unknown", "locs": set()}))
            f = candidate_features(c, attr, loc_labels, source_labels)
            rows.append({
                "id": pid,
                "patient_id": c.get("patient_id", pid),
                "cand_idx": i,
                "target": target,
                "prediction": c["prediction"],
                "source": c["source"],
                "is_oracle": int(c is oracle) if include_oracle else np.nan,
                **f,
            })
    return pd.DataFrame(rows)


def select_by_scores(df, score_col):
    rows = []
    for pid, g in df.groupby("id", sort=False):
        r = g.sort_values(score_col, ascending=False).iloc[0]
        rows.append({
            "id": pid,
            "patient_id": r["patient_id"],
            "target": r["target"],
            "prediction": r["prediction"],
            "selected_source": r["source"],
            "rerank_score": r[score_col],
        })
    return pd.DataFrame(rows)


def feature_columns(df):
    drop = {"id", "patient_id", "cand_idx", "target", "prediction", "source", "is_oracle"}
    return [c for c in df.columns if c not in drop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--candidate_scoring_script",
        default=str(Path(__file__).with_name("candidate_scoring.py")),
    )
    ap.add_argument("--val_attr_csv", required=True)
    ap.add_argument("--test_attr_csv", required=True)
    ap.add_argument("--val_candidates", nargs="+", required=True)
    ap.add_argument("--test_candidates", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    global MOD
    MOD = load_mod(args.candidate_scoring_script)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_by_id, val_targets = MOD.load_candidates(args.val_candidates)
    test_by_id, test_targets = MOD.load_candidates(args.test_candidates)
    val_attr = MOD.load_attr(args.val_attr_csv)
    test_attr = MOD.load_attr(args.test_attr_csv)

    loc_labels, source_labels = collect_labels(val_by_id, test_by_id)
    val_df = make_table(val_by_id, val_targets, val_attr, loc_labels, source_labels, include_oracle=True)
    test_df = make_table(test_by_id, test_targets, test_attr, loc_labels, source_labels, include_oracle=False)

    val_df.to_csv(out_dir / "val_candidate_feature_table.csv", index=False)
    test_df.to_csv(out_dir / "test_candidate_feature_table.csv", index=False)

    print("[INFO] val cases:", val_df["id"].nunique(), "candidates:", len(val_df))
    print("[INFO] test cases:", test_df["id"].nunique(), "candidates:", len(test_df))
    print("[INFO] val oracle source:")
    print(val_df[val_df["is_oracle"] == 1]["source"].value_counts().to_string())

    cols = feature_columns(val_df)
    X_val = val_df[cols].fillna(0).to_numpy(float)
    y_val = val_df["is_oracle"].to_numpy(int)
    X_test = test_df[cols].fillna(0).to_numpy(float)

    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gb": GradientBoostingClassifier(random_state=42),
    }

    for name, model in models.items():
        groups = val_df["id"].to_numpy()
        n_splits = min(5, len(pd.unique(groups)))
        val_oof_prob = np.zeros(len(val_df), dtype=float)

        if n_splits >= 2:
            gkf = GroupKFold(n_splits=n_splits)
            for tr, va in gkf.split(X_val, y_val, groups):
                fold_model = clone(model)
                fold_model.fit(X_val[tr], y_val[tr])
                if hasattr(fold_model, "predict_proba"):
                    val_oof_prob[va] = fold_model.predict_proba(X_val[va])[:, 1]
                else:
                    val_oof_prob[va] = fold_model.decision_function(X_val[va])
        else:
            val_oof_prob[:] = 0.0

        model.fit(X_val, y_val)
        if hasattr(model, "predict_proba"):
            val_prob = model.predict_proba(X_val)[:, 1]
            test_prob = model.predict_proba(X_test)[:, 1]
        else:
            val_prob = model.decision_function(X_val)
            test_prob = model.decision_function(X_test)

        val_oof_tmp = val_df.copy()
        val_tmp = val_df.copy()
        test_tmp = test_df.copy()
        val_oof_tmp[f"{name}_oof_prob"] = val_oof_prob
        val_tmp[f"{name}_prob"] = val_prob
        test_tmp[f"{name}_prob"] = test_prob

        val_oof_sel = select_by_scores(val_oof_tmp, f"{name}_oof_prob")
        val_sel = select_by_scores(val_tmp, f"{name}_prob")
        test_sel = select_by_scores(test_tmp, f"{name}_prob")

        val_oof_out = out_dir / f"val_candidate_rerank_ml_{name}_oof_on_val.csv"
        val_out = out_dir / f"val_candidate_rerank_ml_{name}_trained_on_val.csv"
        test_out = out_dir / f"test_candidate_rerank_ml_{name}_trained_on_val.csv"
        val_oof_sel.to_csv(val_oof_out, index=False)
        val_sel.to_csv(val_out, index=False)
        test_sel.to_csv(test_out, index=False)

        picked_oof = val_oof_tmp.sort_values(f"{name}_oof_prob", ascending=False).groupby("id").head(1)
        picked = val_tmp.sort_values(f"{name}_prob", ascending=False).groupby("id").head(1)
        print(f"\n=== {name} ===")
        print("[SAVED]", val_oof_out)
        print("[SAVED]", val_out)
        print("[SAVED]", test_out)
        print("val OOF oracle-pick-rate:", float(picked_oof["is_oracle"].mean()))
        print("val oracle-pick-rate:", float(picked["is_oracle"].mean()))
        print("test selected source:")
        print(test_sel["selected_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
