import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

LOC_PATTERNS = {
    "aca_territory": [r"\baca\b", r"anterior cerebral"],
    "basal_ganglia": [r"basal ganglia", r"lentiform", r"putamen", r"caudate"],
    "centrum_semiovale": [r"centrum semiovale"],
    "cerebellum": [r"cerebell", r"vermis"],
    "cerebral_hemisphere": [r"cerebral hemisphere"],
    "cingulate_gyrus": [r"cingulate"],
    "corona_radiata": [r"corona radiata"],
    "corpus_callosum": [r"corpus callosum"],
    "cortex": [r"cortex", r"cortical"],
    "frontal_lobe": [r"frontal"],
    "hippocampus": [r"hippocamp"],
    "ica_territory": [r"\bica\b", r"internal carotid"],
    "insula": [r"insula", r"insular"],
    "internal_capsule": [r"internal capsule"],
    "mca_territory": [r"\bmca\b", r"middle cerebral"],
    "medulla": [r"medulla", r"medullary"],
    "midbrain": [r"midbrain"],
    "occipital_lobe": [r"occipital"],
    "parietal_lobe": [r"parietal"],
    "pca_territory": [r"\bpca\b", r"posterior cerebral"],
    "pons": [r"\bpons\b", r"pontine"],
    "temporal_lobe": [r"temporal"],
    "thalamus": [r"thalam"],
    "white_matter": [r"white matter"],
}

ID_COLS = ["patient_id", "case_id", "id"]
TARGET_COLS = ["target", "acute_target_sentence", "clean_target", "caption", "report", "finding"]

def norm_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def split_locs(x):
    if pd.isna(x):
        return set()
    s = str(x).strip()
    if not s:
        return set()
    return {v.strip() for v in s.split("|") if v.strip()}

def pick_col(df, cols, required=True):
    for c in cols:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing one of {cols}. Available={list(df.columns)}")
    return None

def parse_laterality(s):
    t = str(s).lower()
    left = bool(re.search(r"\bleft\b", t))
    right = bool(re.search(r"\bright\b", t))
    bilat = "bilateral" in t or re.search(r"\bboth\b", t) is not None
    if bilat or (left and right):
        return "bilateral"
    if left:
        return "left"
    if right:
        return "right"
    return "unknown"

def parse_locations(s):
    t = str(s).lower().replace("-", " ")
    out = set()
    for loc, pats in LOC_PATTERNS.items():
        if any(re.search(p, t) for p in pats):
            out.add(loc)
    return out

def tuple_set(text):
    lat = parse_laterality(text)
    locs = parse_locations(text)
    if not locs:
        return set()
    return {(lat, loc) for loc in locs}

def set_f1(y, p):
    if not y and not p:
        return 1.0
    if not y or not p:
        return 0.0
    inter = len(y & p)
    prec = inter / len(p) if p else 0
    rec = inter / len(y) if y else 0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0

def load_candidates(paths):
    by_id = defaultdict(list)
    targets = {}

    for path in paths:
        df = pd.read_csv(path).fillna("")
        id_col = "id" if "id" in df.columns else pick_col(df, ID_COLS)
        patient_col = "patient_id" if "patient_id" in df.columns else id_col
        target_col = pick_col(df, TARGET_COLS)
        pred_col = "prediction" if "prediction" in df.columns else "pred"

        source = Path(path).stem

        for _, r in df.iterrows():
            pid = norm_id(r[id_col])
            patient_id = norm_id(r[patient_col])
            pred = str(r[pred_col]).strip()
            target = str(r[target_col]).strip()
            if target:
                targets[pid] = target

            by_id[pid].append({
                "id": pid,
                "patient_id": patient_id,
                "source": source,
                "prediction": pred,
                "target": target,
                "lat": parse_laterality(pred),
                "locs": parse_locations(pred),
                "tuples": tuple_set(pred),
                "pred_len": len(pred.split()),
            })

    return by_id, targets

def load_attr(path):
    if not path:
        return {}

    df = pd.read_csv(path).fillna("")
    id_col = pick_col(df, ID_COLS)
    loc_col = "pred_locations_ensemble_global" if "pred_locations_ensemble_global" in df.columns else "pred_locations_global"
    if "pred_laterality_ensemble" in df.columns:
        lat_col = "pred_laterality_ensemble"
    elif "pred_laterality" in df.columns:
        lat_col = "pred_laterality"
    else:
        lat_col = None

    out = {}
    for _, r in df.iterrows():
        pid = norm_id(r[id_col])
        out[pid] = {
            "lat": str(r[lat_col]).strip().lower() if lat_col else "unknown",
            "locs": split_locs(r[loc_col]),
        }
    return out

def score_candidate(c, attr, mode):
    locs = c["locs"]
    lat = c["lat"]

    attr_locs = attr.get("locs", set())
    attr_lat = attr.get("lat", "unknown")

    overlap = len(locs & attr_locs)
    extra = len(locs - attr_locs)
    missing = len(attr_locs - locs)

    lat_match = float(lat == attr_lat and lat != "unknown")
    lat_bilat_soft = float(lat == "bilateral" and attr_lat in {"bilateral", "left", "right"})

    if mode == "balanced":
        score = 2.0 * overlap + 1.0 * lat_match + 0.3 * lat_bilat_soft - 0.7 * extra - 0.15 * missing
    elif mode == "precision":
        score = 2.2 * overlap + 1.0 * lat_match - 1.2 * extra - 0.05 * missing
    elif mode == "recall":
        score = 1.3 * overlap + 0.8 * lat_match + 0.25 * len(locs) - 0.25 * extra - 0.10 * missing
    elif mode == "hybrid_preferred":
        score = 2.0 * overlap + 1.0 * lat_match - 0.6 * extra - 0.15 * missing
        if "hybrid" in c["source"]:
            score += 0.6
        if "base_plus_consensus2_cap3" in c["source"]:
            score += 0.3
    elif mode == "calibrated_v1":
        score = 2.1 * overlap + 1.0 * lat_match - 0.9 * extra - 0.10 * missing
        if c["source"] == "test_predictions_hybrid_loc_hint_perclass":
            score += 0.35
        if c["source"] == "test_predictions":
            score -= 0.20
        if c["source"] == "base_plus_consensus2_cap3":
            score += 0.15
        if c["source"] == "base_locs_bilateral_if_rank_bilateral":
            score -= 0.25
    elif mode == "calibrated_v2":
        score = 2.0 * overlap + 1.0 * lat_match - 0.8 * extra - 0.12 * missing
        if c["source"] == "test_predictions_hybrid_loc_hint_perclass":
            score += 0.55
        if c["source"] == "test_predictions":
            score -= 0.35
        if c["source"] == "base_plus_consensus2_cap3":
            score += 0.20
        if c["source"] == "base_locs_bilateral_if_rank_bilateral":
            score -= 0.30
    elif mode == "calibrated_v3":
        score = 2.2 * overlap + 1.1 * lat_match - 1.0 * extra - 0.05 * missing
        if c["source"] == "test_predictions_hybrid_loc_hint_perclass":
            score += 0.25
        if c["source"] == "test_predictions":
            score -= 0.10
        if c["source"] == "base_plus_consensus2_cap3":
            score += 0.10
    else:
        raise ValueError(mode)

    if not locs:
        score -= 2.0
    if c["pred_len"] < 5 or c["pred_len"] > 20:
        score -= 0.2

    return score

def choose_oracle(cands, target):
    y_tuple = tuple_set(target)
    y_loc = parse_locations(target)

    best = None
    for c in cands:
        tf1 = set_f1(y_tuple, c["tuples"])
        lf1 = set_f1(y_loc, c["locs"])
        exact = float(y_tuple == c["tuples"])
        key = (tf1, exact, lf1, -c["pred_len"])
        if best is None or key > best[0]:
            best = (key, c)
    return best[1]

def write_selected(by_id, targets, attr, out_dir, mode):
    rows = []

    for pid, cands in by_id.items():
        target = targets.get(pid, cands[0].get("target", ""))
        if mode == "oracle_tuple_f1":
            chosen = choose_oracle(cands, target)
        else:
            patient_id = cands[0].get("patient_id", pid)
            a = attr.get(pid, attr.get(patient_id, {"lat": "unknown", "locs": set()}))
            chosen = max(cands, key=lambda c: score_candidate(c, a, mode))

        rows.append({
            "id": pid,
            "patient_id": chosen.get("patient_id", pid),
            "target": target,
            "prediction": chosen["prediction"],
            "selected_source": chosen["source"],
            "selected_laterality": chosen["lat"],
            "selected_locations": "|".join(sorted(chosen["locs"])),
        })

    out = Path(out_dir) / f"candidate_rerank_{mode}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("[SAVED]", out, "rows:", len(rows))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--attr_csv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    by_id, targets = load_candidates(args.candidates)
    attr = load_attr(args.attr_csv)

    print("[INFO] cases:", len(by_id))
    print("[INFO] attr cases:", len(attr))
    print("[INFO] candidates:")
    for p in args.candidates:
        print(" ", p)

    for mode in ["balanced", "precision", "recall", "hybrid_preferred", "calibrated_v1", "calibrated_v2", "calibrated_v3", "oracle_tuple_f1"]:
        write_selected(by_id, targets, attr, args.out_dir, mode)

if __name__ == "__main__":
    main()
