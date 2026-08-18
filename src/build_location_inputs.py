import argparse
import json
import re
from pathlib import Path

import pandas as pd


LOC_PATTERNS = {
    "aca_territory": [r"\baca\b", r"anterior cerebral"],
    "basal_ganglia": [r"basal ganglia", r"\bbg\b", r"lentiform", r"putamen", r"caudate"],
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
    "pons": [r"\bpons\b", r"pontine", r"pontomedullary"],
    "temporal_lobe": [r"temporal"],
    "thalamus": [r"thalam"],
    "white_matter": [r"white matter"],
}

LAT_PATTERNS = {
    "left": [r"\bleft\b"],
    "right": [r"\bright\b"],
    "bilateral": [r"\bbilateral\b", r"\bboth\b"],
}


def parse_laterality(text):
    s = str(text).lower().replace("-", " ")
    left = any(re.search(p, s) for p in LAT_PATTERNS["left"])
    right = any(re.search(p, s) for p in LAT_PATTERNS["right"])
    bilateral = any(re.search(p, s) for p in LAT_PATTERNS["bilateral"])
    if bilateral or (left and right):
        return "bilateral"
    if left:
        return "left"
    if right:
        return "right"
    return "unknown"


def parse_locations(text):
    s = str(text).lower().replace("-", " ")
    locs = []
    for loc, patterns in LOC_PATTERNS.items():
        if any(re.search(p, s) for p in patterns):
            locs.append(loc)
    return locs


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pick_image(images, suffix):
    for p in images:
        if str(p).endswith(suffix):
            return p
    return ""


def convert(jsonl, out_csv):
    rows = []
    for rec in read_jsonl(jsonl):
        images = rec.get("images", [])
        target = rec.get("target", "")
        locs = parse_locations(target)
        lat = parse_laterality(target)
        raw = pick_image(images, "_raw.png")
        overlay = pick_image(images, "_overlay.png")
        crop = pick_image(images, "_crop.png")
        select = pick_image(images, "_select090.png")
        rows.append(
            {
                "id": rec.get("id", ""),
                "case_id": rec.get("patient_id", rec.get("id", "")),
                "patient_id": rec.get("patient_id", rec.get("id", "")),
                "source": rec.get("source", ""),
                "dataset_source": rec.get("dataset_source", ""),
                "split": rec.get("split", ""),
                "target": target,
                "true_laterality": lat,
                "parsed_laterality_acute": lat,
                "true_locations": "|".join(locs),
                "parsed_locations_acute": "|".join(locs),
                "raw_path": raw,
                "overlay_path": overlay,
                "crop_path": crop,
                "select_overlay_path": select,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("[SAVED]", out_csv)
    print("rows:", len(df))
    if len(df):
        print("source:")
        print(df["source"].value_counts(dropna=False).to_string())
        print("laterality:")
        print(df["true_laterality"].value_counts(dropna=False).to_string())
        print("empty locations:", int((df["true_locations"].fillna("") == "").sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    convert(args.jsonl, args.out_csv)


if __name__ == "__main__":
    main()
