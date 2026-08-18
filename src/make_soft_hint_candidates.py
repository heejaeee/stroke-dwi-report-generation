import argparse
import json
from pathlib import Path

import pandas as pd


ID_COLS = ["patient_id", "case_id", "id"]
PRED_LOC_COLS = [
    "pred_locations_ensemble_global",
    "pred_locations_ensemble_perclass",
    "pred_locations_perclass",
    "pred_locations_global",
]
PRED_LAT_COLS = ["pred_laterality_ensemble", "pred_laterality"]


def norm_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pick_col(df, cols, required=False):
    for c in cols:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing one of {cols}. Available={list(df.columns)}")
    return None


def split_locs(x):
    if pd.isna(x):
        return []
    return [v.strip() for v in str(x).split("|") if v.strip()]


def prob_cols(df):
    cols = [c for c in df.columns if c.startswith("prob_")]
    if cols:
        return cols
    return [c for c in df.columns if c.startswith("ens_prob_")]


def build_attr_map(attr_csv, topk, min_prob):
    df = pd.read_csv(attr_csv, dtype=str).fillna("")
    id_col = pick_col(df, ID_COLS, required=True)
    loc_col = pick_col(df, PRED_LOC_COLS, required=False)
    lat_col = pick_col(df, PRED_LAT_COLS, required=False)
    pcols = prob_cols(df)

    out = {}
    for _, row in df.iterrows():
        pid = norm_id(row[id_col])
        locs = split_locs(row[loc_col]) if loc_col else []
        laterality = str(row[lat_col]).strip().lower() if lat_col else ""

        probs = []
        for c in pcols:
            lab = c.replace("ens_prob_", "").replace("prob_", "")
            try:
                p = float(row[c])
            except Exception:
                continue
            probs.append((lab, p))
        probs.sort(key=lambda x: x[1], reverse=True)

        prob_locs = [lab for lab, p in probs if p >= min_prob]
        if topk > 0:
            prob_locs = [lab for lab, _ in probs[:topk] if not prob_locs or lab in prob_locs]
        if not prob_locs and topk > 0:
            prob_locs = [lab for lab, _ in probs[:topk]]

        merged = []
        for lab in locs + prob_locs:
            if lab and lab not in merged:
                merged.append(lab)

        out[pid] = {
            "laterality": laterality,
            "locations": merged,
            "top_probs": probs[: max(topk, 5)],
        }
    return out


def format_hint(attr):
    locations = attr.get("locations", [])
    laterality = attr.get("laterality", "")
    top_probs = attr.get("top_probs", [])

    loc_text = ", ".join(x.replace("_", " ") for x in locations) if locations else "no reliable location prior"
    prob_text = ", ".join(f"{lab.replace('_', ' ')}={p:.2f}" for lab, p in top_probs[:8])

    parts = [
        "Auxiliary anatomical prior from a DWI location encoder.",
        "The location encoder predicts a multi-label probability vector over the full anatomical label space, not a fixed short list.",
        f"Candidate location labels: {loc_text}.",
    ]
    if laterality:
        parts.append(f"Candidate laterality: {laterality}.")
    if prob_text:
        parts.append(f"Top label probabilities: {prob_text}.")
    parts.append(
        "This prior may be imperfect; use the DWI images and segmentation-derived visual evidence as the primary evidence."
    )
    return " ".join(parts)


def append_hint(record, hint):
    rec = json.loads(json.dumps(record, ensure_ascii=False))
    content = rec["messages"][0]["content"]
    text_idx = None
    for i, item in enumerate(content):
        if isinstance(item, dict) and item.get("type") == "text":
            text_idx = i

    extra = (
        "\n\nAdditional structured hint:\n"
        f"{hint}\n\n"
        "Final instruction: Generate exactly one concise radiology finding sentence for acute ischemic stroke on DWI. "
        "Mention laterality and anatomical location only when supported. "
        "Do not mention the auxiliary encoder, mask, overlay, or probabilities."
    )
    if text_idx is None:
        content.append({"type": "text", "text": extra.strip()})
    else:
        content[text_idx]["text"] = str(content[text_idx]["text"]).strip() + extra
    rec["full_labelspace_hint"] = hint
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_jsonl", required=True)
    ap.add_argument("--attr_csv", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--min_prob", type=float, default=0.30)
    args = ap.parse_args()

    attr = build_attr_map(args.attr_csv, args.topk, args.min_prob)
    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in read_jsonl(args.base_jsonl):
            pid = norm_id(rec.get("patient_id", rec.get("id", "")))
            if pid in attr:
                hint = format_hint(attr[pid])
            else:
                missing += 1
                hint = (
                    "Auxiliary anatomical prior is unavailable for this case. "
                    "Use the DWI images and segmentation-derived visual evidence as the primary evidence."
                )
            f.write(json.dumps(append_hint(rec, hint), ensure_ascii=False) + "\n")
            written += 1

    print("[SAVED]", out)
    print("written:", written)
    print("missing attr:", missing)
    if written:
        first = next(read_jsonl(out))
        print("sample id:", first.get("id"))
        print("sample hint:", first.get("full_labelspace_hint", "")[:500])


if __name__ == "__main__":
    main()
