import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pandas as pd


def normalize_id(value):
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if re.fullmatch(r"0*\d+", text):
        return str(int(text))
    return re.sub(r"\s+", "", text).upper()


def normalize_source(value):
    return str(value or "").strip()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def replace_user_images(messages, image_paths):
    output = deepcopy(messages)
    for message in output:
        if message.get("role") != "user":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            text_blocks = [{"type": "text", "text": content}]
        else:
            text_blocks = [block for block in content if block.get("type") != "image"]
        message["content"] = [
            {"type": "image", "image": path} for path in image_paths
        ] + text_blocks
        return output
    raise ValueError("Record has no user message")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--prediction_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--image_types", default="raw,overlay,crop,select_overlay"
    )
    parser.add_argument("--skip_image_exists_check", action="store_true")
    args = parser.parse_args()

    image_types = [value.strip() for value in args.image_types.split(",") if value.strip()]
    manifest = pd.read_csv(args.prediction_manifest, dtype=str).fillna("")
    manifest["rank"] = manifest["rank"].astype(int)
    manifest["source_norm"] = manifest["source"].map(normalize_source)
    manifest["match_id"] = manifest["patient_id"].map(normalize_id)

    image_map = {}
    for (source, match_id), group in manifest.groupby(["source_norm", "match_id"]):
        group = group.sort_values("rank").head(args.top_k)
        paths = []
        meta = []
        for _, row in group.iterrows():
            for image_type in image_types:
                col = f"{image_type}_path"
                if col not in row or not row[col]:
                    raise ValueError(f"Missing {col} for {source}/{match_id}")
                path = str(row[col])
                if not args.skip_image_exists_check and not Path(path).exists():
                    raise FileNotFoundError(path)
                paths.append(path)
                meta.append(
                    {
                        "rank": int(row["rank"]),
                        "slice_idx": int(float(row["slice_idx"])),
                        "type": image_type,
                        "path": path,
                        "selection_policy": "prediction_only_all_slices",
                    }
                )
        image_map[(source, match_id)] = (paths, meta)

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "source_dir": str(source_dir),
        "prediction_manifest": args.prediction_manifest,
        "top_k": args.top_k,
        "image_types": image_types,
        "selection_policy": "prediction_only_all_slices",
        "gt_used_for_slice_selection": False,
        "splits": {},
    }

    all_missing = []
    for split in ["train", "val", "test"]:
        source_path = source_dir / f"{split}_raw_overlay_crop_messages.jsonl"
        rows = read_jsonl(source_path)
        output = []
        missing = []
        for row in rows:
            source = normalize_source(row.get("source", row.get("dataset_source", "")))
            match_id = normalize_id(row.get("patient_id", row.get("original_id", row.get("id", ""))))
            images = image_map.get((source, match_id))
            if images is None:
                missing.append(
                    {
                        "split": split,
                        "id": row.get("id", ""),
                        "source": source,
                        "patient_id": row.get("patient_id", ""),
                        "match_id": match_id,
                    }
                )
                continue
            image_paths, image_meta = images
            rebuilt = deepcopy(row)
            rebuilt["images"] = image_paths
            rebuilt["image_meta"] = image_meta
            rebuilt["messages"] = replace_user_images(rebuilt["messages"], image_paths)
            rebuilt["candidate_selection"] = {
                "policy": "prediction_only_all_slices",
                "top_k": args.top_k,
                "image_types": image_types,
                "gt_used_for_slice_selection": False,
            }
            rebuilt["experiment_version"] = "prediction_only_top1"
            output.append(rebuilt)

        out_path = out_dir / f"{split}_raw_overlay_crop_messages.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in output:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats["splits"][split] = {
            "input_rows": len(rows),
            "output_rows": len(output),
            "missing_rows": len(missing),
            "source_counts": dict(Counter(row.get("source", "") for row in output)),
            "output": str(out_path),
        }
        all_missing.extend(missing)

    missing_path = out_dir / "missing_prediction_only_cases.csv"
    pd.DataFrame(all_missing).to_csv(missing_path, index=False)
    stats["missing_path"] = str(missing_path)
    (out_dir / "build_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print("[DONE]")
    print(json.dumps(stats, indent=2))
    if all_missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
