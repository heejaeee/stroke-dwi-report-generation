import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def load_export_helpers():
    path = Path(__file__).with_name("prediction_only_utils.py")
    spec = importlib.util.spec_from_file_location("prediction_export_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def files_exist(record):
    columns = ["raw_path", "overlay_path", "crop_path", "select_overlay_path"]
    return all(Path(str(record[column])).is_file() for column in columns)


RANKING_STRATEGIES = {
    "area_first": (
        ["select_area", "select_max_prob", "select_mean_top100_prob", "slice_idx"],
        [False, False, False, True],
    ),
    "max_prob": (
        ["select_max_prob", "select_mean_top100_prob", "select_area", "slice_idx"],
        [False, False, False, True],
    ),
    "top100_mean": (
        ["select_mean_top100_prob", "select_max_prob", "select_area", "slice_idx"],
        [False, False, False, True],
    ),
}


def rank_slices(frame, strategy):
    columns, ascending = RANKING_STRATEGIES[strategy]
    return frame.sort_values(
        columns,
        ascending=ascending,
        kind="mergesort",
    )


def summarize(case_audit, rank_strategy):
    def one(frame):
        positive = frame["first_positive_rank"].dropna()
        return {
            "cases": int(len(frame)),
            "cases_without_gt_positive_slice": int(
                frame["first_positive_rank"].isna().sum()
            ),
            "hit_at_1": float(frame["hit_at_1"].mean()),
            "hit_at_3": float(frame["hit_at_3"].mean()),
            "hit_at_5": float(frame["hit_at_5"].mean()),
            "median_first_positive_rank": (
                float(positive.median()) if len(positive) else None
            ),
        }

    return {
        "overall": one(case_audit),
        "by_split": {
            split: one(frame) for split, frame in case_audit.groupby("split")
        },
        "by_source": {
            source: one(frame) for source, frame in case_audit.groupby("source")
        },
        "selection_policy": "prediction_only_all_slices",
        "rank_strategy": rank_strategy,
        "ranking_columns": RANKING_STRATEGIES[rank_strategy][0],
        "gt_used_for_slice_selection": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_scores_csv", required=True)
    parser.add_argument("--reuse_manifest", required=True)
    parser.add_argument("--selector_ckpt", required=True)
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="prediction_only_top1")
    parser.add_argument("--th_select", type=float, default=0.90)
    parser.add_argument("--th_mask", type=float, default=0.40)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--crop_margin", type=int, default=16)
    parser.add_argument(
        "--rank_strategy",
        default="max_prob",
        choices=sorted(RANKING_STRATEGIES),
    )
    args = parser.parse_args()

    helpers = load_export_helpers()
    scores = pd.read_csv(args.all_scores_csv, low_memory=False)
    reuse = pd.read_csv(args.reuse_manifest, low_memory=False)

    required = {
        "case_uid",
        "source",
        "patient_id",
        "split",
        "image_path",
        "mask_path",
        "axis",
        "slice_idx",
        "is_positive",
        "select_area",
        "select_max_prob",
        "select_mean_top100_prob",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Missing all-score columns: {missing}")

    reuse_lookup = {
        (str(row["case_uid"]), int(row["slice_idx"])): row
        for row in reuse.to_dict("records")
    }

    ranked = (
        scores.groupby("case_uid", sort=False, group_keys=False)
        .apply(lambda frame: rank_slices(frame, args.rank_strategy))
        .reset_index(drop=True)
    )
    selected = ranked.groupby("case_uid", sort=False).head(1).copy()

    out_dir = Path(args.out_dir)
    image_root = out_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] gpu: {torch.cuda.get_device_name(0)}")

    selector = None
    mask_model = None
    records = []
    reused_cases = 0
    generated_cases = 0

    for row in tqdm(
        selected.to_dict("records"),
        total=len(selected),
        desc=f"reexport {args.rank_strategy} rank1",
    ):
        case_uid = str(row["case_uid"])
        slice_idx = int(row["slice_idx"])
        old = reuse_lookup.get((case_uid, slice_idx))

        record = dict(row)
        record.update(
            {
                "rank": 1,
                "prediction_rank": 1,
                "selection_policy": "prediction_only_all_slices",
                "rank_strategy": args.rank_strategy,
                "gt_used_for_selection": False,
            }
        )

        if old is not None and files_exist(old):
            for column in [
                "shape_mask_area",
                "raw_path",
                "overlay_path",
                "crop_path",
                "select_overlay_path",
                "gt_overlay_path",
                "crop_x1",
                "crop_y1",
                "crop_x2",
                "crop_y2",
            ]:
                if column in old:
                    record[column] = old[column]
            record["reused_existing_view"] = True
            reused_cases += 1
            records.append(record)
            continue

        if selector is None:
            selector = helpers.load_model(args.selector_ckpt, device)
            mask_model = helpers.load_model(args.mask_ckpt, device)

        image_volume = helpers.load_image_volume(Path(str(row["image_path"])))
        axis = int(float(row["axis"]))
        image_slices = helpers.volume_to_slices(image_volume, axis)
        image_tensor = helpers.resize_slices(
            image_slices[[slice_idx]], args.image_size
        ).to(device)

        with torch.no_grad():
            select_prob = torch.sigmoid(selector(image_tensor)).cpu().numpy()[0, 0]
            shape_prob = torch.sigmoid(mask_model(image_tensor)).cpu().numpy()[0, 0]
        image = image_tensor.cpu().numpy()[0, 0]
        select_mask = select_prob > args.th_select
        shape_mask = shape_prob > args.th_mask
        if not shape_mask.any() and select_mask.any():
            shape_mask = select_mask

        case_dir = image_root / f"pid_{helpers.safe_name(case_uid)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        stem = f"rank01_slice{slice_idx:04d}"
        raw_path = case_dir / f"{stem}_raw.png"
        overlay_path = case_dir / f"{stem}_overlay.png"
        crop_path = case_dir / f"{stem}_crop.png"
        select_path = case_dir / f"{stem}_select090.png"

        Image.fromarray(helpers.to_uint8(image), mode="L").save(raw_path)
        Image.fromarray(
            helpers.make_overlay(image, shape_mask), mode="RGB"
        ).save(overlay_path)
        Image.fromarray(
            helpers.make_overlay(image, select_mask), mode="RGB"
        ).save(select_path)
        x1, y1, x2, y2 = helpers.bbox_from_mask(shape_mask, args.crop_margin)
        crop = helpers.make_overlay(image, shape_mask)[y1:y2, x1:x2]
        Image.fromarray(crop, mode="RGB").save(crop_path)

        record.update(
            {
                "shape_mask_area": int(shape_mask.sum()),
                "raw_path": str(raw_path),
                "overlay_path": str(overlay_path),
                "crop_path": str(crop_path),
                "select_overlay_path": str(select_path),
                "gt_overlay_path": "",
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "reused_existing_view": False,
            }
        )
        generated_cases += 1
        records.append(record)

    case_rows = []
    for case_uid, case in tqdm(
        scores.groupby("case_uid", sort=False),
        total=scores["case_uid"].nunique(),
        desc="selection QA",
    ):
        order = rank_slices(case, args.rank_strategy)
        positive = np.flatnonzero(order["is_positive"].to_numpy(dtype=int) == 1) + 1
        first = int(positive[0]) if len(positive) else None
        first_row = order.iloc[0]
        case_rows.append(
            {
                "case_uid": case_uid,
                "source": first_row["source"],
                "patient_id": first_row["patient_id"],
                "split": first_row["split"],
                "candidate_pool_size": int(len(order)),
                "gt_positive_slices": int(order["is_positive"].sum()),
                "first_positive_rank": first,
                "hit_at_1": int(first is not None and first <= 1),
                "hit_at_3": int(first is not None and first <= 3),
                "hit_at_5": int(first is not None and first <= 5),
                "selection_policy": "prediction_only_all_slices",
                "rank_strategy": args.rank_strategy,
                "gt_used_for_slice_selection": False,
            }
        )

    manifest = pd.DataFrame(records)
    case_audit = pd.DataFrame(case_rows)
    paths = {
        "selected": out_dir / f"{args.prefix}_selected_top1.csv",
        "manifest": out_dir / f"{args.prefix}_caption_input_manifest.csv",
        "case_audit": out_dir / f"{args.prefix}_case_selection_audit.csv",
        "summary": out_dir / f"{args.prefix}_selection_summary.json",
    }
    manifest.to_csv(paths["selected"], index=False)
    manifest.to_csv(paths["manifest"], index=False)
    case_audit.to_csv(paths["case_audit"], index=False)

    summary = summarize(case_audit, args.rank_strategy)
    summary["view_export"] = {
        "reused_existing_cases": reused_cases,
        "generated_cases": generated_cases,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[DONE]")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
