import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.networks.nets import AttentionUnet
from PIL import Image
from tqdm import tqdm


def build_model():
    return AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        dropout=0.1,
    )


def load_model(path, device):
    model = build_model().to(device)
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state)
    model.eval()
    print(f"[MODEL] {path}")
    print(f"  epoch: {checkpoint.get('epoch')}")
    print(f"  val_pos_dice: {checkpoint.get('val_pos_dice')}")
    return model


def load_image_volume(path):
    volume = nib.load(str(path)).get_fdata(dtype=np.float32)
    volume = np.squeeze(volume)
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D image, got {volume.shape}: {path}")
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    finite = volume[np.isfinite(volume)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        if hi > lo:
            volume = np.clip(volume, lo, hi)
            volume = (volume - lo) / (hi - lo + 1e-6)
        else:
            volume = volume * 0.0
    return volume.astype(np.float32)


def load_mask_volume(path):
    volume = nib.load(str(path)).get_fdata(dtype=np.float32)
    volume = np.squeeze(volume)
    if volume.ndim == 4:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {volume.shape}: {path}")
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    return volume > 0


def volume_to_slices(volume, axis):
    return np.ascontiguousarray(np.moveaxis(volume, axis, 0))


def resize_slices(slices, size, is_mask=False):
    tensor = torch.from_numpy(slices.astype(np.float32, copy=False))[:, None]
    if is_mask:
        return F.interpolate(tensor, size=(size, size), mode="nearest") > 0.5
    return F.interpolate(
        tensor, size=(size, size), mode="bilinear", align_corners=False
    ).clamp_(0, 1)


@torch.no_grad()
def score_all_slices(model, image_slices, device, image_size, batch_size, threshold):
    rows = []
    for start in range(0, len(image_slices), batch_size):
        batch_np = image_slices[start : start + batch_size]
        batch = resize_slices(batch_np, image_size).to(device, non_blocking=True)
        probs = torch.sigmoid(model(batch)).cpu()
        pred = probs > threshold
        flat = probs.flatten(1)
        area = pred.sum(dim=(1, 2, 3))
        width = pred.shape[-1]
        left = pred[..., : width // 2].sum(dim=(1, 2, 3))
        right = pred[..., width // 2 :].sum(dim=(1, 2, 3))
        max_prob = flat.max(dim=1).values
        mean_top100 = torch.topk(flat, k=min(100, flat.shape[1]), dim=1).values.mean(1)

        for offset in range(len(batch_np)):
            rows.append(
                {
                    "slice_idx": start + offset,
                    "select_area": int(area[offset]),
                    "select_max_prob": float(max_prob[offset]),
                    "select_mean_top100_prob": float(mean_top100[offset]),
                    "select_left_area": int(left[offset]),
                    "select_right_area": int(right[offset]),
                }
            )
    return pd.DataFrame(rows)


def rank_all_slices(scores, min_select_pixels, rank_strategy="area_first"):
    scores = scores.copy()
    if rank_strategy == "area_first":
        candidates = scores[scores["select_area"] >= min_select_pixels].copy()
        candidates = candidates.sort_values(
            ["select_area", "select_max_prob", "select_mean_top100_prob", "slice_idx"],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        filler = scores[~scores.index.isin(candidates.index)].sort_values(
            ["select_max_prob", "select_mean_top100_prob", "slice_idx"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        ranked = pd.concat([candidates, filler]).reset_index(drop=True)
    elif rank_strategy == "max_prob":
        ranked = scores.sort_values(
            ["select_max_prob", "select_mean_top100_prob", "select_area", "slice_idx"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported rank strategy: {rank_strategy}")
    ranked["prediction_rank"] = np.arange(1, len(ranked) + 1)
    ranked["fallback"] = ranked["select_area"] < min_select_pixels
    return ranked


def choose_topk(scores, top_k, min_select_pixels, rank_strategy="area_first"):
    chosen = rank_all_slices(scores, min_select_pixels, rank_strategy).head(top_k).copy()
    chosen["rank"] = np.arange(1, len(chosen) + 1)
    return chosen


def to_uint8(image):
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def make_overlay(image, mask, color=(255, 0, 0), alpha=0.45):
    gray = to_uint8(image)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.asarray(color, np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def bbox_from_mask(mask, margin=16, min_size=64):
    ys, xs = np.where(mask)
    height, width = mask.shape
    if not len(xs):
        return 0, 0, width, height
    x1 = max(int(xs.min()) - margin, 0)
    y1 = max(int(ys.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin + 1, width)
    y2 = min(int(ys.max()) + margin + 1, height)
    if x2 - x1 < min_size:
        center = (x1 + x2) // 2
        x1 = max(center - min_size // 2, 0)
        x2 = min(x1 + min_size, width)
        x1 = max(0, x2 - min_size)
    if y2 - y1 < min_size:
        center = (y1 + y2) // 2
        y1 = max(center - min_size // 2, 0)
        y2 = min(y1 + min_size, height)
        y1 = max(0, y2 - min_size)
    return x1, y1, x2, y2


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def summarize_cases(case_df, rank_strategy):
    def summarize(group):
        first = group["first_positive_rank"].dropna()
        return {
            "cases": int(len(group)),
            "cases_without_gt_positive_slice": int(group["first_positive_rank"].isna().sum()),
            "hit_at_1": float(group["hit_at_1"].mean()),
            "hit_at_3": float(group["hit_at_3"].mean()),
            "hit_at_5": float(group["hit_at_5"].mean()),
            "median_first_positive_rank": float(first.median()) if len(first) else None,
            "mean_candidate_pool_size": float(group["candidate_pool_size"].mean()),
        }

    return {
        "overall": summarize(case_df),
        "by_source": {
            source: summarize(group) for source, group in case_df.groupby("source")
        },
        "selection_policy": "prediction_only_all_slices",
        "rank_strategy": rank_strategy,
        "gt_used_for_slice_selection": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_manifest", required=True)
    parser.add_argument("--selector_ckpt", required=True)
    parser.add_argument("--mask_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="prediction_only")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--th_select", type=float, default=0.90)
    parser.add_argument("--th_mask", type=float, default=0.40)
    parser.add_argument("--min_select_pixels", type=int, default=10)
    parser.add_argument(
        "--rank_strategy",
        default="area_first",
        choices=["area_first", "max_prob"],
    )
    parser.add_argument("--min_gt_pixels", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--crop_margin", type=int, default=16)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--max_cases_per_source", type=int, default=0)
    parser.add_argument("--write_gt_overlay", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] gpu: {torch.cuda.get_device_name(0)}")

    cases = pd.read_csv(args.case_manifest, dtype=str).fillna("")
    if args.max_cases_per_source > 0:
        cases = (
            cases.groupby("source", sort=True, group_keys=False)
            .head(args.max_cases_per_source)
            .reset_index(drop=True)
        )
    if args.max_cases > 0:
        cases = cases.head(args.max_cases).copy()
    if cases["case_uid"].duplicated().any():
        dup = cases[cases["case_uid"].duplicated(False)]["case_uid"].tolist()
        raise ValueError(f"Duplicate case_uid values: {dup[:20]}")

    out_dir = Path(args.out_dir)
    image_root = out_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    selector = load_model(args.selector_ckpt, device)
    mask_model = load_model(args.mask_ckpt, device)

    all_rows = []
    selected_rows = []
    manifest_rows = []
    case_audit_rows = []

    for _, case in tqdm(cases.iterrows(), total=len(cases), desc="prediction-only cases"):
        source = str(case["source"])
        patient_id = str(case["patient_id"])
        case_uid = str(case["case_uid"])
        axis = int(float(case.get("axis", 2)))
        image_path = Path(case["image_path"])
        mask_path = Path(case["mask_path"])

        image_volume = load_image_volume(image_path)
        image_slices = volume_to_slices(image_volume, axis)

        # Selection is completed before the GT mask is loaded.
        scores = score_all_slices(
            selector,
            image_slices,
            device,
            args.image_size,
            args.batch_size,
            args.th_select,
        )
        chosen = choose_topk(
            scores,
            args.top_k,
            args.min_select_pixels,
            args.rank_strategy,
        )

        mask_volume = load_mask_volume(mask_path)
        if mask_volume.shape != image_volume.shape:
            raise ValueError(
                f"Image/mask mismatch for {case_uid}: "
                f"{image_volume.shape} vs {mask_volume.shape}"
            )
        mask_slices = volume_to_slices(mask_volume, axis)
        gt_pixels = mask_slices.reshape(len(mask_slices), -1).sum(1).astype(int)
        scores["mask_pixels"] = gt_pixels
        scores["is_positive"] = (gt_pixels >= args.min_gt_pixels).astype(int)
        rank_order = rank_all_slices(
            scores,
            args.min_select_pixels,
            args.rank_strategy,
        )
        rank_map = rank_order.set_index("slice_idx")["prediction_rank"]
        scores["prediction_rank"] = scores["slice_idx"].map(rank_map).astype(int)
        positive_ranks = np.flatnonzero(rank_order["is_positive"].to_numpy() == 1) + 1
        first_positive = float(positive_ranks[0]) if len(positive_ranks) else np.nan
        case_audit_rows.append(
            {
                "case_uid": case_uid,
                "source": source,
                "patient_id": patient_id,
                "split": case["split"],
                "candidate_pool_size": len(scores),
                "gt_positive_slices": int(scores["is_positive"].sum()),
                "first_positive_rank": first_positive,
                "hit_at_1": int(len(positive_ranks) and positive_ranks[0] <= 1),
                "hit_at_3": int(len(positive_ranks) and positive_ranks[0] <= 3),
                "hit_at_5": int(len(positive_ranks) and positive_ranks[0] <= 5),
                "selection_policy": "prediction_only_all_slices",
                "rank_strategy": args.rank_strategy,
                "gt_used_for_slice_selection": False,
            }
        )

        common = {
            "case_uid": case_uid,
            "source": source,
            "patient_id": patient_id,
            "match_id": case["match_id"],
            "split": case["split"],
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "axis": axis,
            "candidate_pool_size": len(scores),
            "selection_policy": "prediction_only_all_slices",
            "rank_strategy": args.rank_strategy,
            "gt_used_for_selection": False,
        }
        for row in scores.to_dict("records"):
            all_rows.append({**common, **row})

        chosen_indices = chosen["slice_idx"].astype(int).to_numpy()
        chosen_images = resize_slices(
            image_slices[chosen_indices], args.image_size
        ).to(device)
        with torch.no_grad():
            select_probs = torch.sigmoid(selector(chosen_images)).cpu().numpy()[:, 0]
            shape_probs = torch.sigmoid(mask_model(chosen_images)).cpu().numpy()[:, 0]
        chosen_images_np = chosen_images.cpu().numpy()[:, 0]
        chosen_gt = resize_slices(
            mask_slices[chosen_indices].astype(np.float32), args.image_size, is_mask=True
        ).numpy()[:, 0]

        case_dir = image_root / f"pid_{safe_name(case_uid)}"
        case_dir.mkdir(parents=True, exist_ok=True)

        for j, chosen_row in chosen.iterrows():
            rank = int(chosen_row["rank"])
            slice_idx = int(chosen_row["slice_idx"])
            image = chosen_images_np[j]
            select_mask = select_probs[j] > args.th_select
            shape_mask = shape_probs[j] > args.th_mask
            if not shape_mask.any() and select_mask.any():
                shape_mask = select_mask
            gt_mask = chosen_gt[j]

            prefix = f"rank{rank:02d}_slice{slice_idx:04d}"
            raw_path = case_dir / f"{prefix}_raw.png"
            overlay_path = case_dir / f"{prefix}_overlay.png"
            crop_path = case_dir / f"{prefix}_crop.png"
            select_path = case_dir / f"{prefix}_select090.png"
            gt_path = case_dir / f"{prefix}_gt.png"

            Image.fromarray(to_uint8(image), mode="L").save(raw_path)
            Image.fromarray(make_overlay(image, shape_mask), mode="RGB").save(overlay_path)
            Image.fromarray(make_overlay(image, select_mask), mode="RGB").save(select_path)
            x1, y1, x2, y2 = bbox_from_mask(shape_mask, args.crop_margin)
            crop = make_overlay(image, shape_mask)[y1:y2, x1:x2]
            Image.fromarray(crop, mode="RGB").save(crop_path)
            if args.write_gt_overlay:
                Image.fromarray(make_overlay(image, gt_mask), mode="RGB").save(gt_path)

            selected_record = {
                **common,
                **chosen_row.to_dict(),
                "is_positive": int(gt_pixels[slice_idx] >= args.min_gt_pixels),
                "gt_area": int(gt_mask.sum()),
                "shape_mask_area": int(shape_mask.sum()),
                "raw_path": str(raw_path),
                "overlay_path": str(overlay_path),
                "crop_path": str(crop_path),
                "select_overlay_path": str(select_path),
                "gt_overlay_path": str(gt_path) if args.write_gt_overlay else "",
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
            }
            selected_rows.append(selected_record)
            manifest_rows.append(selected_record)

    all_df = pd.DataFrame(all_rows)
    selected_df = pd.DataFrame(selected_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    case_audit_df = pd.DataFrame(case_audit_rows)

    paths = {
        "all_scores": out_dir / f"{args.prefix}_all_slice_scores.csv",
        "selected": out_dir / f"{args.prefix}_selected_top{args.top_k}.csv",
        "manifest": out_dir / f"{args.prefix}_caption_input_manifest.csv",
        "case_audit": out_dir / f"{args.prefix}_case_selection_audit.csv",
        "summary": out_dir / f"{args.prefix}_selection_summary.json",
    }
    all_df.to_csv(paths["all_scores"], index=False)
    selected_df.to_csv(paths["selected"], index=False)
    manifest_df.to_csv(paths["manifest"], index=False)
    case_audit_df.to_csv(paths["case_audit"], index=False)
    summary = summarize_cases(case_audit_df, args.rank_strategy)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[DONE]")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
