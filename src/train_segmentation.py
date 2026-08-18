import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss
from monai.networks.nets import AttentionUnet
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CachedSliceDataset(Dataset):
    def __init__(self, path, train=False, source_filter=""):
        data = torch.load(path, map_location="cpu")
        self.images = data["images"]
        self.masks = data["masks"]
        patient_ids = [str(value) for value in data["patient_id"]]
        raw_sources = data.get("source")
        if raw_sources is None:
            raise ValueError(
                "The cache must include a source value for every slice. "
                "Do not infer institutions from patient identifiers."
            )
        sources = [str(value) for value in raw_sources]
        if source_filter:
            self.indices = [index for index, source in enumerate(sources) if source == source_filter]
        else:
            self.indices = list(range(len(patient_ids)))
        self.labels = data["is_positive"][self.indices].long()
        self.patient_ids = [patient_ids[index] for index in self.indices]
        self.sources = [sources[index] for index in self.indices]
        self.train = train
        print(f"[CACHE] {path}")
        print(f"  source filter: {source_filter or 'all'}")
        print(f"  images: {tuple(self.images.shape)} {self.images.dtype}")
        print(f"  masks : {tuple(self.masks.shape)} {self.masks.dtype}")
        print(f"  selected rows: {len(self.indices)}")
        print(f"  cases : {len(set(self.patient_ids))}")
        print(f"  source rows: {dict(Counter(self.sources))}")
        print(f"  labels: {dict(Counter(self.labels.tolist()))}")

    def __len__(self):
        return len(self.indices)

    def augment(self, image, mask):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        if random.random() < 0.4:
            image = torch.clamp(
                image * random.uniform(0.85, 1.15) + random.uniform(-0.07, 0.07),
                0,
                1,
            )
        if random.random() < 0.25:
            image = torch.clamp(image + torch.randn_like(image) * random.uniform(0.01, 0.035), 0, 1)
        if random.random() < 0.15:
            gamma = random.uniform(0.8, 1.25)
            image = torch.clamp(image, 1e-6, 1).pow(gamma)
        return image, mask

    def __getitem__(self, index):
        raw_index = self.indices[index]
        image = self.images[raw_index].float()
        mask = self.masks[raw_index].float()
        if self.train:
            image, mask = self.augment(image, mask)
        return {
            "image": image,
            "mask": mask,
            "is_positive": self.labels[index],
            "patient_id": self.patient_ids[index],
            "source": self.sources[index],
        }


def build_model():
    return AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        dropout=0.1,
    )


def load_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint)), checkpoint


def make_sampler(dataset, positive_weight, source_balance_power):
    source_counts = Counter(dataset.sources)
    total = len(dataset)
    num_sources = len(source_counts)
    weights = []
    for label, source in zip(dataset.labels.tolist(), dataset.sources):
        source_weight = (total / (num_sources * source_counts[source])) ** source_balance_power
        class_weight = positive_weight if int(label) == 1 else 1.0
        weights.append(source_weight * class_weight)
    weights = torch.as_tensor(weights, dtype=torch.double)
    print("[SAMPLER] source counts:", dict(source_counts))
    print("[SAMPLER] positive weight:", positive_weight)
    print("[SAMPLER] source balance power:", source_balance_power)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def append_history(path, row):
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc="train")
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = loss_fn(logits, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        progress.set_postfix(loss=total_loss / batches)
    return total_loss / max(batches, 1)


def rank_case(rows, min_select_pixels):
    candidates = [row for row in rows if row["area"] >= min_select_pixels]
    filler = [row for row in rows if row["area"] < min_select_pixels]
    candidates.sort(key=lambda row: (-row["area"], -row["max_prob"], -row["mean_top100"], row["order"]))
    filler.sort(key=lambda row: (-row["max_prob"], -row["mean_top100"], row["order"]))
    return candidates + filler


@torch.no_grad()
def validate(model, loader, loss_fn, device, mask_threshold, select_threshold, min_select_pixels):
    model.eval()
    losses = []
    dice_values = []
    source_dice = defaultdict(list)
    case_rows = defaultdict(list)
    source_by_case = {}
    tp = fp = fn = tn = 0
    order = 0

    for batch in tqdm(loader, desc="val"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(loss_fn(logits, masks).item()))
        probs = torch.sigmoid(logits)

        mask_pred = probs > mask_threshold
        gt_positive = masks.sum(dim=(1, 2, 3)) > 0
        pred_area_mask = mask_pred.sum(dim=(1, 2, 3))
        pred_positive = pred_area_mask >= min_select_pixels
        tp += int((pred_positive & gt_positive).sum())
        fp += int((pred_positive & ~gt_positive).sum())
        fn += int((~pred_positive & gt_positive).sum())
        tn += int((~pred_positive & ~gt_positive).sum())

        inter = (mask_pred.float() * masks).sum(dim=(1, 2, 3))
        denom = mask_pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (denom + 1e-6)

        selector = probs > select_threshold
        area = selector.sum(dim=(1, 2, 3)).cpu().tolist()
        flat = probs.flatten(1)
        max_prob = flat.max(1).values.cpu().tolist()
        mean_top100 = torch.topk(flat, min(100, flat.shape[1]), dim=1).values.mean(1).cpu().tolist()
        labels = gt_positive.cpu().tolist()
        patient_ids = list(batch["patient_id"])
        sources = list(batch["source"])
        dice_cpu = dice.cpu().tolist()

        for idx, patient_id in enumerate(patient_ids):
            if labels[idx]:
                dice_values.append(dice_cpu[idx])
                source_dice[sources[idx]].append(dice_cpu[idx])
            case_rows[patient_id].append(
                {
                    "area": int(area[idx]),
                    "max_prob": float(max_prob[idx]),
                    "mean_top100": float(mean_top100[idx]),
                    "gt_positive": bool(labels[idx]),
                    "order": order,
                }
            )
            source_by_case[patient_id] = sources[idx]
            order += 1

    hits = {1: [], 3: [], 5: []}
    source_hits = defaultdict(lambda: {1: [], 3: [], 5: []})
    for patient_id, rows in case_rows.items():
        ranked = rank_case(rows, min_select_pixels)
        source = source_by_case[patient_id]
        for k in hits:
            hit = any(row["gt_positive"] for row in ranked[:k])
            hits[k].append(float(hit))
            source_hits[source][k].append(float(hit))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    slice_f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    result = {
        "val_loss": float(np.mean(losses)),
        "val_pos_dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "val_slice_precision": precision,
        "val_slice_recall": recall,
        "val_slice_f1": slice_f1,
        "val_hit1": float(np.mean(hits[1])),
        "val_hit3": float(np.mean(hits[3])),
        "val_hit5": float(np.mean(hits[5])),
    }
    for source in sorted(source_hits):
        key = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_").lower()
        result[f"val_{key}_pos_dice"] = float(np.mean(source_dice[source])) if source_dice[source] else 0.0
        for k in hits:
            result[f"val_{key}_hit{k}"] = float(np.mean(source_hits[source][k]))
    return result


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_pos_dice": metrics["val_pos_dice"],
            "val_hit1": metrics["val_hit1"],
            "val_hit3": metrics["val_hit3"],
            "metrics": metrics,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--init_ckpt", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--positive_weight", type=float, default=1.5)
    parser.add_argument("--source_balance_power", type=float, default=0.5)
    parser.add_argument("--train_source", default="")
    parser.add_argument("--val_source", default="")
    parser.add_argument("--mask_threshold", type=float, default=0.40)
    parser.add_argument("--select_threshold", type=float, default=0.90)
    parser.add_argument("--min_select_pixels", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    if torch.cuda.is_available():
        print("[INFO] gpu:", torch.cuda.get_device_name(0))

    train_ds = CachedSliceDataset(args.train_cache, train=True, source_filter=args.train_source)
    val_ds = CachedSliceDataset(args.val_cache, train=False, source_filter=args.val_source)
    sampler = make_sampler(train_ds, args.positive_weight, args.source_balance_power)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model().to(device)
    if args.init_ckpt:
        state, checkpoint = load_state(args.init_ckpt)
        model.load_state_dict(state)
        print(f"[INIT] {args.init_ckpt} epoch={checkpoint.get('epoch')}")

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and torch.cuda.is_available() else None

    best_mask = -math.inf
    best_selector = -math.inf
    best_joint = -math.inf
    stale = 0
    history_path = out_dir / "history.csv"
    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        metrics = validate(
            model,
            val_loader,
            loss_fn,
            device,
            args.mask_threshold,
            args.select_threshold,
            args.min_select_pixels,
        )
        scheduler.step()
        selector_score = metrics["val_hit1"] + 0.25 * metrics["val_hit3"]
        joint_score = 0.55 * metrics["val_hit1"] + 0.30 * metrics["val_pos_dice"] + 0.15 * metrics["val_slice_f1"]
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            **metrics,
            "selector_score": selector_score,
            "joint_score": joint_score,
        }
        append_history(history_path, row)
        print("[EPOCH RESULT]", row)
        save_checkpoint(out_dir / "last_model.pt", model, optimizer, epoch, args, metrics)

        improved = False
        if metrics["val_pos_dice"] > best_mask:
            best_mask = metrics["val_pos_dice"]
            save_checkpoint(out_dir / "best_mask_model.pt", model, optimizer, epoch, args, metrics)
            improved = True
            print(f"[BEST MASK] pos_dice={best_mask:.4f}")
        if selector_score > best_selector:
            best_selector = selector_score
            save_checkpoint(out_dir / "best_selector_model.pt", model, optimizer, epoch, args, metrics)
            improved = True
            print(f"[BEST SELECTOR] score={best_selector:.4f}")
        if joint_score > best_joint:
            best_joint = joint_score
            save_checkpoint(out_dir / "best_joint_model.pt", model, optimizer, epoch, args, metrics)
            improved = True
            print(f"[BEST JOINT] score={best_joint:.4f}")

        stale = 0 if improved else stale + 1
        if stale >= args.early_stop_patience:
            print(f"[EARLY STOP] no checkpoint improvement for {stale} epochs")
            break

    print("[DONE]")
    print("best mask pos_dice:", best_mask)
    print("best selector score:", best_selector)
    print("best joint score:", best_joint)


if __name__ == "__main__":
    main()
