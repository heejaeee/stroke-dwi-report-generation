import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def split_locs(x):
    if pd.isna(x): return []
    return [s.strip().lower() for s in str(x).split("|") if s.strip()]

def build_label_space(*dfs):
    labs = set()
    for df in dfs:
        col = "parsed_locations_acute" if "parsed_locations_acute" in df.columns else "true_locations"
        for x in df[col].fillna(""):
            labs.update(split_locs(x))
    return sorted(labs)

def make_multihot(x, label_to_idx):
    y = np.zeros(len(label_to_idx), dtype=np.float32)
    for lab in split_locs(x):
        if lab in label_to_idx:
            y[label_to_idx[lab]] = 1.0
    return y

class LocDataset(Dataset):
    def __init__(self, df, image_col, label_to_idx, image_size):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_to_idx = label_to_idx
        self.loc_col = "parsed_locations_acute" if "parsed_locations_acute" in df.columns else "true_locations"
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r[self.image_col]).convert("RGB")
        x = self.tf(img)
        y = torch.from_numpy(make_multihot(r[self.loc_col], self.label_to_idx))
        return x, y, str(r.get("case_id", r.get("patient_id", r.get("id", i)))), str(r[self.loc_col])

class LocModel(nn.Module):
    def __init__(self, model_name, num_locations, pretrained=True, dropout=0.2, freeze_encoder=False):
        super().__init__()
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.encoder = models.resnet50(weights=weights)
            feat_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.encoder = models.efficientnet_b0(weights=weights)
            feat_dim = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
        elif model_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            self.encoder = models.efficientnet_b2(weights=weights)
            feat_dim = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()
        else:
            raise ValueError(model_name)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_locations),
        )

    def forward(self, x):
        return self.head(self.encoder(x))

def loc_metrics(y_true, probs, labels, threshold=0.5):
    pred = probs >= threshold
    yt = y_true.astype(bool)
    inter = (pred & yt).sum()
    pp = pred.sum()
    yy = yt.sum()
    micro_p = inter / max(1, pp)
    micro_r = inter / max(1, yy)
    micro_f1 = 2*micro_p*micro_r/max(1e-8, micro_p+micro_r)
    exact = (pred == yt).all(axis=1).mean()

    f1s = []
    for j in range(yt.shape[1]):
        tp = (pred[:,j] & yt[:,j]).sum()
        fp = (pred[:,j] & ~yt[:,j]).sum()
        fn = (~pred[:,j] & yt[:,j]).sum()
        p = tp/max(1,tp+fp); r = tp/max(1,tp+fn)
        f1s.append(2*p*r/max(1e-8,p+r))
    return {"micro_f1": float(micro_f1), "macro_f1": float(np.mean(f1s)), "exact_match": float(exact)}

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys, ids, trues = [], [], [], []
    for x, y, case_id, true_locs in tqdm(loader, desc="predict"):
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu().numpy()
        probs.append(p); ys.append(y.numpy())
        ids += list(case_id); trues += list(true_locs)
    return {
        "probs": np.concatenate(probs),
        "y": np.concatenate(ys),
        "case_id": ids,
        "true_locations": trues,
    }

def tune_global(y, probs):
    best = (0, 0.5)
    for t in np.arange(0.05, 0.96, 0.05):
        m = loc_metrics(y, probs, [], t)
        score = m["micro_f1"] + 0.25*m["exact_match"]
        if score > best[0]:
            best = (score, float(t))
    return best[1]

def tune_perclass(y, probs):
    th = []
    for j in range(y.shape[1]):
        best = (0, 0.5)
        yt = y[:,j].astype(bool)
        for t in np.arange(0.05, 0.96, 0.05):
            pr = probs[:,j] >= t
            tp = (pr & yt).sum(); fp = (pr & ~yt).sum(); fn = (~pr & yt).sum()
            p = tp/max(1,tp+fp); r = tp/max(1,tp+fn)
            f1 = 2*p*r/max(1e-8,p+r)
            if f1 > best[0]:
                best = (f1, float(t))
        th.append(best[1])
    return np.array(th, dtype=np.float32)

def save_pred(pred, out_csv, labels, global_t, per_t):
    probs = pred["probs"]
    rows = []
    for i in range(len(probs)):
        pg = [labels[j] for j,p in enumerate(probs[i]) if p >= global_t]
        pp = [labels[j] for j,p in enumerate(probs[i]) if p >= per_t[j]]
        if not pg: pg = [labels[int(np.argmax(probs[i]))]]
        if not pp: pp = [labels[int(np.argmax(probs[i]))]]
        row = {
            "case_id": pred["case_id"][i],
            "true_locations": pred["true_locations"][i],
            "pred_locations_global": "|".join(pg),
            "pred_locations_perclass": "|".join(pp),
            "pred_location_top1": labels[int(np.argmax(probs[i]))],
        }
        for j, lab in enumerate(labels):
            row[f"prob_{lab}"] = float(probs[i,j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_col", required=True)
    ap.add_argument("--model_name", default="resnet50")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--freeze_encoder", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_json", default=None)
    args = ap.parse_args()

    seed_all(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = map(pd.read_csv, [args.train_csv, args.val_csv, args.test_csv])
    if args.label_json:
        with open(args.label_json, encoding="utf-8") as f:
            obj = json.load(f)
        labels = obj["location_labels"] if isinstance(obj, dict) else obj
        labels = [str(x).strip() for x in labels if str(x).strip()]
    else:
        labels = build_label_space(train_df, val_df, test_df)
    label_to_idx = {x:i for i,x in enumerate(labels)}
    json.dump({"args": vars(args), "location_labels": labels}, open(out_dir/"config.json","w"), indent=2)

    train_ds = LocDataset(train_df, args.image_col, label_to_idx, args.image_size)
    val_ds = LocDataset(val_df, args.image_col, label_to_idx, args.image_size)
    test_ds = LocDataset(test_df, args.image_col, label_to_idx, args.image_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_train = np.stack([make_multihot(x, label_to_idx) for x in train_df[train_ds.loc_col].fillna("")])
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 20), dtype=torch.float32, device=device)

    model = LocModel(args.model_name, len(labels), bool(args.pretrained), freeze_encoder=bool(args.freeze_encoder)).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score, best_epoch = -1, -1
    for epoch in range(1, args.epochs+1):
        model.train(); losses = []
        for x,y,_,_ in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            x,y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))

        val_pred = predict(model, val_loader, device)
        gt = tune_global(val_pred["y"], val_pred["probs"])
        m = loc_metrics(val_pred["y"], val_pred["probs"], labels, gt)
        score = m["micro_f1"] + 0.25*m["exact_match"]
        print({"epoch": epoch, "loss": np.mean(losses), "global_t": gt, **m})

        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "location_labels": labels, "best_epoch": epoch}, out_dir/"best.pt")
            print("[BEST]", epoch, best_score)

    ckpt = torch.load(out_dir/"best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    val_pred = predict(model, val_loader, device)
    test_pred = predict(model, test_loader, device)
    global_t = tune_global(val_pred["y"], val_pred["probs"])
    per_t = tune_perclass(val_pred["y"], val_pred["probs"])

    save_pred(val_pred, out_dir/"val_predictions.csv", labels, global_t, per_t)
    save_pred(test_pred, out_dir/"test_predictions.csv", labels, global_t, per_t)

    metrics = {
        "best_epoch": best_epoch,
        "global_threshold": global_t,
        "per_class_thresholds": {lab: float(t) for lab,t in zip(labels, per_t)},
        "val_global": loc_metrics(val_pred["y"], val_pred["probs"], labels, global_t),
        "test_global": loc_metrics(test_pred["y"], test_pred["probs"], labels, global_t),
    }
    json.dump(metrics, open(out_dir/"metrics.json","w"), indent=2)
    print("[DONE]", out_dir)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
