import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def load_audit_mod(path):
    spec = importlib.util.spec_from_file_location("audit80", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_named_paths(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected name=path, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Bad name=path item: {item}")
        out[name] = Path(path)
    return out


def metric_row(path, audit):
    df = pd.read_csv(path).fillna("")
    _, pred_col, target_col = audit.cols(df)

    rows = []
    for _, r in df.iterrows():
        pred = str(r[pred_col])
        target = str(r[target_col])
        tl, pl = audit.lat(target), audit.lat(pred)
        tloc, ploc = audit.locs(target), audit.locs(pred)
        tt, pt = audit.tuples(target), audit.tuples(pred)
        rows.append({
            "laterality_exact": int(tl == pl and len(tl) > 0),
            "location_f1": audit.f1(ploc, tloc),
            "location_exact": int(ploc == tloc and len(tloc) > 0),
            "tuple_f1": audit.f1(pt, tt),
            "tuple_exact": int(pt == tt and len(tt) > 0),
        })

    m = pd.DataFrame(rows)
    return {
        "n": len(df),
        "laterality_exact": float(m["laterality_exact"].mean()),
        "location_f1": float(m["location_f1"].mean()),
        "location_exact": float(m["location_exact"].mean()),
        "tuple_f1": float(m["tuple_f1"].mean()),
        "tuple_exact": float(m["tuple_exact"].mean()),
        "unique_predictions": int(df[pred_col].nunique()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audit_script",
        default=str(Path(__file__).with_name("evaluate_report_metrics.py")),
    )
    ap.add_argument("--val_models", nargs="+", required=True, help="Items formatted as name=path")
    ap.add_argument("--test_models", nargs="+", required=True, help="Items formatted as name=path")
    ap.add_argument("--metric", default="tuple_f1")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_csv_name", default="test_candidate_selected_by_val.csv")
    args = ap.parse_args()

    audit = load_audit_mod(args.audit_script)
    val = parse_named_paths(args.val_models)
    test = parse_named_paths(args.test_models)

    missing_test = [name for name in val if name not in test]
    if missing_test:
        raise ValueError(f"Missing matching test model(s): {missing_test}")

    rows = []
    for name, path in val.items():
        row = metric_row(path, audit)
        row["model"] = name
        row["val_path"] = str(path)
        row["test_path"] = str(test[name])
        rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    tie_cols = [args.metric, "tuple_exact", "location_f1", "location_exact", "laterality_exact"]
    tie_cols = [c for c in tie_cols if c in summary.columns]
    summary = summary.sort_values(tie_cols, ascending=[False] * len(tie_cols))

    best = summary.iloc[0].to_dict()
    selected_name = str(best["model"])
    selected_test_path = test[selected_name]

    selected = pd.read_csv(selected_test_path).fillna("")
    selected["selected_model_by_val"] = selected_name
    selected["selected_metric"] = args.metric
    selected["selected_val_metric_value"] = best[args.metric]

    summary_path = out_dir / "val_model_selection_summary.csv"
    out_csv = out_dir / args.out_csv_name
    summary.to_csv(summary_path, index=False)
    selected.to_csv(out_csv, index=False)

    print("[VAL MODEL SELECTION]")
    print(summary.to_string(index=False))
    print("\n[SELECTED]", selected_name, args.metric, best[args.metric])
    print("[SAVED]", summary_path)
    print("[SAVED]", out_csv, "rows:", len(selected))


if __name__ == "__main__":
    main()
