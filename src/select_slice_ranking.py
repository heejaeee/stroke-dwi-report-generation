import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STRATEGIES = {
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


def rank_case(case, strategy):
    columns, ascending = STRATEGIES[strategy]
    return case.sort_values(columns, ascending=ascending, kind="mergesort")


def evaluate(frame, strategy):
    rows = []
    for case_uid, case in frame.groupby("case_uid", sort=False):
        ranked = rank_case(case, strategy)
        positive = np.flatnonzero(ranked["is_positive"].to_numpy(dtype=int) == 1) + 1
        first = int(positive[0]) if len(positive) else None
        rows.append(
            {
                "case_uid": case_uid,
                "first_positive_rank": first,
                "hit_at_1": int(first is not None and first <= 1),
                "hit_at_3": int(first is not None and first <= 3),
                "hit_at_5": int(first is not None and first <= 5),
            }
        )
    cases = pd.DataFrame(rows)
    return cases, {
        "strategy": strategy,
        "cases": int(len(cases)),
        "hit_at_1": float(cases["hit_at_1"].mean()),
        "hit_at_3": float(cases["hit_at_3"].mean()),
        "hit_at_5": float(cases["hit_at_5"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_scores_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--selection_split", default="val")
    parser.add_argument(
        "--strategies",
        default="area_first,max_prob,top100_mean",
        help="Comma-separated ranking strategies evaluated on the selection split.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.all_scores_csv)
    required = {
        "case_uid",
        "split",
        "slice_idx",
        "is_positive",
        "select_area",
        "select_max_prob",
        "select_mean_top100_prob",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    selection = frame[frame["split"].astype(str) == args.selection_split].copy()
    if selection.empty:
        raise ValueError(f"No rows for selection split: {args.selection_split}")

    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    unknown = sorted(set(strategies) - set(STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for strategy in strategies:
        cases, summary = evaluate(selection, strategy)
        summaries.append(summary)
        cases.to_csv(out_dir / f"{strategy}_{args.selection_split}_cases.csv", index=False)

    comparison = pd.DataFrame(summaries).sort_values(
        ["hit_at_1", "hit_at_3", "hit_at_5", "strategy"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    selected = comparison.iloc[0].to_dict()
    selected.update(
        {
            "selection_split": args.selection_split,
            "selection_metric_order": ["hit_at_1", "hit_at_3", "hit_at_5"],
            "test_labels_used_for_selection": False,
            "all_scores_csv": args.all_scores_csv,
        }
    )

    comparison.to_csv(out_dir / "validation_ranking_comparison.csv", index=False)
    (out_dir / "selected_ranking.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    (out_dir / "selected_ranking.env").write_text(
        f"SELECTED_RANK_STRATEGY={selected['strategy']}\n", encoding="utf-8"
    )

    print("[VALIDATION RANKING COMPARISON]")
    print(comparison.to_string(index=False))
    print("\n[SELECTED BY VALIDATION]")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
