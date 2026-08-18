"""Patient-cluster bootstrap for paired report-model comparisons."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "laterality_exact": "laterality_exact",
    "location_f1": "location_f1",
    "tuple_f1": "tuple_f1",
}


def choose_cluster_column(frame):
    for column in ("patient_id", "id", "case_uid"):
        if column in frame.columns:
            return column
    raise ValueError("Expected patient_id, id, or case_uid for clustered resampling")


def align_frames(final, comparator):
    if "case_id" in final.columns and "case_id" in comparator.columns:
        key = "case_id"
    elif "id" in final.columns and "id" in comparator.columns:
        key = "id"
    else:
        key = "patient_id"
    if key not in final.columns or key not in comparator.columns:
        raise ValueError("Both files must share an id or patient_id column")
    common = final.merge(comparator, on=key, suffixes=("_final", "_comparator"))
    if len(common) != len(final) or len(common) != len(comparator):
        raise ValueError("Paired files do not contain the same examinations")
    return common, key


def bootstrap_metric(frame, cluster_col, final_col, comparator_col, n_boot, rng):
    clusters = frame[cluster_col].astype(str).unique()
    deltas = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        draw = pd.concat(
            [frame[frame[cluster_col].astype(str) == cluster] for cluster in sampled],
            ignore_index=True,
        )
        deltas[index] = draw[final_col].mean() - draw[comparator_col].mean()
    estimate = frame[final_col].mean() - frame[comparator_col].mean()
    low, high = np.quantile(deltas, [0.025, 0.975])
    p_value = min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    return estimate, low, high, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final_csv", required=True)
    parser.add_argument("--comparator_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    final = pd.read_csv(args.final_csv).fillna("")
    comparator = pd.read_csv(args.comparator_csv).fillna("")
    paired, exam_key = align_frames(final, comparator)
    patient_final = "patient_id_final" if "patient_id_final" in paired.columns else exam_key
    cluster_col = patient_final if patient_final in paired.columns else choose_cluster_column(paired)
    rng = np.random.default_rng(args.seed)

    rows = []
    for metric, source_column in METRICS.items():
        final_col = f"{source_column}_final"
        comparator_col = f"{source_column}_comparator"
        if final_col not in paired or comparator_col not in paired:
            continue
        paired[final_col] = pd.to_numeric(paired[final_col])
        paired[comparator_col] = pd.to_numeric(paired[comparator_col])
        estimate, low, high, p_value = bootstrap_metric(
            paired, cluster_col, final_col, comparator_col, args.n_boot, rng
        )
        rows.append(
            {
                "metric": metric,
                "n_examinations": len(paired),
                "n_patients": paired[cluster_col].nunique(),
                "final": paired[final_col].mean(),
                "comparator": paired[comparator_col].mean(),
                "delta": estimate,
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
            }
        )

    output = Path(args.out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"[SAVED] {output}")


if __name__ == "__main__":
    main()
