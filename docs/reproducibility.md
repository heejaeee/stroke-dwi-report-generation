# Reproducibility map

This document maps the manuscript workflow to the public code. Additional implementation details and parameter tables are available in [`supplementary_methods.md`](supplementary_methods.md). Protected source data and clinical-review records are intentionally excluded from the repository.

| Manuscript component | Public entry point |
|---|---|
| Attention U-Net training | `src/train_segmentation.py` |
| Prediction-only full-series scoring | `src/prediction_only_utils.py` |
| Validation-only ranking selection | `src/select_slice_ranking.py` |
| Top-1 four-view export | `src/export_prediction_only_top1.py` |
| Qwen dataset construction | `src/build_qwen_dataset.py` |
| Qwen2.5-VL LoRA adaptation | `src/train_qwen_lora.py` |
| Location classifiers | `src/train_location_predictor.py` |
| Three-view probability ensemble | `src/ensemble_location_predictors.py` |
| Soft-hint candidate construction | `src/make_soft_hint_candidates.py` |
| Candidate feature scoring and optional reranking | `src/candidate_scoring.py`, `src/train_candidate_rerankers.py` |
| Validation-locked branch selection | `src/select_validation_branch.py` |
| Clinical-content and BLEU evaluation | `src/evaluate_report_metrics.py`, `src/evaluate_bleu.py` |
| Patient-cluster bootstrap | `src/bootstrap_compare.py` |

The final branch must be selected on validation data and then applied unchanged to the test set. A test-best or oracle candidate is diagnostic only and must not be reported as the primary result.

## Public and protected components

| Component | Release status |
|---|---|
| Training, inference, candidate-selection, and evaluation code | Public |
| Synthetic data schemas and example records | Public |
| Study configuration and location-label space | Public |
| Multicenter DWI examinations and DICOM metadata | Not public |
| Lesion masks and radiology reports | Not public |
| Patient identifiers and expert-review records | Not public |
| Trained checkpoints | Not included; release requires institutional approval |
