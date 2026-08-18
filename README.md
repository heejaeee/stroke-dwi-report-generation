# Prediction-Only Segmentation-Guided DWI Report Generation

Reference implementation for a multicenter acute ischemic stroke DWI pipeline that:

1. scores every axial DWI slice with a 2D Attention U-Net;
2. selects a Top-1 evidence slice using predicted maximum pixel probability only;
3. constructs four complementary image views from predicted masks;
4. adapts Qwen2.5-VL with LoRA for concise report generation; and
5. optionally adds a three-view anatomical location ensemble and validation-locked output selection.

Ground-truth masks are used for segmentation training and evaluation. They are not used to rank or select evidence slices during inference.

## Repository status

This release contains research code and synthetic schemas. It does **not** contain patient images, lesion masks, radiology reports, clinical-review files, institutional identifiers, or trained weights. Users must obtain local ethics approval and data-use permission before applying the code to clinical data.

The code accompanies the manuscript **"Prediction-Only Segmentation-Guided Vision-Language Model for Multicenter Acute Ischemic Stroke DWI Report Generation."** Detailed parameter tables, label spaces, and the manuscript-to-code map are available in [`docs/supplementary_methods.md`](docs/supplementary_methods.md) and [`docs/reproducibility.md`](docs/reproducibility.md).

## Installation

Python 3.10 and a CUDA-capable PyTorch environment were used for the study.

```bash
git clone https://github.com/heejaeee/stroke-dwi-report-generation.git
cd stroke-dwi-report-generation
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Qwen2.5-VL also requires access to `Qwen/Qwen2.5-VL-7B-Instruct` under its upstream license.

## Private data layout

Keep protected data outside the repository. Start from [`examples/case_manifest.example.csv`](examples/case_manifest.example.csv) and see [`docs/data_schema.md`](docs/data_schema.md). To avoid exposing local or institutional directory structures, every filesystem argument in the commands below is represented by the placeholder `path`.

Patient-level splitting must be completed before training. All examinations sharing a de-identified `patient_id` must remain in one partition.

## Core workflow

### 1. Train the Attention U-Net

The cache files must contain `images`, `masks`, `patient_id`, `source`, and `is_positive` arrays.

```bash
python src/train_segmentation.py \
  --train_cache path \
  --val_cache path \
  --out_dir path \
  --epochs 30 --batch_size 512 --lr 1e-3 \
  --weight_decay 1e-5 --positive_weight 1.5 \
  --source_balance_power 0.5 --amp --seed 42
```

This writes separate selector and mask checkpoints chosen on validation data.

### 2. Score all slices and lock the ranking on validation data

```bash
python src/prediction_only_utils.py \
  --case_manifest path \
  --selector_ckpt path \
  --mask_ckpt path \
  --out_dir path \
  --prefix prediction_only --top_k 5 \
  --rank_strategy max_prob --th_select 0.90 --th_mask 0.40

python src/select_slice_ranking.py \
  --all_scores_csv path \
  --out_dir path --selection_split val
```

The study selected `max_prob` on validation data. Test Hit@k must not be used to change this choice.

### 3. Export the Top-1 four-view input

```bash
python src/export_prediction_only_top1.py \
  --all_scores_csv path \
  --reuse_manifest path \
  --selector_ckpt path \
  --mask_ckpt path \
  --out_dir path --prefix prediction_only_top1 \
  --rank_strategy max_prob --th_select 0.90 --th_mask 0.40
```

The four views are the original DWI, predicted shape-mask overlay, lesion-centered crop, and high-confidence selection-mask overlay.

### 4. Build records and adapt Qwen2.5-VL

```bash
python src/build_qwen_dataset.py \
  --source_dir path \
  --prediction_manifest path \
  --out_dir path --top_k 1 \
  --image_types raw,overlay,crop,select_overlay

python src/train_qwen_lora.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl path \
  --val_jsonl path \
  --out_dir path --epochs 5 --lr 2e-4 \
  --weight_decay 0.01 --grad_accum 8 --lora_r 16 \
  --lora_alpha 32 --lora_dropout 0.05 --seed 42

python src/infer_qwen_lora.py \
  --base_model Qwen/Qwen2.5-VL-7B-Instruct \
  --lora_dir path \
  --input_jsonl path \
  --output_jsonl path \
  --max_new_tokens 64
```

### 5. Add anatomical soft hints

Build location-model tables with `src/build_location_inputs.py`; train one ResNet-50 for each of `crop_path`, `overlay_path`, and `select_overlay_path`; and average probabilities with `src/ensemble_location_predictors.py`. Use `src/make_soft_hint_candidates.py` and `src/infer_qwen_lora.py` to produce no-hint, precision-hint, and recall-hint candidates.

Lock one output branch using validation tuple F1 and apply the corresponding test predictions unchanged:

```bash
python src/select_validation_branch.py \
  --val_models no_hint=path precision_hint=path recall_hint=path \
  --test_models no_hint=path precision_hint=path recall_hint=path \
  --metric tuple_f1 --out_dir path
```

Optional rerankers are implemented in `src/candidate_scoring.py` and `src/train_candidate_rerankers.py`. An oracle candidate must never be reported as the primary test result.

### 6. Evaluate and bootstrap

```bash
python src/evaluate_report_metrics.py \
  --csv path path \
  --out_dir path

python src/evaluate_bleu.py \
  --input_csv path \
  --prediction_col prediction --reference_col target \
  --output_txt path

python src/bootstrap_compare.py \
  --final_csv path \
  --comparator_csv path \
  --out_csv path \
  --n_boot 10000 --seed 42
```

## Public-release check

Run before every commit or push:

```bash
python scripts/check_public_release.py
python -m compileall -q src scripts
python -m unittest discover -s tests -v
```

The complete first-push checklist is in [`docs/release_checklist.md`](docs/release_checklist.md).

## Reproducibility and reporting

- Configuration values are summarized in [`configs/paper.yaml`](configs/paper.yaml).
- Supplementary implementation details are mirrored in [`docs/supplementary_methods.md`](docs/supplementary_methods.md).
- Manuscript-to-code mapping is in [`docs/reproducibility.md`](docs/reproducibility.md).
- Strict examination-level metrics must remain separate from expert-adjudicated slice-conditioned sensitivity analyses.
- The clinical dataset is not publicly redistributable. Never open an issue containing patient data or free-text reports.

## Data and code availability

Source code and configuration files for segmentation training, prediction-only evidence selection, four-view input generation, Qwen2.5-VL LoRA adaptation, anatomical soft-hint generation, validation-locked output selection, and evaluation are publicly available in this repository.

The multicenter DWI examinations, lesion masks, radiology reports, patient identifiers, and clinical-review files cannot be publicly distributed because of patient privacy, institutional data-use restrictions, and IRB requirements. Trained checkpoints are not included in this release and remain subject to institutional approval and upstream model licenses.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Code is released under Apache-2.0. Models and datasets remain subject to their original licenses, ethics approvals, and institutional data-use agreements.
