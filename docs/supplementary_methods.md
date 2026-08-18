# Supplementary implementation details

This document mirrors the implementation details supporting the manuscript
"Prediction-Only Segmentation-Guided Vision-Language Model for Multicenter
Acute Ischemic Stroke DWI Report Generation." It is a public reproducibility
companion. The journal submission may also include a separately uploaded,
typeset Supplementary Material file.

## S1. DWI reconstruction, data interfaces, and privacy

For the DICOM subset, high-b-value DWI was selected using a diffusion b-value
of at least 800 s/mm2. When the standard DICOM field was unavailable, b-value
elements and high-b-value terms in `SeriesDescription`, `ProtocolName`,
`SequenceName`, and `ImageType` were inspected. ADC or apparent-diffusion
series were excluded. Images were grouped by `SeriesInstanceUID`,
`AcquisitionNumber`, `Rows`, and `Columns`. Series without
`ImageOrientationPatient`, `ImagePositionPatient`, or `PixelSpacing`, or with
fewer than eight unique slice positions, were excluded. Duplicate physical
positions were resolved in favor of the highest-b-value image, and slices were
ordered by physical position. A predefined latter-half fallback was used only
when explicit high-b-value evidence was unavailable.

Reconstructed images and masks were resampled in LPS space to 0.875 x 0.875 x
1.8 mm spacing and a 256 x 256 x 80 grid. Linear interpolation was used for DWI
and nearest-neighbor interpolation for masks. DWI intensity was normalized per
three-dimensional volume: nonfinite values were set to zero, values were
clipped to the 1st and 99th percentiles, and the clipped range was min-max
scaled to [0, 1].

All examinations from the same de-identified patient identifier must remain in
one split. The expected local schemas are documented in
[`data_schema.md`](data_schema.md). Protected data must remain outside the
repository or under ignored directories.

The public release does not contain DICOM files, reconstructed volumes, lesion
masks, radiology reports, patient identifiers, institutional metadata,
clinical-review spreadsheets, or trained weights. Raw clinical data cannot be
redistributed because of patient privacy, institutional data-use restrictions,
and IRB requirements.

## S2. Segmentation and evidence selection

The segmentation model is a two-dimensional MONAI Attention U-Net operating on
single-channel 256 x 256 axial DWI slices. Training uses an equally weighted
Dice and binary cross-entropy objective. Separate validation-selected
checkpoints are retained for evidence-slice ranking and lesion-shape masks.
During inference, every axial slice is scored without access to the
ground-truth mask. The primary ranking score is the maximum predicted pixel
probability, and the highest-scoring slice is selected as Top-1 evidence.

A slice was lesion-positive when the reference mask contained at least five
positive pixels. Segmentation training used all positive slices, negative
context slices within plus or minus two positions of each positive slice, and
random negative slices equal to 0.75 times the number of positive slices.
Duplicate negative selections were retained once. Validation and final
inference evaluated every axial slice.

Training augmentations were sampled independently. Left-right flipping was
applied to the image and mask with probability 0.50. Image-only intensity
scaling `a` in [0.85, 1.15] and offset `b` in [-0.07, 0.07] were applied with
probability 0.40, followed by clipping to [0, 1]. Gaussian noise with standard
deviation sampled from [0.01, 0.035] was applied with probability 0.25. Gamma
transformation with exponent sampled from [0.8, 1.25] was applied with
probability 0.15.

| Parameter | Value |
|---|---|
| Encoder channels | 32, 64, 128, 256, 512 |
| Strides | 2, 2, 2, 2 |
| Dropout | 0.10 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 512 |
| Positive-slice sampling weight | 1.5 |
| Source-balance exponent | 0.5 |
| Maximum epochs | 30 |
| Selection-mask threshold | 0.90 |
| Lesion-shape-mask threshold | 0.40 |
| Ranking strategy | Maximum pixel probability |
| Random seed | 42 |

The selected slice is exported as four complementary image representations:
the original high-b-value DWI, predicted lesion-shape-mask overlay,
lesion-centered crop, and high-confidence selection-mask overlay.

## S3. Qwen2.5-VL LoRA adaptation

Qwen2.5-VL-7B-Instruct receives the four Top-1 image representations and a
fixed instruction prompt. Base-model parameters are frozen. LoRA adapters are
applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. Loss is computed only on assistant target-report tokens.

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-VL-7B-Instruct |
| Epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Gradient accumulation | 8 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Maximum generated tokens | 64 |
| Decoding | Greedy, without sampling |
| Random seed | 42 |

## S4. Anatomical location predictors and soft hints

Three ImageNet-pretrained ResNet-50 multi-label classifiers are fine-tuned on
the lesion crop, lesion-shape overlay, and high-confidence selection overlay.
Their class probabilities are averaged. Positive-class weights are calculated
from training prevalence and clipped to the range 1 to 20.

| Parameter | Value |
|---|---|
| Backbone | ResNet-50 |
| Input size | 224 x 224 |
| Epochs | 20 |
| Learning rate | 3e-5 |
| Weight decay | 1e-4 |
| Precision hint | probability >= 0.55, maximum 4 labels |
| Recall hint | probability >= 0.25, maximum 8 labels |

The fixed 24-label predictor vocabulary is:

`aca_territory`, `basal_ganglia`, `centrum_semiovale`, `cerebellum`,
`cerebral_hemisphere`, `cingulate_gyrus`, `corona_radiata`,
`corpus_callosum`, `cortex`, `frontal_lobe`, `hippocampus`,
`ica_territory`, `insula`, `internal_capsule`, `mca_territory`, `medulla`,
`midbrain`, `occipital_lobe`, `parietal_lobe`, `pca_territory`, `pons`,
`temporal_lobe`, `thalamus`, and `white_matter`.

## S5. Candidate selection and evaluation

The candidate pool contains a LoRA-adapted image-only report, a
precision-oriented soft-hint report, and a recall-oriented soft-hint report.
Optional rule-based, logistic-regression, random-forest, and gradient-boosting
rerankers use report length, number of predicted locations, overlap, extra and
missing locations, Jaccard similarity, candidate source, and candidate/prior
label indicators. Output selection is performed using validation tuple F1 and
then locked before test evaluation.

The report evaluator extracts laterality and anatomical locations using the
published code in [`../src/evaluate_report_metrics.py`](../src/evaluate_report_metrics.py).
It reports laterality exact accuracy, per-case location F1, location exact
match, per-case laterality-location tuple F1, and tuple exact match. Corpus BLEU
is evaluated separately. Confidence intervals and paired model differences use
10,000 patient-cluster bootstrap replicates.

The fixed 25-category evaluation lexicon is distinct from the 24-label hint
vocabulary. Accepted terms are defined in code and include:

| Evaluation category | Accepted surface forms |
|---|---|
| basal_ganglia | basal ganglia; BG |
| corona_radiata | corona radiata |
| centrum_semiovale | centrum semiovale |
| internal_capsule | internal capsule; posterior limb |
| thalamus | thalamus; thalamic |
| hippocampus | hippocampus |
| caudate | caudate |
| frontal_lobe | frontal |
| parietal_lobe | parietal |
| temporal_lobe | temporal |
| occipital_lobe | occipital |
| frontoparietal | frontoparietal; fronto parietal |
| temporoparietal | temporoparietal; temporo parietal |
| precentral_gyrus | precentral |
| postcentral_gyrus | postcentral |
| mca_territory | MCA territory |
| aca_territory | ACA territory |
| pca_territory | PCA territory |
| pons | pons; pontine |
| medulla | medulla; medullary |
| midbrain | midbrain |
| cerebellum | cerebellum; cerebellar |
| vermis | vermis |
| corpus_callosum | corpus callosum |
| cortex | cortex; cortical |

## S6. Reproduction entry points

The end-to-end command sequence is documented in the repository
[`README.md`](../README.md), and the manuscript-to-code correspondence is
listed in [`reproducibility.md`](reproducibility.md). Run the public-release
check, Python compilation, and unit tests before committing or distributing
derived code.
