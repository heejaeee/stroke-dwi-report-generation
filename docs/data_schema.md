# Data schema

The repository does not distribute patient images, lesion annotations, reports, or identifiers. Prepare de-identified local files with the following schemas.

## Case manifest

Required columns:

| Column | Description |
|---|---|
| `case_uid` | Unique examination identifier with no direct patient identifier |
| `source` | Institution or acquisition-domain code |
| `patient_id` | De-identified grouping key used to prevent patient leakage |
| `split` | `train`, `val`, or `test`; all examinations from one patient must share a split |
| `image_path` | Local path to a reconstructed three-dimensional DWI volume |
| `mask_path` | Local path to the lesion mask, used for training and evaluation only |
| `axis` | Axial slice axis, normally `2` |

## Prediction manifest

`export_prediction_only_top1.py` writes one selected row per examination with the original DWI, predicted-mask overlay, lesion-centered crop, and high-confidence overlay paths. Ground-truth masks are loaded only after prediction-based ranking for Hit@k and quality-control calculations.

## Qwen JSONL

Each record contains a de-identified examination ID, source, split, target report, four selected image paths, and chat-formatted messages. The four images must come from the same prediction-only Top-1 slice.

## Protected information

Do not commit DICOM headers, accession numbers, medical-record numbers, dates, free-text reports, clinical review spreadsheets, absolute institutional paths, or output logs. Use `scripts/check_public_release.py` before every push.
