# Chronic Wound Image Classification - Gated Multi-Model Fusion

A deep learning pipeline for automated classification of chronic wound images into six clinically relevant categories, using a gated fusion ensemble of three convolutional neural network backbones (VGG19, DenseNet201, MobileNetV2) combined via a learned gating network trained on out-of-fold (OOF) predictions.

---

## Table of Contents

- [Chronic Wound Image Classification - Gated Multi-Model Fusion](#chronic-wound-image-classification---gated-multi-model-fusion)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [Dataset](#dataset)
  - [Architecture](#architecture)
  - [Data Integrity \& Methodology Note](#data-integrity--methodology-note)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Leakage Audit (recommended before training)](#2-leakage-audit-recommended-before-training)
    - [3. Generate Out-of-Fold Predictions](#3-generate-out-of-fold-predictions)
    - [4. Train the Gating Network](#4-train-the-gating-network)
    - [5. Final Evaluation](#5-final-evaluation)
  - [Results](#results)
    - [Overall Metrics](#overall-metrics)
    - [Per-Class Accuracy](#per-class-accuracy)
    - [Confusion Matrix](#confusion-matrix)
    - [Observations](#observations)
  - [Reproducibility](#reproducibility)
  - [Roadmap](#roadmap)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

---

## Overview

Chronic wounds (diabetic, pressure, surgical, venous, and related presentations) require accurate visual classification to support clinical triage and monitoring. This project implements a **gated fusion architecture** that combines predictions from three independently trained CNN backbones, using a lightweight gating network to learn how much to trust each backbone's prediction on a per-sample basis, rather than relying on fixed-weight averaging.

The pipeline is designed around **rigorous, leakage-free evaluation**: every backbone is trained and validated using group-aware stratified cross-validation, ensuring that augmented variants of a single source image can never appear in both the training and validation partitions of any fold. The gated fusion model has been trained end-to-end on this clean pipeline and evaluated on a held-out, untouched test set - see [Results](#results).

**Target classes:**

| Class         | Description                        |
| ------------- | ---------------------------------- |
| `background`  | Non-wound / background skin images |
| `diabetic`    | Diabetic foot ulcers               |
| `normal-skin` | Healthy, unaffected skin           |
| `pressure`    | Pressure (bed) ulcers              |
| `surgical`    | Surgical wounds                    |
| `venous`      | Venous leg ulcers                  |

---

## Key Features

- **Three-backbone ensemble** - VGG19, DenseNet201, and MobileNetV2, each fine-tuned via staged transfer learning (frozen-backbone Phase 1, optional fine-tuning Phase 2).
- **Gated fusion** - a trainable MLP gate (`src/models/gating.py`) learns per-sample confidence weighting across backbone outputs, rather than static averaging or majority voting.
- **Group-aware K-fold cross-validation** - `StratifiedGroupKFold` ensures that an original image and all of its augmented derivatives are confined to a single fold, preventing near-duplicate leakage across train/validation splits.
- **Out-of-fold (OOF) prediction generation** - the gating network is trained exclusively on OOF probabilities, so it never sees predictions produced by a model on data it was trained on.
- **Dedicated leakage-audit tooling** - `leakage_audit.py` and `scripts/check_clean_split.py` verify split integrity before any reported metrics are generated.
- **Deterministic, reproducible runs** - fixed random seeds, sorted file iteration, and explicit train/validation transform separation.
- **End-to-end validated on a clean, held-out test set** - final metrics (accuracy, macro/weighted F1, per-class breakdown, confusion matrix) are computed strictly on images the fused model has never seen in any form.

---

## Dataset

The pipeline is built on the **AZH (Advancing the Zenith of Healthcare) chronic wound dataset**, vendored under `data/raw/azh/wound_classification-main/`. Training data is augmented offline (horizontal/vertical flips, rotation) prior to model training; validation and test partitions are held strictly free of augmentation.

Augmented filenames follow the convention `aug_{index}_{original_stem}.{ext}` (e.g. `aug_0000_10.jpg` is an augmented derivative of source image `10.jpg`), which is used downstream to recover group membership for leakage-safe splitting.

Class-to-label mapping (verified against the processed split, alphabetical order):

```
0 = background
1 = diabetic
2 = normal-skin
3 = pressure
4 = surgical
5 = venous
```

> Raw dataset assets and generated processed splits are excluded from version control (see `.gitignore`). Only the upstream `wound_classification-main` reference repo and label metadata (`data/npy/wound_label.npy`) are vendored under `data/raw/azh/`.

---

## Architecture

```
                 ┌─────────────┐
   Input Image → │   VGG19     │ → P₁ (probabilities)
                 └─────────────┘
                 ┌─────────────┐
   Input Image → │ DenseNet201 │ → P₂ (probabilities)          ┌────────────┐      Final
                 └─────────────┘                        ────→  │ Gating MLP │  →  Prediction
                 ┌─────────────┐                               └────────────┘
   Input Image → │ MobileNetV2 │ → P₃ (probabilities)
                 └─────────────┘
```

Backbone architectures are defined in `src/models/backbones.py`; the gating network is defined in `src/models/gating.py`. Each backbone is trained independently via 5-fold group-aware stratified cross-validation (`scripts/gen_oof_preds.py`), producing out-of-fold probability vectors for every training sample. The concatenated OOF probability vectors `[P₁ | P₂ | P₃]` form the input feature space for the gating MLP (`scripts/fusion/train_gating_mlp.py`), which learns to weight each backbone's contribution per sample before producing the final fused prediction.

**Training regimen per backbone, per fold:**

| Phase                | Description                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------- |
| Phase 1              | Backbone frozen; only the classification head is trained. Early stopping on validation loss.       |
| Phase 2 _(optional)_ | Last convolutional block unfrozen; fine-tuned at `lr / 10`. Disabled by default (`--phase2` flag). |

---

## Data Integrity & Methodology Note

An early version of this pipeline exhibited data leakage defects that were identified, diagnosed, and remediated prior to producing any reportable results. Two separate instances were found and fixed:

**1. Original pipeline (base-model training/evaluation):**

- A fallback code path could copy a training image into the test directory when a corresponding test file was missing, allowing identical or near-identical images to appear in both partitions.
- The original cross-validation logic (`StratifiedKFold`) balanced folds by class label only, with no awareness that multiple augmented derivatives of a single source image existed in the training set - allowing an original image and its augmented siblings to be split across the train and validation portions of the same fold.

**2. Gating-fusion script (discovered during a later audit):**

- The initial `train_gating.py` / `train_gating_mlp.py` implementations reintroduced the same fallback-copy pattern independently, while constructing test-set paths by string-replacing `/train/` with `/test/` in OOF metadata. Since augmented filenames (e.g. `aug_0000_10.jpg`) never legitimately exist under `data/processed/test/`, this fallback would copy the training file into the test directory - recreating leakage in a different part of the codebase.

**Remediation applied:**

- `scripts/rebuild_clean_split.py` reconstructs train/val/test partitions directly from the raw AZH source, with augmentation applied only to the training partition.
- `scripts/gen_oof_preds.py` assigns folds using `StratifiedGroupKFold`, with group membership derived from each file's original source stem (augmentation prefixes stripped via regex), guaranteeing that all variants of a source image reside in a single fold.
- Training and validation subsets use strictly separate transform pipelines - augmentation (flip/rotation) is applied only to the training subset; validation always uses deterministic resize + normalize.
- `scripts/fusion/train_gating_mlp.py` was rewritten to build test rows by directly scanning `data/processed/test/` - never reconstructing or mirroring paths from training metadata, and never copying files across splits. A missing test file now raises an explicit error instead of being silently backfilled.
- `leakage_audit.py` and `scripts/check_clean_split.py` provide standalone verification that no filename/group overlaps exist across the final train/val/test partitions.
- All outputs generated under the leaked configuration have been archived (not deleted) under `archive/cleanup_2026_09_12/` and are excluded from any final reported metrics.

This project only reports metrics generated under the corrected, group-aware, leakage-audited pipeline. Earlier leaked-run artifacts are retained solely for internal before/after comparison and audit purposes.

---

## Project Structure

```
chronic-wound-fusion/
├── README.md                                  # Project overview, setup, and final usage
├── leakage_audit.py                           # Standalone leakage/overlap verification
├── requirements.txt                           # Python dependencies for training and evaluation
├── configs/                                   # Project config and experiment defaults
│   └── default.yaml                           # Central experiment configuration
├── archive/                                   # Archived legacy or leaked artifacts for audit history
│   └── cleanup_2026_09_12/                    # Archived pre-cleanup runs and leakage-prone outputs
├── data/                                      # Dataset storage root
│   └── raw/                                   # Raw source datasets, kept local and not versioned
│       └── azh/                               # AZH wound dataset reference copy
│           └── wound_classification-main/     # Vendored upstream AZH dataset and labels
├── docs/                                      # Documentation and draft writing material
│   ├── project-plan.docx                      # Initial project plan / FYP planning doc
│   └── REPORT.md                              # Working project write-up and report notes
├── notebooks/                                 # Jupyter exploration and validation notebooks
│   ├── 01_setup_check.ipynb                   # Environment and package sanity check
│   └── gated_fusion_prototype.ipynb           # Prototype notebook kept only for historical context
├── outputs/                                   # Generated artifacts for model runs and final reports
│   ├── 01_oof/                                # Per-model OOF probabilities and metadata
│   │   ├── vgg19_clean/vgg19_oof_probs.npy    # VGG19 OOF probabilities array
│   │   ├── densenet201_clean/densenet201_oof_probs.npy # DenseNet201 OOF probabilities array
│   │   └── mobilenetv2_clean/mobilenet_v2_oof_probs.npy # MobileNetV2 OOF probabilities array
│   ├── 02_gating/                             # Gating network training and fused test outputs
│   │   ├── gating_mlp_model_v1.pt             # Trained gating network weights
│   │   ├── gating_test_probs_v1.npy           # Fused test-set probability matrix
│   │   └── gating_test_preds_v1.csv           # Final clean-test predictions
│   └── 03_figures/                            # Final report figures and short outputs
│       ├── gated_fusion_evaluation_report.txt # Final metrics report in text form
│       └── gated_fusion_confusion_matrix.png # Confusion matrix for final evaluation
├── scripts/                                   # Core training, data, and evaluation scripts
│   ├── gen_oof_preds.py                       # Group-aware K-fold OOF generation for each backbone
│   ├── rebuild_clean_split.py                 # Rebuilds clean train/val/test splits from raw data
│   ├── check_clean_split.py                   # Verifies no overlap across partitions
│   ├── augment_and_save.py                    # Legacy augmentation helper
│   ├── augment_train_only.py                  # Applies augmentation only to training subset
│   ├── dataset_stats.py                       # Dataset summary and distribution checks
│   ├── evaluate_checkpoint.py                 # Single-backbone evaluation helper
│   ├── evaluate_gated_fusion.py               # Final fused-model evaluation script
│   ├── evaluate_vgg.py                        # Legacy VGG evaluation helper
│   ├── train_base.py                          # Early base training experiments
│   ├── train_backbone_small.py                # Small-scale backbone training prototype
│   ├── train_vgg_small.py                     # Legacy VGG small-model experiments
│   ├── train_gating.py                        # OOF-only gating trainer, not final test evaluator
│   ├── parse_docx.py                          # Document parsing utility
│   ├── verify_data.py                         # Local data integrity checks
│   ├── augment/augment_offline.py             # Older offline augmentation implementation
│   ├── fusion/train_gating_mlp.py             # Canonical gating trainer + clean test evaluation
│   ├── organize/archive_unused_artifacts.py   # Cleanup utility for old artifacts
│   ├── organize/organize_outputs.py           # Output organization utility
│   └── train/train_vgg19_staged.py            # Earlier staged-training prototype
└── src/                                       # Main source package for data and model code
    ├── data/                                  # Data loading and preprocessing utilities
    │   ├── dataset.py                          # Dataset wrapper and sample loading
    │   └── preprocess.py                       # Image preprocessing / normalization helpers
    └── models/                                 # Model definitions and architecture code
        ├── backbones.py                       # VGG19 / DenseNet201 / MobileNetV2 definitions
        └── gating.py                          # Learned fusion gate network definition
```

> `data/processed/` (generated train/val/test splits) and model checkpoints are generated locally and excluded from version control - see `.gitignore`.

---

## Installation

**Prerequisites:** Python 3.11, a CUDA-capable GPU (recommended), Git.

```powershell
git clone https://github.com/AbdurRehmanKhan-ARK/chronic-wound-fusion.git
cd chronic-wound-fusion

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

**Core dependencies:** `torch`, `torchvision`, `scikit-learn>=1.1` (required for `StratifiedGroupKFold`), `numpy`, `pandas`, `Pillow`, `matplotlib`.

Verify scikit-learn version:

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

---

## Usage

### 1. Data Preparation

Rebuild clean train/val/test partitions directly from the raw AZH source (augmentation applied to `train/` only):

```powershell
.venv\Scripts\python.exe scripts\rebuild_clean_split.py --source data\raw\azh --output data\processed
.venv\Scripts\python.exe scripts\augment_train_only.py --train-dir data\processed\train
```

### 2. Leakage Audit (recommended before training)

Verify there is no filename/group overlap across partitions before spending compute on training:

```powershell
.venv\Scripts\python.exe leakage_audit.py --data-dir data\processed
.venv\Scripts\python.exe scripts\check_clean_split.py --data-dir data\processed
```

### 3. Generate Out-of-Fold Predictions

Run each backbone independently:

```powershell
.venv\Scripts\python.exe scripts\gen_oof_preds.py --model vgg19 --folds 5 --phase1-epochs 8 --batch-size 16 --lr 1e-4 --weight-decay 1e-4 --output outputs\01_oof\vgg19_clean

.venv\Scripts\python.exe scripts\gen_oof_preds.py --model densenet201 --folds 5 --phase1-epochs 8 --batch-size 16 --lr 1e-4 --weight-decay 1e-4 --output outputs\01_oof\densenet201_clean

.venv\Scripts\python.exe scripts\gen_oof_preds.py --model mobilenet_v2 --folds 5 --phase1-epochs 8 --batch-size 16 --lr 1e-4 --weight-decay 1e-4 --output outputs\01_oof\mobilenetv2_clean
```

Each run produces:

- `{model}_oof_probs.npy` - out-of-fold probability matrix, shape `(N, num_classes)`
- `{model}_oof_meta.csv` - index-to-filepath-to-label mapping
- `checkpoints/{model}_fold{k}_best.pt` - best checkpoint per fold

### 4. Train the Gating Network

The canonical trainer (`scripts/fusion/train_gating_mlp.py`) trains the gate on concatenated OOF probabilities **and** evaluates it on the clean test set in one run:

```powershell
.venv\Scripts\python.exe scripts\fusion\train_gating_mlp.py --models vgg19,densenet201,mobilenet_v2 --oof-dir outputs\01_oof --output outputs\02_gating
```

This produces:

- `gating_mlp_model_v1.pt` - trained gating network weights
- `gating_test_probs_v1.npy` - fused probability matrix on the clean test set
- `gating_test_preds_v1.csv` - `index, file_path, true_label, pred_label` for every clean test sample

> `scripts/train_gating.py` is a lighter, OOF-only variant retained for validation/debugging; it does not perform test-set evaluation.

### 5. Final Evaluation

Compute final metrics from the fused test predictions:

```powershell
.venv\Scripts\python.exe scripts\evaluate_gated_fusion.py --preds-csv outputs\02_gating\gating_test_preds_v1.csv --output outputs\03_figures
```

Outputs: overall accuracy, macro F1, weighted F1, per-class accuracy, confusion matrix (printed + saved as `.png`), and a full `sklearn` classification report (saved as `.txt`).

---

## Results

Final evaluation was performed on **111 held-out test images** that were never seen - in original or augmented form - by any base model or the gating network at any stage of training.

### Overall Metrics

| Metric               | Value      |
| -------------------- | ---------- |
| **Overall Accuracy** | **73.87%** |
| **Macro F1**         | **74.49%** |
| **Weighted F1**      | **74.33%** |

### Per-Class Accuracy

| Class       | Precision | Recall | F1-score | Support |
| ----------- | --------- | ------ | -------- | ------- |
| background  | 1.0000    | 0.9333 | 0.9655   | 15      |
| diabetic    | 0.7619    | 0.6957 | 0.7273   | 23      |
| normal-skin | 0.9333    | 0.9333 | 0.9333   | 15      |
| pressure    | 0.3684    | 0.4667 | 0.4118   | 15      |
| surgical    | 0.7333    | 0.5789 | 0.6471   | 19      |
| venous      | 0.7407    | 0.8333 | 0.7843   | 24      |

### Confusion Matrix

Rows = true label, columns = predicted label (order: background, diabetic, normal-skin, pressure, surgical, venous):

```
[[14  0  1  0  0  0]
 [ 0 16  0  3  0  4]
 [ 0  1 14  0  0  0]
 [ 0  2  0  7  4  2]
 [ 0  2  0  5 11  1]
 [ 0  0  0  4  0 20]]
```

### Observations

- `background` and `normal-skin` are classified near-perfectly (93.3% accuracy each) - these classes are visually distinct from wound presentations.
- `pressure` (46.7%) and `surgical` (57.9%) show the weakest performance and are frequently confused with each other (4 pressure→surgical, 5 surgical→pressure misclassifications), consistent with the clinically observed visual similarity between healing pressure ulcers and surgical wounds.
- `diabetic` samples are occasionally confused with `venous` (4 cases), which is plausible given both are lower-limb chronic ulcer presentations with overlapping visual characteristics.
- These confusion patterns are clinically interpretable rather than random, suggesting the model has learned meaningful wound-type features rather than dataset artifacts - consistent with the pipeline now being leakage-free.

These figures reflect genuinely unseen-data performance and are markedly lower than the pre-remediation (leaked) run, which is the expected and correct direction of change once train/test and cross-fold leakage are eliminated. See [Data Integrity & Methodology Note](#data-integrity--methodology-note) for full details of what was fixed.

---

## Reproducibility

All training runs use a fixed seed (`--seed 42` by default), deterministic sorted file iteration, and explicit train/validation transform separation. To reproduce a specific backbone's OOF run, use the same `--model`, `--folds`, and `--seed` values recorded in that run's checkpoint directory. Central experiment settings can be tracked in `configs/default.yaml`.

---

## Roadmap

- [x] Identify and remediate train/test path-overlap leakage
- [x] Identify and remediate source-unaware fold-splitting leakage
- [x] Implement `StratifiedGroupKFold` with source-stem grouping
- [x] Regenerate clean OOF predictions for VGG19, DenseNet201, MobileNetV2
- [x] Identify and remediate a second, independent leakage instance in the gating-fusion script
- [x] Retrain gating MLP on clean, concatenated OOF probabilities
- [x] Run final evaluation on the clean, held-out test set
- [x] Publish final confusion matrix and per-class metrics
- [ ] Document leaked-vs-clean metric comparison as an appendix in `docs/REPORT.md`
- [ ] Optional: enable Phase-2 fine-tuning per backbone and re-evaluate for potential accuracy gains, particularly on `pressure` and `surgical` classes

---

## Contributing

This is an academic Final Year Project (FYP). External contributions are not currently being accepted, but issues and suggestions are welcome via the repository's Issues tab.

---

## License

Specify a license (e.g. MIT, Apache 2.0) appropriate to your institution's FYP submission policy. No license is currently declared.

---

## Acknowledgments

- AZH chronic wound dataset (`wound_classification-main`)
- PyTorch / torchvision pretrained ImageNet backbones (VGG19, DenseNet201, MobileNetV2)
- scikit-learn `StratifiedGroupKFold` for leakage-safe cross-validation
