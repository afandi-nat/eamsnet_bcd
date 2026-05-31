# EAMSNet: An Efficient Edge-Aware Multi-Scale Attention Network for Building Change Detection in Remote Sensing Imagery

A PyTorch implementation of **EAMSNet**, a binary change-detection network for
remote-sensing imagery built on the lightweight **MobileNetV2** backbone with three
proposed modules:

- **ATDAM** (Attentive Temporal Difference Attention Module) : builds the T1/T2 difference feature using channel + spatial attention and gating.
- **MSDA** (Multi-Scale Dilated Aggregation) : ASPP-like multi-scale context aggregation on the deepest difference feature.
- **EABRM** (Edge-Aware Boundary Refinement Module) : sharpens change-object boundaries using Sobel operators.

This repository provides a full training pipeline, evaluation (threshold-optimized,
flip-TTA, multi-scale TTA, post-processing), a **resumable ablation study**, and
qualitative visualizations.

## Project structure

```
eamsnet-cd/
├── eamsnet/                  # core package
│   ├── models/               # encoder, modules (ATDAM/MSDA/EABRM), EAMSNet
│   ├── data/                 # LEVIR-CD dataset + dataloaders
│   ├── losses/               # HybridLoss (Focal+Tversky+EdgeAware+Lovasz)
│   ├── engine/               # train/eval, TTA, visualization
│   └── utils/                # metrics, EMA, scheduler, optimizer, benchmark
├── scripts/                  # CLI entry points
│   ├── train.py              # train a single model (resumable)
│   ├── evaluate.py           # test evaluation + visualization
│   ├── train_ablation.py     # train 6 ablation configurations (resumable)
│   └── visualize_ablation.py # heatmap / sample-per-config
├── configs/default.yaml      # default hyperparameters
├── requirements.txt
└── setup.py
```

## Installation

```bash
git clone https://github.com/<user>/eamsnet-cd.git
cd eamsnet-cd
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

For the optional `convnext_tiny` backbone, also run `pip install timm`.

## Dataset preparation

Arrange **LEVIR-CD** (the 256×256 variant) as follows:

```
LEVIR-CD-256/
├── train/   A/  B/  label/
├── val/     A/  B/  label/
└── test/    A/  B/  label/
```

`A` = T1 image, `B` = T2 image, `label` = binary mask (white = changed). The file
names in `A`, `B`, and `label` must match for each pair.

## Usage

All scripts read `configs/default.yaml`; any value can be overridden on the command
line (e.g. `--data-root`, `--epochs`, `--batch-size`).

### Train the full model

```bash
python scripts/train.py --data-root /path/LEVIR-CD-256 --epochs 250 --out-dir outputs
```

Training saves `outputs/best_eamsnet.pth` (best weights) and
`outputs/last_eamsnet.pth` (full state). To continue an interrupted run:

```bash
python scripts/train.py --data-root /path/LEVIR-CD-256 --resume
```

### Evaluation

```bash
python scripts/evaluate.py --ckpt outputs/best_eamsnet.pth --tta --msf --visualize
```

Reports standard, threshold-optimized, flip-TTA, multi-scale TTA, and
post-processing metrics, plus complexity (params/FPS), and saves a qualitative grid.

### Ablation study

Trains six configurations under an identical pipeline:

| # | Configuration |
|---|---|
| 1 | Baseline |
| 2 | Baseline + ATDAM |
| 3 | Baseline + MSDA |
| 4 | Baseline + EABRM |
| 5 | Baseline + ATDAM + MSDA |
| 6 | Baseline + ATDAM + MSDA + EABRM (full EAMSNet) |

```bash
python scripts/train_ablation.py --data-root /path/LEVIR-CD-256 --epochs 120 --out-dir outputs
```

**Resumable**: it can be stopped at any time; rerun the same command — finished
configurations are skipped, and interrupted ones resume from the last epoch.
Combined metrics are written to `outputs/ablation_results.csv`.

### Ablation visualization

```bash
# one sample tested across all configurations (T1|T2|GT|Prediction|Error Map)
python scripts/visualize_ablation.py --mode sample --idx 25 --out-dir outputs

# grid of change-feature magnitude heatmaps
python scripts/visualize_ablation.py --mode heatmap --n 5 --out-dir outputs
```

In the *Error Map*: green = TP, red = FP, blue = FN.

## Configuration

`configs/default.yaml` holds the hyperparameters (backbone, width, epochs, per
param-group learning rates, EMA, etc.). The `use_atdam` / `use_msda` / `use_eabrm`
flags control which modules are active; for full-model training keep all three `true`.

## Reproducibility notes

Results depend on the input resolution. When comparing against other papers, make
sure the resolution (256 vs 512) and evaluation protocol are equivalent. The scripts
use AMP (mixed precision) and EMA by default; the seed is set via `--seed`.
