# SARC

Official research code for **Semantic-Guided Anomaly Response Constraint for Few-Shot Localization of Tiny Industrial Defects with Frozen Vision-Language Models**.

SARC uses only normal reference images and keeps the FG-CLIP backbone frozen. The public pipeline follows the three stages in the manuscript:

- **SP — Semantic Prior Generation:** spatial-aware anomaly prompts, risk-aware prompt selection, and cross-modal semantic-prior generation.
- **ARC — Anomaly Response Constraint:** prior-guided token modulation and attention-bias constraints in the visual Transformer.
- **LEC — Local Evidence Calibration:** overlapping local inference, local/global fusion, and normal-reference residual calibration.

## Framework

<p align="center">
  <img src="assets/sarc_framework.png" width="100%" alt="SARC framework with SP, ARC, and LEC stages">
</p>

## Installation

```bash
git clone https://github.com/hec98047-commits/SARC.git
cd SARC
conda env create -f environment.yml
conda activate sarc
```

Alternatively, install the Python dependencies with `pip install -r requirements.txt`.

The repository contains non-weight model source and configuration files only:

- `models/FGCLIP/`: frozen FG-CLIP backbone definition;
- `models/SARC/`: SARC model definition and ARC-enabled vision configuration.

Provide pretrained weights and tokenizer files locally. Checkpoints, datasets, caches, logs, and generated outputs are excluded from Git.

## Dataset protocol

Prepare VisA or MVTec AD using their official layouts. For each category, SARC samples `K = 1, 2, 4` normal reference images with seed `42`. Anomalous training images and test masks are not used for optimization.

Edit `configs/release_visa.yaml` or `configs/release_mvtec.yaml`, then run:

```bash
python src/sarc/run_sarc_protocol.py --config configs/release_visa.yaml
```

Single-shot convenience scripts are also provided:

```bash
bash scripts/run_visa_1shot.sh /path/to/visa /path/to/FGCLIP /path/to/SARC outputs/visa_1shot
```

At runtime, the method stages are reported only as `SP`, `ARC`, and `LEC`.

## ARC training

ARC contains the lightweight trainable token modulator; the FG-CLIP backbone remains frozen.

```bash
python src/sarc/train_arc.py \
  --dataset visa \
  --data_root /path/to/visa \
  --sampled_normals outputs/samples/1shot/sampled_normal_paths.json \
  --model_dir /path/to/SARC \
  --output_dir checkpoints/visa_1shot_seed42
```

## VisA results

Pixel AUROC (P-AUROC) and region overlap (AU-PRO) are percentages.

| Class | 1-shot P-AUROC | 1-shot AU-PRO | 2-shot P-AUROC | 2-shot AU-PRO | 4-shot P-AUROC | 4-shot AU-PRO |
|---|---:|---:|---:|---:|---:|---:|
| candle | 97.86 | 97.12 | 98.18 | 97.94 | 98.36 | 97.82 |
| capsules | 96.41 | 89.92 | 96.38 | 90.44 | 95.73 | 89.61 |
| cashew | 96.89 | 85.88 | 96.35 | 82.19 | 97.15 | 89.14 |
| chewinggum | 99.34 | 88.41 | 99.22 | 88.15 | 99.50 | 89.27 |
| fryum | 96.89 | 91.70 | 97.05 | 91.50 | 97.12 | 91.52 |
| macaroni1 | 99.47 | 97.98 | 99.66 | 98.93 | 99.95 | 98.60 |
| macaroni2 | 98.47 | 92.09 | 98.29 | 92.85 | 98.64 | 92.30 |
| pcb1 | 98.88 | 85.04 | 98.82 | 83.00 | 99.20 | 87.36 |
| pcb2 | 96.16 | 91.33 | 96.86 | 93.36 | 96.81 | 93.30 |
| pcb3 | 95.21 | 91.20 | 94.81 | 91.43 | 94.76 | 92.56 |
| pcb4 | 95.94 | 89.71 | 96.39 | 92.11 | 96.52 | 91.88 |
| pipe_fryum | 99.45 | 89.33 | 99.56 | 90.22 | 99.70 | 90.71 |
| **Average** | **97.58** | **90.81** | **97.63** | **91.01** | **97.79** | **92.01** |

## Qualitative results

<p align="center">
  <img src="assets/sarc_qualitative.png" width="100%" alt="Qualitative tiny-defect localization comparison">
</p>

## Repository layout

```text
SARC/
├── assets/          # Paper framework and qualitative visualization
├── configs/         # Release protocols
├── models/          # FG-CLIP backbone and SARC model definitions
├── scripts/         # 1/2/4-shot launchers
├── src/sarc/        # SP, ARC, LEC implementation and evaluation
├── .gitignore
├── LICENSE
├── README.md
├── README_CN.md
├── environment.yml
└── requirements.txt
```

## License

This repository is released under the MIT License. Third-party datasets and pretrained models retain their original licenses.
