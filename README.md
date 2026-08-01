# PGCRE-FGCLIP

**Few-shot tiny industrial defect localization using normal references only.**

PGCRE-FGCLIP combines a frozen FG-CLIP backbone with cross-modal semantic alignment, candidate-guided local response decoupling, and local-detail compensation. The released protocol uses `K = 1, 2, 4` normal reference images per category with seed `42`; no anomalous training images or test masks are used for optimization.

> This repository contains source code only. Datasets, model weights, checkpoints, experiment outputs, logs, and caches are not distributed.

## Framework

<p align="center">
  <img src="assets/pgcre_fgclip_framework.png" width="100%" alt="Overall framework of PGCRE-FGCLIP">
</p>

The method is organized into three modules:

- **Module A — Cross-modal semantic alignment.** Spatial-aware prompts, low-risk prompt routing, and semantic candidate priors identify likely defect regions.
- **Module B — Candidate-guided local response decoupling.** A shared lightweight CandidateTokenModulator is trained from normal references and applied to frozen visual Transformer layers 3–12. The accompanying attention bias is fixed and has no trainable parameters.
- **Module C — Local-detail compensation.** Overlapping local inference, positive residual fusion, and normal-reference calibration recover weak defect evidence while suppressing false positives.

Only the CandidateTokenModulator in Module B is trainable. The visual encoder, text encoder, attention layers, and the remaining modules stay frozen. Runtime messages use only `Module A`, `Module B`, and `Module C`, without manuscript section numbering.

## Installation

Python 3.10 and a CUDA-enabled PyTorch build are recommended.

```bash
git clone https://github.com/hec98047-commits/PGCRE-FGCLIP.git
cd PGCRE-FGCLIP
conda env create -f environment.yml
conda activate pgcre-fgclip
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Data and checkpoints

Download MVTec AD and VisA from their official sources. Keep datasets and checkpoints outside version control.

```text
/path/to/mvtec/
  bottle/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*_mask.png

/path/to/visa/
  split_csv/1cls.csv
  <category>/...
```

The repository includes the model source and configuration under `models/FGCLIP/` and `models/MGFGGCLIP/`, but not pretrained weights or tokenizer assets.

## Train Module B

First sample the normal references:

```bash
python src/pgcre_fgclip/fewshot_sampler.py \
  --dataset visa \
  --data_root /path/to/visa \
  --shots 1 \
  --seed 42 \
  --output_path outputs/samples/visa_1shot_seed42.json
```

Then train the normal-only CandidateTokenModulator:

```bash
python src/pgcre_fgclip/train_candidate_token_modulator.py \
  --model_dir /path/to/MGFGGCLIP \
  --sampled_normals outputs/samples/visa_1shot_seed42.json \
  --output_dir checkpoints/visa_1shot_seed42 \
  --epochs 10
```

Repeat with `--shots 2` and `--shots 4` when reproducing the corresponding protocols. The default release configuration resolves checkpoints as:

```text
checkpoints/{dataset}_{shot}shot_seed{seed}/ctm_epoch10_trained.pt
```

## Evaluation

Edit the placeholder paths in `configs/release_visa.yaml` or override them from the command line:

```bash
python src/pgcre_fgclip/run_fewshot_protocol.py \
  --config configs/release_visa.yaml \
  --data_root /path/to/visa \
  --model_path /path/to/FGCLIP \
  --mg_model_path /path/to/MGFGGCLIP \
  --output_dir outputs/visa
```

Use `configs/release_mvtec.yaml` for MVTec AD. P-AUROC and AU-PRO are written to `eval_metrics.json` and the summary CSV files. AU-PRO is implemented in `src/pgcre_fgclip/fgclip_ad/metrics.py` with an FPR integration limit of `0.3`.

## VisA results

**Table B1. Category-level results of PGCRE-FGCLIP on VisA under the 1/2/4-shot protocols (%).**

| Category | 1-shot P-AUROC | 1-shot AU-PRO | 2-shot P-AUROC | 2-shot AU-PRO | 4-shot P-AUROC | 4-shot AU-PRO |
|---|---:|---:|---:|---:|---:|---:|
| candle | 97.80 | 97.01 | 98.18 | 97.94 | 98.36 | 97.82 |
| capsules | 96.40 | 89.90 | 96.38 | 90.44 | 95.73 | 89.61 |
| cashew | 96.90 | 85.84 | 96.35 | 82.19 | 97.15 | 89.14 |
| chewinggum | 99.22 | 87.77 | 99.22 | 88.15 | 99.50 | 89.27 |
| fryum | 96.89 | 91.67 | 97.05 | 91.50 | 97.12 | 91.52 |
| macaroni1 | 99.56 | 98.23 | 99.66 | 98.93 | 99.95 | 98.60 |
| macaroni2 | 98.59 | 92.65 | 98.29 | 92.85 | 98.64 | 92.30 |
| pcb1 | 98.88 | 85.02 | 98.82 | 83.00 | 99.20 | 87.36 |
| pcb2 | 96.00 | 91.47 | 96.86 | 93.36 | 96.81 | 93.30 |
| pcb3 | 94.10 | 89.43 | 94.81 | 91.43 | 94.76 | 92.56 |
| pcb4 | 95.94 | 90.26 | 96.39 | 92.11 | 96.52 | 91.88 |
| pipe_fryum | 99.45 | 89.32 | 99.56 | 90.22 | 99.70 | 90.71 |
| **Average** | **97.48** | **90.71** | **97.63** | **91.01** | **97.79** | **92.01** |

### Qualitative comparison on VisA

<p align="center">
  <img src="assets/qualitative_visa.png" width="100%" alt="Qualitative comparison on VisA">
</p>

The visualization compares the input image, ground-truth mask, WinCLIP, PromptAD, frozen FG-CLIP, and PGCRE-FGCLIP. It is provided as qualitative evidence and does not replace the quantitative evaluation protocol.

## Repository structure

```text
PGCRE-FGCLIP/
├── assets/                         # framework and VisA visualization
├── configs/                        # release configurations
├── models/                         # FG-CLIP and MG-FGCLIP model source
├── scripts/                        # 1/2/4-shot launchers
└── src/pgcre_fgclip/               # training, evaluation, and method code
```

## Citation

Update the following entry after the final publication metadata is available:

```bibtex
@article{pgcre_fgclip,
  title   = {PGCRE-FGCLIP},
  author  = {Authors},
  journal = {Manuscript},
  year    = {2026}
}
```

FG-CLIP, MVTec AD, VisA, and external checkpoints remain subject to their original licenses.
