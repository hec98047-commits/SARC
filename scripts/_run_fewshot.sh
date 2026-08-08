#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then echo "Usage: $0 <mvtec|visa> <1|2|4> <data_root> <fgclip_model> <sarc_model> <output_dir>" >&2; exit 2; fi
DATASET="$1"; SHOTS="$2"; DATA_ROOT="$3"; FGCLIP_MODEL="$4"; SARC_MODEL="$5"; OUTPUT_DIR="$6"
python src/sarc/run_sarc_protocol.py --dataset "$DATASET" --data_root "$DATA_ROOT" --model_path "$FGCLIP_MODEL" --sarc_model_path "$SARC_MODEL" --output_dir "$OUTPUT_DIR" --shots_list "$SHOTS" --seeds 42
