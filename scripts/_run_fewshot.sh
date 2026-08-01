#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then echo "Usage: $0 <mvtec|visa> <1|2|4> <data_root> <fgclip_ckpt> <mg_ckpt> <output_dir>" >&2; exit 2; fi
DATASET="$1"; SHOTS="$2"; DATA_ROOT="$3"; FGCLIP_CKPT="$4"; MG_CKPT="$5"; OUTPUT_DIR="$6"
python src/pgcre_fgclip/run_fewshot_protocol.py --dataset "$DATASET" --data_root "$DATA_ROOT" --model_path "$FGCLIP_CKPT" --mg_model_path "$MG_CKPT" --output_dir "$OUTPUT_DIR" --shots_list "$SHOTS" --seeds 42 --methods ours
