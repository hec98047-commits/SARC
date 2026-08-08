#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then echo "Usage: $0 <mvtec_root> <visa_root> <fgclip_model> <sarc_model> <mvtec_output_root> <visa_output_root>" >&2; exit 2; fi
for SHOTS in 1 2 4; do
  bash "$(dirname "$0")/_run_fewshot.sh" mvtec "$SHOTS" "$1" "$3" "$4" "$5/${SHOTS}shot"
  bash "$(dirname "$0")/_run_fewshot.sh" visa "$SHOTS" "$2" "$3" "$4" "$6/${SHOTS}shot"
done
