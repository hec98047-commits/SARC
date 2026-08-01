#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname "$0")/_run_fewshot.sh" visa 1 "$@"
