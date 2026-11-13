#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig7a_ablations "training of the four ablations on the same pre-trained base"
"$PYTHON" -m afterwm_stats.aggregate --outputs "$OUTPUTS" --out "$RESULTS/table1.json"
