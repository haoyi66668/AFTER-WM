#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage table2_alfa "stage 1, stage 2 and evaluation on ALFA"
"$PYTHON" -m afterwm_stats.aggregate --outputs "$OUTPUTS" --out "$RESULTS/table1.json"
