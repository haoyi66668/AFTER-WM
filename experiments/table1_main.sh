#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage table1_main "stage 1, stage 2 and evaluation of AFTER-WM and the ten compared models on the five simulated datasets"
"$PYTHON" -m afterwm_stats.aggregate --outputs "$OUTPUTS" --out "$RESULTS/table1.json"
"$PYTHON" -m figures.table1 --results "$RESULTS" --out "$RESULTS/tables/table1.tex"
