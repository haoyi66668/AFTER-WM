#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig7b_sensitivity "training with the other context lengths and inverse routing temperatures (configs/arms.json)"
"$PYTHON" -m afterwm_stats.aggregate --outputs "$OUTPUTS" --out "$RESULTS/table1.json"
"$PYTHON" -m afterwm_stats.sensitivity --outputs "$OUTPUTS" --out "$RESULTS/sensitivity.json"
"$PYTHON" -m figures.fig7_ablation_sensitivity --results "$RESULTS" --out "$RESULTS/figures/fig7_ablation_sensitivity.pdf"
