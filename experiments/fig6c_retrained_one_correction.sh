#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig6c_retrained_one_correction "training of the models with one correction (fig7a_ablations)"
"$PYTHON" -m afterwm_stats.aggregate --outputs "$OUTPUTS" --out "$RESULTS/table1.json"
"$PYTHON" -m afterwm_stats.per_step --outputs "$OUTPUTS" --out "$RESULTS/per_step.json"
"$PYTHON" -m analysis.mechanism --outputs "$OUTPUTS" --out "$RESULTS/mechanism.json"
"$PYTHON" -m figures.fig6_three_designs --results "$RESULTS" --out "$RESULTS/figures/fig6_three_designs.pdf"
