#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig3_per_step "per-step errors of the Table 1 runs"
"$PYTHON" -m afterwm_stats.per_step --outputs "$OUTPUTS" --out "$RESULTS/per_step.json"
"$PYTHON" -m figures.fig3_per_step --results "$RESULTS" --out "$RESULTS/figures/fig3_per_step.pdf"
