#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig1b_change_phases "evaluation of the Table 1 runs"
"$PYTHON" -m afterwm_stats.phases --outputs "$OUTPUTS" --out "$RESULTS/phases.json"
"$PYTHON" -m figures.fig1b_change_phases --results "$RESULTS" --out "$RESULTS/figures/fig1b_change_phases.pdf"
