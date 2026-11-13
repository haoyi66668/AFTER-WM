#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig4_robot_arm_examples "example extraction (analysis.examples_extract) and rendering (analysis.examples_render)"
"$PYTHON" -m analysis.examples_select --data-root "$DATA_ROOT" --outputs "$OUTPUTS" --out "$RESULTS/examples.json"
"$PYTHON" -m analysis.per_joint --outputs "$OUTPUTS" --out "$RESULTS/per_joint.json"
"$PYTHON" -m figures.fig4_robot_arm --outputs "$OUTPUTS" --out "$RESULTS/figures/fig4_robot_arm.pdf"
