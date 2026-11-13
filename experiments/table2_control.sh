#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage table2_control "the closed-loop runs of the eleven world models with the cross-entropy-method planner (control.run)"
"$PYTHON" -m control.analysis --returns "$OUTPUTS/control/returns.npz" --outputs "$OUTPUTS" --out "$RESULTS/control.json"
