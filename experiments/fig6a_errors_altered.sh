#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig6a_errors_altered "the alterations of the errors at test time (analysis.interventions)"
"$PYTHON" -m analysis.mechanism --outputs "$OUTPUTS" --out "$RESULTS/mechanism.json"
