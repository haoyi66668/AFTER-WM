#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage table2_beyond_range "re-scoring of the eleven models on the ESCP-HC test files beyond the training range (analysis.beyond_range)"
"$PYTHON" -m analysis.mechanism --outputs "$OUTPUTS" --out "$RESULTS/mechanism.json"
