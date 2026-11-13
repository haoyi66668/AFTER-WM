#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage sec42_windows "evaluation of the Table 1 runs"
"$PYTHON" -m analysis.windows --outputs "$OUTPUTS" --out "$RESULTS/windows.json"
