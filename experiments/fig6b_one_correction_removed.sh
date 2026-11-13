#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig6b_one_correction_removed "test-time removal of one correction (analysis.removal)"
"$PYTHON" -m analysis.mechanism --outputs "$OUTPUTS" --out "$RESULTS/mechanism.json"
