#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage supp_e3_probes "the state rows of analysis.removal"
"$PYTHON" -m analysis.probes --scores "$OUTPUTS/analysis/probes.json" --out "$RESULTS/probes.json"
