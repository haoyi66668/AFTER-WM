#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
gpu_stage fig6d_routing_without_errors "the routing test without errors (analysis.routing)"
"$PYTHON" -m analysis.mechanism --outputs "$OUTPUTS" --out "$RESULTS/mechanism.json"
