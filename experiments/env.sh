set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/datasets${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
RESULTS="${RESULTS:-results}"
OUTPUTS="${AFTERWM_OUTPUTS:-outputs}"
DATA_ROOT="${AFTERWM_DATA_ROOT:-data}"
MODE="--from-outputs"
for arg in "$@"; do
  case "$arg" in
    --from-outputs|--from-checkpoints|--from-scratch) MODE="$arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
mkdir -p "$RESULTS"
gpu_stage() {
  if [ "$MODE" != "--from-outputs" ]; then
    echo "$1: $MODE needs $2. Train the models (docs/REPRODUCE.md), put the runs under \$AFTERWM_RUNS_ROOT and run experiments/regenerate_outputs.sh (GPU). It writes the per-query, per-step, analysis, control and label outputs; outputs/examples/ is not rebuilt by it (copy that folder into the regenerated outputs or see docs/REPRODUCE.md). Then run this script with AFTERWM_OUTPUTS set to the regenerated outputs." >&2
    exit 3
  fi
}
