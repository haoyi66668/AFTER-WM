#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
RUNS_ROOT="${AFTERWM_RUNS_ROOT:-runs}"
OUT="${AFTERWM_REGENERATED_OUTPUTS:-$RESULTS/regenerated/outputs}"
ROWS="$RESULTS/regenerated/rows"
DEVICE="${AFTERWM_DEVICE:-cuda}"
SEEDS="20260910 20270913 20280916 20290919 20300922"
SIM="jshc escphc panda ur5e hopper"
mkdir -p "$OUT" "$ROWS"
evaluate_all() {
  local ds="$1" root="$2"
  for run in "$root"/*/*; do
    [ -f "$run/best.pt" ] || continue
    [ -f "$run/eval/COMPLETE.json" ] && continue
    local extra=""
    case "$(basename "$(dirname "$run")")" in rssm) extra="--rssm-eval-mode sample16" ;; esac
    "$PYTHON" -m pipeline.evaluate --dataset "$ds" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --run "$run" $extra "${@:3}"
  done
}
for ds in $SIM alfa; do
  evaluate_all "$ds" "$RUNS_ROOT/runs/$ds"
  "$PYTHON" -m afterwm_stats.compact per-query --dataset "$ds" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --outputs "$OUT"
done
for ds in $SIM; do
  "$PYTHON" -m afterwm_stats.compact per-step --dataset "$ds" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --outputs "$OUT"
  for s in $SEEDS; do
    run="$RUNS_ROOT/runs/$ds/afterwm/$s"
    [ -f "$run/best.pt" ] || continue
    for mod in removal routing interventions; do
      "$PYTHON" -m "analysis.$mod" --dataset "$ds" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --seed "$s" --run "$run" --out "$ROWS/$mod/$ds" --device "$DEVICE" || true
    done
  done
  "$PYTHON" -m afterwm_stats.compact removal --dataset "$ds" --data-root "$DATA_ROOT" --rows "$ROWS/removal/$ds" --outputs "$OUT"
  "$PYTHON" -m afterwm_stats.compact routing --dataset "$ds" --data-root "$DATA_ROOT" --rows "$ROWS/routing/$ds" --outputs "$OUT"
  "$PYTHON" -m afterwm_stats.compact interventions --dataset "$ds" --data-root "$DATA_ROOT" --rows "$ROWS/interventions/$ds" --outputs "$OUT"
done
for arm in c1ctx32 c1ctx128; do
  evaluate_all escphc "$RUNS_ROOT/arms/$arm/runs" --arm "$arm"
  "$PYTHON" -m afterwm_stats.compact per-query --dataset escphc --arm "$arm" --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" \
    --set "escphc_ctx${arm#c1ctx}" --prepared "$DATA_ROOT/prepared/escphc_$arm" \
    --models afterwm,tdmpc2,tdmpc2_h40,rebal,mts3,mlp --eval-template "{runs_root}/arms/$arm/runs/{model}/{seed}/eval" --outputs "$OUT"
done
for pair in c1g005:afterwm_gamma_0.05 c1g05:afterwm_gamma_0.5; do
  arm="${pair%%:*}"
  evaluate_all escphc "$RUNS_ROOT/arms/$arm/runs" --arm "$arm"
  "$PYTHON" -m afterwm_stats.compact per-query --dataset escphc --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --set escphc \
    --models afterwm --as "${pair#*:}" --eval-template "{runs_root}/arms/$arm/runs/{model}/{seed}/eval" --outputs "$OUT"
done
for m in afterwm tdmpc2 tdmpc2_h40 rebal grbal mole mbcd mts3 rssm gpmm mlp; do
  for s in $SEEDS; do
    run="$RUNS_ROOT/runs/escphc/$m/$s"
    [ -f "$run/best.pt" ] || continue
    "$PYTHON" -m analysis.beyond_range --dataset escphc --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" --model "$m" --seed "$s" --run "$run" --out "$ROWS/beyond_range" --device "$DEVICE" || true
  done
done
"$PYTHON" -m afterwm_stats.compact beyond-range --dataset escphc --data-root "$DATA_ROOT" --rows "$ROWS/beyond_range" --outputs "$OUT"
mkdir -p "$ROWS/removal_all"
for ds in $SIM; do cp "$ROWS/removal/$ds/"*.npz "$ROWS/removal_all/" 2>/dev/null || true; done
"$PYTHON" -m analysis.probes --rows "$ROWS/removal_all" --data-root "$DATA_ROOT" --seeds "ur5e:20270913,20280916,20290919,20300922" --out "$OUT/analysis/probes.json"
CONTROL_SPEC="${AFTERWM_CONTROL_SPEC:-$DATA_ROOT/control_episodes/episodes_test_v1.npz}"
for row in afterwm tdmpc2 tdmpc2_h40 rebal grbal mole mbcd mts3 rssm gpmm mlp; do
  PYTHONPATH="$PWD:$PWD/datasets" "$PYTHON" -m control.run --dataset jshc --data-root "$DATA_ROOT" --runs-root "$RUNS_ROOT" \
    --spec "$CONTROL_SPEC" --out "$ROWS/control" --row "$row" --device "$DEVICE"
done
"$PYTHON" -m afterwm_stats.compact control --rows "$ROWS/control" --outputs "$OUT"
"$PYTHON" datasets/labels.py --data-root "$DATA_ROOT" --out "$OUT/labels"
echo "regenerated outputs in $OUT (outputs/examples/ is not rebuilt by this script, see docs/REPRODUCE.md); run the entry scripts with AFTERWM_OUTPUTS=$OUT"
