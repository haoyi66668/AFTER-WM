#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: regenerate_all.sh OUT_DIR [jshc escphc escphc_beyond panda ur5e hopper control_episodes alfa]}"
shift
SETS="${*:-jshc escphc escphc_beyond panda ur5e hopper control_episodes alfa}"
PY_GEN="${PY_GEN:-python3}"
PY_ALFA="${PY_ALFA:-python3}"
ALFA_ZIP="${ALFA_ZIP:-$HERE/alfa/processed.zip}"
CONTROL_ROOT="${CONTROL_ROOT:-$(dirname "$HERE")}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
RUN=(env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1)
cd "$HERE"
for s in $SETS; do
  echo "== $s $(date -u +%H:%M:%S)"
  case "$s" in
    jshc)
      "${RUN[@]}" "$PY_GEN" -B -m jshc.tests.run_tests --tmp "$OUT/work/jshc_tests"
      "${RUN[@]}" "$PY_GEN" -B -m jshc.cli collect --counts train=24,val=12,test=12 --seed-offset 0 --workers 4 --out "$OUT/work/jshc/main"
      "${RUN[@]}" "$PY_GEN" -B -m jshc.cli collect --counts test=12 --seed-offset 1 --workers 2 --out "$OUT/work/jshc/testB"
      "${RUN[@]}" "$PY_GEN" -B -m jshc.cli assemble --main "$OUT/work/jshc/main" --testb "$OUT/work/jshc/testB" --out "$OUT/jshc"
      ;;
    escphc)
      "${RUN[@]}" "$PY_GEN" -B -m escphc.run_tests
      "${RUN[@]}" "$PY_GEN" -B -m escphc.cli collect --out "$OUT/escphc" --log-scale-limit 3
      ;;
    escphc_beyond)
      "${RUN[@]}" "$PY_GEN" -B -m escphc.cli collect --out "$OUT/escphc_beyond" --log-scale-limit 4
      ;;
    panda|ur5e|hopper)
      "${RUN[@]}" "$PY_GEN" -B -m "$s.tests.run_tests"
      "${RUN[@]}" "$PY_GEN" -B -m "$s.cli" collect --design "$s" --counts train=24,val=12,test=12 --seed-offset 0 --out "$OUT/work/$s/main"
      "${RUN[@]}" "$PY_GEN" -B -m "$s.cli" collect --design "$s" --counts test=12 --seed-offset 1 --out "$OUT/work/$s/testB"
      "${RUN[@]}" "$PY_GEN" -B -m "$s.cli" assemble --main "$OUT/work/$s/main" --testb "$OUT/work/$s/testB" --out "$OUT/$s"
      ;;
    control_episodes)
      mkdir -p "$OUT/control_episodes"
      env PYTHONPATH="$CONTROL_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
        "$PY_GEN" -B -m control_episodes --split test --n 48 --out "$OUT/control_episodes/episodes_test_v1.npz"
      ;;
    alfa)
      env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
        "$PY_ALFA" -I "$HERE/alfa/alfa_windows.py" define --zip "$ALFA_ZIP" --quotas "$HERE/alfa/QUOTAS.json" --out "$OUT/alfa"
      ;;
    *)
      echo "unknown dataset $s" >&2
      exit 2
      ;;
  esac
done
DS="$(echo $SETS | tr ' ' ',')"
"$PY_ALFA" -B "$HERE/check_data.py" --data "$OUT" --datasets "$DS" --report "$OUT/CHECK.json"
