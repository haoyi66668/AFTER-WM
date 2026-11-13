#!/usr/bin/env bash
source "$(dirname "$0")/env.sh" "$@"
for s in table1_main fig3_per_step fig1b_change_phases sec42_windows fig4_robot_arm_examples supp_e3_probes fig6c_retrained_one_correction fig7b_sensitivity table2_control; do
  bash "experiments/$s.sh" "$MODE"
done
"$PYTHON" -m figures.table2 --results "$RESULTS" --out "$RESULTS/tables/table2.tex"
"$PYTHON" -m afterwm_stats.paper_numbers --results "$RESULTS" --out "$RESULTS/paper_numbers.json"
echo "done: results written to $RESULTS"
