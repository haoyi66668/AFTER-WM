# Results map

Paper item -> entry script -> code -> stored outputs read -> files written, for the items that this repository reproduces (table 'What this repository reproduces' in `README.md`). All paths are relative to the repository root; `results/` is the default output folder (`RESULTS`), `outputs/` the default input folder (`AFTERWM_OUTPUTS`). Every entry script reads its settings from `experiments/env.sh`. `experiments/all_from_outputs.sh` runs these scripts and additionally writes Table 2 and `results/paper_numbers.json` (it calls nine of the scripts below; the commands of the other seven are among the commands of these nine).

| Paper item | Script | Code | Reads | Writes |
|---|---|---|---|---|
| Table 1 (multi-step mean and step-40 errors, five simulated datasets, eleven models, last row against the strongest compared model) | `experiments/table1_main.sh` | `afterwm_stats.aggregate`, `figures.table1` | `outputs/per_query/` (the five simulated datasets) | `results/table1.json` (`main`, `main_summary`), `results/tables/table1.tex` |
| Fig. 1b and change phases (Sec. 4.2) | `experiments/fig1b_change_phases.sh` | `afterwm_stats.phases`, `figures.fig1b_change_phases` | `outputs/per_query/`, `outputs/labels/` | `results/phases.json`, `results/figures/fig1b_change_phases.pdf`, `results/figures/fig1b_change_phases.png`, `results/figures/fig1b_change_phases_data.json` |
| Fig. 3 and per-step comparisons (Sec. 4.2) | `experiments/fig3_per_step.sh` | `afterwm_stats.per_step`, `figures.fig3_per_step` | `outputs/per_step/` | `results/per_step.json`, `results/figures/fig3_per_step.pdf`, `results/figures/fig3_per_step.png`, `results/figures/fig3_per_step_data.json` |
| Queries without a change and with transient disturbances (Sec. 4.2) | `experiments/sec42_windows.sh` | `analysis.windows` | `outputs/per_query/`, `outputs/labels/` | `results/windows.json` |
| Fig. 4 (arm poses with link deviations) and per-joint step-40 errors (Sec. 4.2) | `experiments/fig4_robot_arm_examples.sh` | `analysis.examples_select`, `analysis.per_joint`, `figures.fig4_robot_arm` | `outputs/per_query/panda.npz`, `outputs/per_query/ur5e.npz`, `outputs/examples/panda.json`, `outputs/examples/ur5e.json`, `outputs/examples/frames/`, `data/panda/`, `data/ur5e/` | `results/examples.json`, `results/per_joint.json`, `results/figures/fig4_robot_arm.pdf`, `results/figures/fig4_robot_arm.png`, `results/figures/fig4_robot_arm_data.json` |
| Fig. 6a, alterations of the errors at test time | `experiments/fig6a_errors_altered.sh` | `analysis.mechanism` | `outputs/analysis/interventions.npz` | `results/mechanism.json` (`interventions`) |
| Fig. 6b, one correction removed at test time (steps 1 and 40, by phase) | `experiments/fig6b_one_correction_removed.sh` | `analysis.mechanism` | `outputs/analysis/removal.npz` | `results/mechanism.json` (`removal`) |
| Fig. 6c, models retrained with one correction; draws Fig. 6 | `experiments/fig6c_retrained_one_correction.sh` | `afterwm_stats.aggregate`, `afterwm_stats.per_step`, `analysis.mechanism`, `figures.fig6_three_designs` | `outputs/per_query/`, `outputs/per_step/`, `outputs/analysis/` | `results/table1.json` (`single_correction`), `results/figures/fig6_three_designs.pdf`, `results/figures/fig6_three_designs.png`, `results/figures/fig6_three_designs_data.json` |
| Fig. 6d, the routing test without errors | `experiments/fig6d_routing_without_errors.sh` | `analysis.mechanism` | `outputs/analysis/routing.npz` | `results/mechanism.json` (`routing`) |
| Fig. 7a, ablations | `experiments/fig7a_ablations.sh` | `afterwm_stats.aggregate` | `outputs/per_query/` | `results/table1.json` (`ablations`) |
| Fig. 7b, sensitivity on ESCP-HC; draws Fig. 7 | `experiments/fig7b_sensitivity.sh` | `afterwm_stats.aggregate`, `afterwm_stats.sensitivity`, `figures.fig7_ablation_sensitivity` | `outputs/per_query/escphc.npz`, `outputs/per_query/escphc_ctx32.npz`, `outputs/per_query/escphc_ctx128.npz` | `results/sensitivity.json`, `results/figures/fig7_ablation_sensitivity.pdf`, `results/figures/fig7_ablation_sensitivity.png`, `results/figures/fig7_ablation_sensitivity_data.json` |
| Linear probes of the fast and slow states (supp. Sec. E.3; numbers only) | `experiments/supp_e3_probes.sh` | `analysis.probes` | `outputs/analysis/probes.json` | `results/probes.json` |
| Table 2, beyond the training range | `experiments/table2_beyond_range.sh` | `analysis.mechanism` | `outputs/analysis/beyond_range.npz`, `outputs/labels/labels_escphc_beyond.npz` | `results/mechanism.json` (`beyond_range`) |
| Table 2, no change | `experiments/table2_no_change.sh` | `analysis.windows` | `outputs/per_query/`, `outputs/labels/` | `results/windows.json` |
| Table 2, real flights (ALFA, 4 seeds) | `experiments/table2_alfa.sh` | `afterwm_stats.aggregate` | `outputs/per_query/alfa.npz` | `results/table1.json` (`alfa`) |
| Table 2, control (JSHC, 48 episodes, 5 seeds) | `experiments/table2_control.sh` | `control.analysis` | `outputs/control/returns.npz`, `outputs/per_query/jshc.npz` | `results/control.json` |
| Table 2 | `experiments/all_from_outputs.sh` | `figures.table2` | the result files above | `results/tables/table2.tex` |
| Numbers of the paper text that follow from the stored outputs | `experiments/all_from_outputs.sh` | `afterwm_stats.paper_numbers` | the result files above | `results/paper_numbers.json` |

Fig. 6 is drawn by `figures/fig6_three_designs.py`, Fig. 7 by `figures/fig7_ablation_sensitivity.py`. Fig. 1a and Fig. 2 are drawings without data.

## Keys of `results/paper_numbers.json`

The values are formatted as the paper prints them. The last column says where the paper states the value: in the main text, or in the supplement.

Main-text keys:

| Key | Meaning | Where the paper states it |
|---|---|---|
| `main.strongest_models` | the strongest compared models of the five simulated datasets | main, Sec. 4.2 |
| `main.all_seeds_lower` | comparisons with a compared model in which the multi-step mean error of AFTER-WM is lower in all 5 seeds | main, Sec. 4.2 |
| `main.ci_below_zero` | comparisons whose 95% interval of the difference lies below zero | main, Sec. 4.2 |
| `main.vs_strongest_range` | range of the reduction of the multi-step mean error against the strongest compared model (%) | main, Sec. 4.2 |
| `main.vs_strongest_range_rounded` | the same range, rounded to whole percent | main, abstract |
| `main.afterwm_best_both_metrics` | whether AFTER-WM is best on both errors on the five simulated datasets | main, Sec. 4.2 |
| `per_step.lowest_from_step` | step from which AFTER-WM has the lowest per-step error on every simulated dataset | main, Sec. 4.2 |
| `per_step.vs_strongest_from_step` | step from which the intervals against the strongest compared model lie below zero | main, Sec. 4.2 |
| `per_step.step40_range` | range of the reduction of the step-40 error against the strongest compared model (%) | main, Sec. 4.2 |
| `per_step.clipped` | dataset, model and step from which a curve lies above the axis of the left plots | main, Fig. 3 |
| `phases.phase1_min` | smallest reduction 1–4 steps after a change (%) | main, Sec. 4.2 |
| `phases.escphc_all_positive` | whether the reductions are positive in the phases of ESCP-HC | main, Sec. 4.2 |
| `windows.no_change_min` | smallest margin in queries without a change (%) | main, Sec. 4.2 and Table 2 |
| `windows.transient_min` | smallest margin in queries with transient disturbances on Panda, UR5e and Hopper (%) | main, Sec. 4.2 |
| `examples.steps_since_change` | steps from the change to the prediction start in the two robot-arm examples | main, Fig. 4 |
| `examples.panda_payload_after` | payload of the Panda example after the change (kg) | main, Sec. 4.2 |
| `examples.step40` | mean link deviation at step 40 of AFTER-WM and of the compared model | main, Sec. 4.2 |
| `examples.less_than_half` | whether the step-40 link deviation of AFTER-WM is less than half of the compared model's | main, Sec. 4.2 |
| `per_joint` | joints of the two arms on which the step-40 error of AFTER-WM is lower | main, Sec. 4.2 |
| `interventions.other_sequence` | range of the error increase with another sequence's errors (%) | main, Sec. 4.4 |
| `interventions.other_sequence_largest` | whether this alteration raises the error most on every dataset | main, Sec. 4.4 |
| `removal.no_fast_step1` | range of the step-1 error increase without the fast correction (%) | main, Sec. 4.4 |
| `removal.no_slow_step40` | range of the step-40 error increase without the slow correction (%) | main, Sec. 4.4 |
| `removal.other_step_max` | largest change of the error at the other step (%) | main, Sec. 4.4 |
| `removal.seeds` | seeds per dataset in the test that removes one correction | main, Fig. 6 |
| `routing.without_escphc` | range of the error increase in the routing test without errors, on the datasets other than ESCP-HC (%) | main, Sec. 4.4 |
| `routing.escphc` | the same increase on ESCP-HC (%) | main, Sec. 4.4 |
| `routing.seeds` | seeds per dataset in the routing test without errors | main, Fig. 6 |
| `ablations.range` | range of the error increase of the four ablations (%) | main, Sec. 4.6 |
| `ablations.all_positive` | whether every ablation raises the error on every dataset | main, Sec. 4.6 |
| `beyond.min` | smallest margin beyond the training range (%) | main, Sec. 4.5 and Table 2 |
| `beyond.beaten` | compared models beaten beyond the training range | main, Sec. 4.5 and Table 2 |
| `beyond.closest` | closest compared model beyond the training range | main, Table 2 |
| `alfa.seeds` | seeds used on ALFA | main, Sec. 4.5 |
| `alfa.min` | smallest margin on ALFA (%) | main, Sec. 4.5 and Table 2 |
| `alfa.closest` | closest compared model on ALFA | main, Table 2 |
| `alfa.all_seeds` | whether AFTER-WM is lower than each compared model in all 4 seeds | main, Sec. 4.5 |
| `control.afterwm` | mean return of AFTER-WM in the control task | main, Sec. 4.5 |
| `control.gap` | gap of the mean return to the second best model | main, Sec. 4.5 and Table 2 |
| `control.second` | second best model in the control task | main, Sec. 4.5 and Table 2 |
| `control.highest` | whether AFTER-WM has the highest mean return | main, Sec. 4.5 |
| `control.spearman` | Spearman rank correlation between accuracy and mean return | main, Sec. 4.5 |
| `control.episodes` | test episodes of the control task | main, Sec. 4.5 |
| `control.seeds` | seeds of the control task | main, Sec. 4.5 |
| `sensitivity.n` | cells of Fig. 7b with a changed setting | main, Sec. 4.6 |
| `sensitivity.all_seeds` | whether AFTER-WM is lower in all 5 seeds in these cells | main, Sec. 4.6 |
| `sensitivity.range` | range of the error reduction in these cells (%) | main, Sec. 4.6 |

Supplement keys:

| Key | Meaning | Where the paper states it |
|---|---|---|
| `per_step.wins` | per-step comparisons in which the error of AFTER-WM is lower | supp., Sec. E |
| `per_step.ci_below_zero` | per-step comparisons whose 95% interval of the reduction lies above zero | supp., Sec. E |
| `phases.cells` | comparisons by change phase, dataset and compared model | supp., Sec. E.1 |
| `phases.afterwm_lower` | such comparisons in which the error of AFTER-WM is lower | supp., Sec. E.1 |
| `phases.ci_below_zero` | such comparisons whose 95% interval of the reduction lies above zero | supp., Sec. E.1 |
| `examples.candidates` | test queries with a change near the prediction start, per arm | supp., Sec. C.6 |
| `examples.median_rank` | rank of the example shown among these queries | supp., Sec. C.6 |
| `interventions.no_correction` | range of the error increase when both corrections are removed (%) | supp., Sec. E.3 |
| `interventions.zero_errors` | range of the error increase when the errors are set to zero (%) | supp., Sec. E.3 |
| `interventions.delay_4` | range of the error increase when the errors are delayed by 4 steps (%) | supp., Sec. E.3 |
| `probes.fast_r2` | range of the mean R² of the linear probes on the fast state | supp., Sec. E.3 |
| `probes.slow_r2` | range of the mean R² of the linear probes on the slow state | supp., Sec. E.3 |
| `probes.shuffled_below_zero` | whether the probes trained on shuffled labels score below zero | supp., Sec. E.3 |
| `probes.jshc_ba` | balanced accuracy on JSHC of the fast state, the slow state and the shuffled-label control | supp., Sec. E.3 |
| `removal.no_fast_step40_max` | largest change of the step-40 error without the fast correction (%) | supp., Sec. E.3 |
| `removal.no_slow_step1_max` | largest change of the step-1 error without the slow correction (%) | supp., Sec. E.3 |
| `removal.no_fast_early` | error increase without the fast correction 1–4 steps after a change, on Panda and UR5e (%) | supp., Sec. E.3 |
| `removal.no_fast_late` | error increase without the fast correction when the latest change is at least 60 steps old or absent, on Panda and UR5e (%) | supp., Sec. E.3 |
| `removal.no_slow_early_max` | largest error increase without the slow correction 1–4 steps after a change, on Panda and UR5e (%) | supp., Sec. E.3 |
| `removal.no_slow_late` | error increase without the slow correction when the latest change is at least 60 steps old or absent, on Panda and UR5e (%) | supp., Sec. E.3 |
| `crossing.step1` | range by which the slow-only model is worse than the fast-only model at step 1 (% of the error of AFTER-WM) | supp., Sec. E.3 |
| `crossing.step40_without_ur5e` | range by which the slow-only model is better than the fast-only model at step 40, on the datasets other than UR5e (% of the error of AFTER-WM) | supp., Sec. E.3 |
| `crossing.step40_ur5e` | the same difference on UR5e (% of the error of AFTER-WM) | supp., Sec. E.3 |
| `single.fast_step1_within` | largest difference between AFTER-WM and the fast-only model at step 1 (%) | supp., Sec. E.3 |
| `single.fast_step40` | range by which the fast-only model is worse than AFTER-WM at step 40 (%) | supp., Sec. E.3 |
| `single.slow_step40` | range by which the slow-only model is worse than AFTER-WM at step 40 (%) | supp., Sec. E.3 |
| `control.n_models` | models in the control task | supp., Sec. C.5 |

## Stored outputs

`outputs/MANIFEST.json` lists every file with its size and sha256 (`outputs/SHA256SUMS` in `sha256sum` format).

| Folder | Content | Built from (GPU) by |
|---|---|---|
| `outputs/per_query/` | per-query multi-step mean, step-1 and step-40 errors of every run (the Table 1 runs, the ablations, ALFA, the settings of Fig. 7b as `escphc_ctx32`, `escphc_ctx128` and the runs with a fixed γ inside `escphc`) | `pipeline.evaluate`, `afterwm_stats.compact per-query` |
| `outputs/per_step/` | per-step errors of the Table 1 runs and of the models retrained with one correction (float32) | `afterwm_stats.compact per-step` |
| `outputs/labels/` | change phases, queries without a change, queries with transient disturbances and queries beyond the training range | `datasets/labels.py` (CPU; also run by `experiments/regenerate_outputs.sh`) |
| `outputs/analysis/` | alterations of the errors (`outputs/analysis/interventions.npz`), removal of one correction (`outputs/analysis/removal.npz`), routing test without errors (`outputs/analysis/routing.npz`), re-scoring beyond the training range (`outputs/analysis/beyond_range.npz`), probe scores (`outputs/analysis/probes.json`) | `analysis.interventions`, `analysis.removal`, `analysis.routing`, `analysis.beyond_range`, `analysis.probes` |
| `outputs/control/` | returns of the eleven world models, 5 seeds each, on the 48 control episodes | `control.run`, `afterwm_stats.compact control` |
| `outputs/examples/` | the two robot-arm examples (link deviations, per-joint errors) and the stored frames | `analysis.examples_extract`, `analysis.examples_render` (not run by `experiments/regenerate_outputs.sh`); the stored frames are steps 1, 5, 10, 20, 40 |

`experiments/regenerate_outputs.sh` rebuilds these folders, except `outputs/examples/`, from a runs root (see `docs/REPRODUCE.md`).
