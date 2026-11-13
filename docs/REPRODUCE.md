# Reproducing the results

There are three modes. Each mode ends with the same step: the entry scripts in `experiments/` read an outputs folder and write `results/`. The table 'Paper items and their scripts' in `README.md` lists the paper items they cover.

| Mode | Needs | Time | Command |
|---|---|---|---|
| `--from-outputs` | CPU, `environment/requirements-figures.txt` | about 30 s | `bash experiments/all_from_outputs.sh` |
| `--from-checkpoints` | GPU, `environment/requirements-train.txt`, trained runs | hours | prepare the data, then `experiments/regenerate_outputs.sh` |
| `--from-scratch` | GPU, `environment/requirements-train.txt` | about 660 GPU-hours for the runs whose evaluations are stored in `outputs/`; stage 1 and the selection runs come on top | all steps below |

Run every command from the repository root. Scripts in `experiments/` set `PYTHONPATH` themselves; for the module calls below, first run `export PYTHONPATH=$PWD:$PWD/datasets`. Common options of every module: `--dataset` (`jshc`, `escphc`, `panda`, `ur5e`, `hopper`, `alfa`), `--data-root` (default `./data`), `--runs-root` (default `./runs`), `--arm` (settings of Fig. 7b); they can also be set with `AFTERWM_DATASET`, `AFTERWM_DATA_ROOT`, `AFTERWM_RUNS_ROOT` and `AFTERWM_ARM`.

Seeds: 20260910, 20270913, 20280916, 20290919, 20300922 (ALFA uses the 4 seeds other than 20280916 for all models; see the README).

The splits are TRAIN, VAL, TEST and TEST_B (the paper's TE-B).

## From the stored outputs

```
PYTHON=venv-figures/bin/python bash experiments/all_from_outputs.sh
```

Recomputes the results listed in the table 'Paper items and their scripts' from the stored outputs. For these results the stored outputs are the reference. The script writes `results/` (see `docs/RESULTS_MAP.md`). To run a single item, call its entry script, for example `bash experiments/table1_main.sh`.

## Preparing the data

The data files are in `data/`. To regenerate them instead, see `docs/DATASETS.md` (`bash datasets/regenerate_all.sh OUT_DIR`, generation environment). Then write the query manifests, training schedules and normalisation statistics of every dataset:

```
python -m pipeline.prepare --dataset DS --data-root data
python -m pipeline.prepare --dataset escphc --data-root data --arm c1ctx32
python -m pipeline.prepare --dataset escphc --data-root data --arm c1ctx128
```

`pipeline.prepare` never overwrites an existing file that differs; delete `data/prepared/` to rebuild it.

MTS3 needs the model code of the MTS3 authors: `bash baselines/third_party/mts3/fetch_and_patch.sh` (see `THIRD_PARTY_NOTICES.md`).

## Training from scratch

The configuration of every model on every dataset is in `configs/models/`, one file per dataset and model, for example `configs/models/jshc/afterwm.json` (module, budget class, learning-rate multiplier, protocol, variant, number of updates). The values recorded for the selections behind the stored outputs are in `configs/selection/`.

### Stage 1: pre-training the base (one base per seed, 5 per dataset; each base holds the router and K = 3 experts)

```
python -m afterwm.stage1 --dataset DS --data-root data --runs-root runs --seed S --device cuda --threads 1
python -m selection.pins --dataset DS --runs-root runs --write
```

On ALFA, no epoch of the base of seed 20280916 becomes a candidate (see the README); pass `--abandoned 20280916` to the second command so that this seed is recorded as discarded. `selection.pins` replays the epoch rule on the training history of every base and writes `runs/bases/DS/PINS.json` (recorded values: `configs/selection/pins.json`).

### Inverse routing temperature γ (chosen on validation data)

```
python -m selection.gamma --dataset DS --data-root data --runs-root runs --device cuda --write
```

Writes `runs/gamma/DS/GAMMA.json` (recorded values: `configs/selection/gamma.json`).

### Stage 2 of AFTER-WM and the ablations

```
python -m pipeline.trainer --dataset DS --data-root data --runs-root runs --module afterwm.model --model-id afterwm \
  --seed S --budget-class afterwm_eq --variant router_gamma=pinned --lr-mult 10 --protocol p2 --device cuda
```

Ablations (Fig. 7a): the same command with `--module afterwm.variants.fast_only --model-id afterwm_fast_only`, `--module afterwm.variants.slow_only --model-id afterwm_slow_only` or `--module afterwm.variants.no_error_input --model-id afterwm_no_error_input`; the model with γ fixed to 1 uses `--module afterwm.model --model-id afterwm_gamma1` without `--variant`. The models without the slow correction and without the fast correction are also the models retrained with one correction of Fig. 6c.

### The ten compared models

```
python -m pipeline.trainer --dataset DS --data-root data --runs-root runs --module baselines.NAME --model-id NAME \
  --seed S --budget-class baseline --lr-mult M --protocol p2 --device cuda [--variant KEY=VALUE ...]
```

`NAME` is one of `rssm`, `tdmpc2`, `tdmpc2_h40`, `mlp`, `grbal`, `rebal`, `mole`, `mbcd`, `gpmm`, `mts3`. Take `M` and the variant from the configuration file of the model (`lr_mult`, `variant`; the MLP uses `history_k=1`, GrBAL `inner_lr=<chosen step>`). GPMM is built in one pass with `--protocol p1` and no `--lr-mult`.

Selections made on validation data with the seeds 20260910 and 20270913:

- GrBAL's test-time step size: train GrBAL with the steps 0.001 and 0.01 (`--variant inner_lr=STEP`, output `--out runs/runs/DS/grbal_step/inner_lr_STEP/S`), then run `python -m selection.grbal_step stage1 --dataset DS --runs-root runs`; if it asks for an extension step (0.1 or 0.0001), train that step as well, then run `python -m selection.grbal_step final --dataset DS --runs-root runs` (recorded values: `configs/selection/grbal_step.json`).
- RSSM evaluation mode: evaluate the RSSM runs on the validation split with both modes (`python -m pipeline.evaluate ... --splits val --rssm-eval-mode MODE --out-dir runs/runs/DS/rssm_mode/MODE/S`), then `python -m selection.rssm_mode --dataset DS --runs-root runs` (recorded: `sample16` on all datasets, `configs/selection/rssm_mode.json`).

The learning-rate multiplier of each model was selected once, on a HalfCheetah development dataset that is not used for evaluation, and then kept for all datasets (`configs/selection/lr.json`).

### Other context lengths and γ values of Fig. 7b (ESCP-HC)

Add `--arm ARM` (a named sensitivity setting, not a robot arm) to the trainer and evaluation commands; runs go to `runs/arms/ARM/runs/`. The settings are listed in `configs/arms.json`; the code reads them from `ARMS` in `pipeline/registry.py`.

- `c1ctx32`, `c1ctx128` (context 32 or 128, own prepared files): AFTER-WM and the five compared models of Fig. 7b (`rebal`, `mts3`, `mlp`, `tdmpc2`, `tdmpc2_h40`).
- `c1g005`, `c1g05` (γ fixed to 0.05 or 0.5): AFTER-WM only, same command as the main AFTER-WM run.

The default and γ = 1 cells of Fig. 7b are the main runs of ESCP-HC.

## Evaluation and regenerated outputs

Evaluate one run (VAL, TEST and TEST_B; the results go to the subfolder eval of the run folder):

```
python -m pipeline.evaluate --dataset DS --data-root data --runs-root runs --run runs/runs/DS/MODEL/S --device cuda
```

RSSM runs are evaluated with `--rssm-eval-mode sample16`. `--verify-ckpt-sha` checks the checkpoint against its training record.

Rebuild the outputs from a runs root (evaluation of every run with a `best.pt`, the GPU analyses of AFTER-WM, the re-scoring beyond the training range, the control runs and the labels), then recompute the results from them:

```
AFTERWM_RUNS_ROOT=runs bash experiments/regenerate_outputs.sh
AFTERWM_OUTPUTS=results/regenerated/outputs bash experiments/all_from_outputs.sh
```

`experiments/regenerate_outputs.sh` writes to `results/regenerated/outputs` (another folder: `AFTERWM_REGENERATED_OUTPUTS`), runs the GPU steps on `cuda` (another device: `AFTERWM_DEVICE`) and reads the control episodes from `data/control_episodes/episodes_test_v1.npz` (another file: `AFTERWM_CONTROL_SPEC`).

`experiments/regenerate_outputs.sh` does not rebuild `outputs/examples/`. Copy `outputs/examples/` into `results/regenerated/outputs/`, or rebuild the two robot-arm examples as described below.

Results computed on a different GPU type can differ slightly from the stored outputs; see the Notes in the README.

## Robot-arm renders (Fig. 4)

`analysis/examples_select.py` applies the fixed selection rule, `analysis/examples_extract.py` extracts the selected examples (true sequence and predictions) from the evaluated runs, and `analysis/examples_render.py` renders them with MuJoCo (OpenGL needed; the Menagerie models must be installed with `bash datasets/assets/fetch_menagerie.sh`). `figures/crop_frames.py` and `figures/fig4_robot_arm.py` assemble the figure; the stored frames (steps 1, 5, 10, 20, 40; the paper shows steps 5 to 40) are in `outputs/examples/frames/`. Each of these modules lists its options with `--help`.
