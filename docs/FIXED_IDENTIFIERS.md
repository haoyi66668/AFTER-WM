# Fixed identifiers

Some string constants in this repository are fixed. They feed random seeds, file-format checks or pinned hashes, so editing them would change the initial weights, the generated data or the pinned files.

| Constant | File | What it feeds |
|---|---|---|
| `CONSTRUCTOR_STREAM` | `afterwm/legacy.py`; used in `afterwm/model.py` and `afterwm/variants/slow_only.py` | seed of the constructor of the memory networks of AFTER-WM, of the slow-only model, of the model without error inputs and of the model with γ fixed to 1 |
| `SCHEMA` | `afterwm/lib/memory/budget.py` | first element of every hashed seed list (`seed_for`) |
| `NAME` | `afterwm/variants/fast_only.py` | passed to the seed derivation of the constructor of the fast-only model |
| the seed groups `architecture-constructor`, `Uq`, `UF`, `GF`, `F_A`, the local names `<k>.joint-output`, and the attribute names of all network modules | `afterwm/lib/`, `afterwm/core.py`, `afterwm/model.py`, `afterwm/variants/`, `baselines/` | seeds of the per-module initialization, and the keys of the stored weights |
| the `namespace` entry of `PROTOCOL_PANDA`, `PROTOCOL_UR5E` and `PROTOCOL_HOPPER` | `datasets/panda/gen.py`, `datasets/ur5e/gen.py`, `datasets/hopper/gen.py` (and their tests) | every random draw of the Panda, UR5e and Hopper data |
| `NAMESPACE` (JSHC, ESCP-HC) and `NS` (control episodes) | `datasets/jshc/config.py`, `datasets/escphc/gen.py`, `control/spec.py` | every random draw of the JSHC and ESCP-HC data and of the control episodes |
| `STAGE1_CHECKPOINT_FORMAT` | `afterwm/legacy.py` | format name stored inside every `base_stage1a.pt`; the readers check it; the sha256 of the file is pinned |
| `STAGE1_SOURCE` | `afterwm/legacy.py` | stored inside every `base_stage1a.pt` (part of the pinned file bytes) |
| the dataset schema strings (`jshc_v1`, `escphc_v1`, `panda_v1`, `ur5e_v1`, `hopper_v1`, `mpcjshc_episodes_v1`), `alfa_def v1 (2026-10-21)`, and the seed constants (`BOOT_SEED`, `SIGMA_FIT_SEED`, `NOISE_SEED`, `HOLDOUT_SEED`, `PERM_SEED`, `SHUF_SEED`, the five run seeds) | `datasets/`, and the modules that define the constants | data arrays and random draws of the evaluation |
| `mpcjshc-cem` | `control/planner.py` | seed of the planner noise in the control task |

Not fixed: `protocol_sha256` and the `schema` labels of run records. `protocol_sha256` includes the sha256 of every imported code file, so it changes with any code edit; it is compared only when an unfinished run is resumed.
