# Panda: robot arm with payload, joint-friction and motor-strength changes

Simulated dataset built by us (`datasets/panda/`, shared core `datasets/simcore/`).

## Content

- Robot: Franka Panda (MuJoCo Menagerie), commit 6da8b971, Apache-2.0; MuJoCo 3.11.0, frame skip 25, dt = 0.05 s; compliant joint-impedance servos, gravity compensation of the nominal arm, gripper fixed closed, a payload body at the tool point.
- Persistent changes: payload mass (0 or 0.5-2.5 kg), joint friction (x0.1-3) and motor strength (1 or x0.8-1.2).
- Transient disturbances: pulses of 4-12 steps (tool force 5-25 N, motor strength, or friction) and one-step impacts on the tool point. Transient disturbances are not changes.
- Actions: open-loop plans (posture offset plus one of four plan kinds); the actions do not depend on the state. The plan draw of a cohort is rejected and redrawn if any of its six sequences makes the arm touch itself or the floor, or reach a joint limit.
- Structure: parent (start state) -> 2 cohorts (action plans of two of the four kinds) -> 6 condition sequences (`null`, `persistent`, `switch`, `recur`, `recur_nominal`, `drift`) on the segments [0, 40, 120, 190, 260, 300]; the condition values of a parent are shared by its 12 sequences. In `drift` the values change at step 40 and then move on by one small step at each later segment boundary (slow drift).
- The ranges above are sampling ranges. In `drift` sequences the later steps can leave these ranges; the values are clipped at payload [0, 3] kg, motor strength [0.7, 1.3] and joint friction [0.1, 3]. The data contain payloads up to 3.0 kg.

## Splits and files

TRAIN 24 parents, VAL 12, TEST 12, TEST_B 12 (the paper's TE-B; second test split, seed offset 1); 12 sequences per parent. The paper uses TEST and TEST_B together as the test set (1,440 queries). Parent ids repeat between TEST and TEST_B, so a group key is (split, parent).

| file | sequences | bytes | sha256 |
|---|---|---|---|
| `data/panda/panda_train.npz` | 288 | 3,413,899 | `295a04c2c951717e9ea6934e6cd9b919aef333f097d1a9a3a53b7ef51dc5f730` |
| `data/panda/panda_val.npz` | 144 | 1,709,845 | `777d0f5a972cd54438d47516f86470e85b9f8d4604a18e2039aa2ba284537d35` |
| `data/panda/panda_test.npz` | 144 | 1,708,384 | `b0e8943d6f6ffd5d1b59f493cea24e210c513ca682e5433023ce515ecb9e8bdb` |
| `data/panda/panda_testB.npz` | 144 | 1,708,020 | `9096e158fd38abf2bb46e6252976f99c2e626c3f1b2574b568b7bfe9a9f8c7de` |

## Arrays (N sequences)

| key | dtype, shape | meaning |
|---|---|---|
| `obs` | float32 [N, 301, 14] | 7 joint angles, 7 joint velocities |
| `act` | float32 [N, 300, 7] | normalized joint-servo targets |
| `friction`, `gain`, `payload` | float32 [N, 300] | persistent values at each transition: joint-friction multiplier, motor-strength factor, payload mass (kg) |
| `friction_eff`, `gain_eff` | float32 [N, 300] | effective friction and motor strength including transient pulses |
| `wrench` | float32 [N, 300, 3] | tool-force pulse at the tool point (N, world frame; 0 when there is none) |
| `impulse` | float32 [N, 300, 3] | one-step impact force at the tool point (N; 0 when there is none) |
| `seg` | int8 [N, 300] | segment index (0-4) of the condition schedule |
| `sequence` | str [N] | condition sequence type: `null`, `persistent`, `switch`, `recur`, `recur_nominal`, `drift` |
| `parent`, `cohort`, `plan_attempt` | [N] | parent id, cohort (0, 1), accepted action-plan attempt |

## Generation

Environment as for JSHC (`environment/requirements-generate.txt`). Panda and UR5e need the MuJoCo Menagerie models at commit 6da8b971: `bash datasets/assets/fetch_menagerie.sh` places them in `datasets/panda_model/` and `datasets/ur5e_model/` and checks every file against the sha256 lists in `datasets/assets/`; the generators check them again at load time. `bash datasets/regenerate_all.sh OUT panda` runs the unit tests and (from `datasets/`)

```
python -B -m panda.cli collect --design panda --counts train=24,val=12,test=12 --seed-offset 0 --out OUT/work/panda/main
python -B -m panda.cli collect --design panda --counts test=12 --seed-offset 1 --out OUT/work/panda/testB
python -B -m panda.cli assemble --main OUT/work/panda/main --testb OUT/work/panda/testB --out OUT/panda
```

Seeds are derived from a fixed namespace string (the `namespace` entry of `PROTOCOL_PANDA` in `datasets/panda/gen.py`) and the position of the draw.
The namespace string is a fixed identifier (`docs/FIXED_IDENTIFIERS.md`); changing it changes every random draw. The protocol constants are in `datasets/panda/gen.py` (`PROTOCOL_PANDA`).

## Labels and licence

`datasets/labels.py` computes the change phases (from changes of `friction`, `gain` and `payload`; pulses and impacts do not count as changes), the queries without a change, and the queries with transient disturbances but no change: no change in the context or horizon, and at least one transition in the last 24 context steps or in the horizon inside a friction or motor-strength pulse (`friction_eff != friction` or `gain_eff != gain`) or carrying an impact (`impulse != 0`); tool-force pulses (`wrench`) are not part of this rule. Licence: CC BY 4.0 (`DATA_LICENSE.md`).
