# UR5e: robot arm with payload, joint-friction and motor-strength changes

Simulated dataset built by us (`datasets/ur5e/`, shared core `datasets/simcore/`); the generator follows the Panda generator line by line with the UR5e plant.

## Content

- Robot: UR5e (MuJoCo Menagerie), commit 6da8b971, BSD-3-Clause; MuJoCo 3.11.0, dt = 0.05 s; compliant joint-impedance servos, gravity compensation of the nominal arm, a payload body at the flange.
- Persistent changes: payload mass (0 or 1-4 kg), joint friction (x0.1-3) and motor strength (1 or x0.8-1.2).
- Transient disturbances: pulses of 4-12 steps (tool force 5-25 N, motor strength, or friction) and one-step impacts at the flange. Transient disturbances are not changes.
- Actions: open-loop plans; the actions do not depend on the state. The plan draw of a cohort is rejected and redrawn if any of its six sequences makes the arm touch itself or the floor, or reach a joint limit.
- Structure: parent (start state) -> 2 cohorts (action plans of two of the four kinds) -> 6 condition sequences (`null`, `persistent`, `switch`, `recur`, `recur_nominal`, `drift`) on the segments [0, 40, 120, 190, 260, 300]; the condition values of a parent are shared by its 12 sequences. In `drift` the values change at step 40 and then move on by one small step at each later segment boundary (slow drift).
- The ranges above are sampling ranges. In `drift` sequences the later steps can leave these ranges; the values are clipped at payload [0, 5] kg, motor strength [0.7, 1.3] and joint friction [0.1, 3]. The data contain payloads up to 5.0 kg.

## Splits and files

TRAIN 24 parents, VAL 12, TEST 12, TEST_B 12 (the paper's TE-B; second test split, seed offset 1); 12 sequences per parent. The paper uses TEST and TEST_B together as the test set (1,440 queries). Parent ids repeat between TEST and TEST_B, so a group key is (split, parent).

| file | sequences | bytes | sha256 |
|---|---|---|---|
| `data/ur5e/ur5e_train.npz` | 288 | 2,896,872 | `ba258808781e059f5a576b97c83c7f7a288f3febd6e496d5f3a70a8acb7f7879` |
| `data/ur5e/ur5e_val.npz` | 144 | 1,449,098 | `38502b7563059c75b588c1c38a35bf8e3ce6c72af4d9f4c4879692ffa5e9b9cc` |
| `data/ur5e/ur5e_test.npz` | 144 | 1,449,061 | `cabee1d1ff6bd77888d185296163c50b99cc2a33450937d5e0f55817963a8e27` |
| `data/ur5e/ur5e_testB.npz` | 144 | 1,449,089 | `65d3aa6e3fc6493a984fb9d158487bdb095dba7751376ddfe04e082570adb4ef` |

## Arrays (N sequences)

| key | dtype, shape | meaning |
|---|---|---|
| `obs` | float32 [N, 301, 12] | 6 joint angles, 6 joint velocities |
| `act` | float32 [N, 300, 6] | normalized joint-servo targets |
| `friction`, `gain`, `payload` | float32 [N, 300] | persistent values at each transition: joint-friction multiplier, motor-strength factor, payload mass (kg) |
| `friction_eff`, `gain_eff` | float32 [N, 300] | effective friction and motor strength including transient pulses |
| `wrench` | float32 [N, 300, 3] | tool-force pulse at the flange (N, world frame; 0 when there is none) |
| `impulse` | float32 [N, 300, 3] | one-step impact force at the flange (N; 0 when there is none) |
| `seg` | int8 [N, 300] | segment index (0-4) of the condition schedule |
| `sequence` | str [N] | condition sequence type: `null`, `persistent`, `switch`, `recur`, `recur_nominal`, `drift` |
| `parent`, `cohort`, `plan_attempt` | [N] | parent id, cohort (0, 1), accepted action-plan attempt |

## Generation

Environment as for JSHC (`environment/requirements-generate.txt`). Panda and UR5e need the MuJoCo Menagerie models at commit 6da8b971: `bash datasets/assets/fetch_menagerie.sh` places them in `datasets/panda_model/` and `datasets/ur5e_model/` and checks every file against the sha256 lists in `datasets/assets/`; the generators check them again at load time. `bash datasets/regenerate_all.sh OUT ur5e` runs the unit tests and (from `datasets/`)

```
python -B -m ur5e.cli collect --design ur5e --counts train=24,val=12,test=12 --seed-offset 0 --out OUT/work/ur5e/main
python -B -m ur5e.cli collect --design ur5e --counts test=12 --seed-offset 1 --out OUT/work/ur5e/testB
python -B -m ur5e.cli assemble --main OUT/work/ur5e/main --testb OUT/work/ur5e/testB --out OUT/ur5e
```

Seeds are derived from a fixed namespace string (the `namespace` entry of `PROTOCOL_UR5E` in `datasets/ur5e/gen.py`) and the position of the draw.
The namespace string is a fixed identifier (`docs/FIXED_IDENTIFIERS.md`); changing it changes every random draw. The protocol constants are in `datasets/ur5e/gen.py` (`PROTOCOL_UR5E`).

## Labels and licence

`datasets/labels.py` computes the change phases (from changes of `friction`, `gain` and `payload`; pulses and impacts do not count as changes), the queries without a change, and the queries with transient disturbances but no change: no change in the context or horizon, and at least one transition in the last 24 context steps or in the horizon inside a friction or motor-strength pulse (`friction_eff != friction` or `gain_eff != gain`) or carrying an impact (`impulse != 0`); tool-force pulses (`wrench`) are not part of this rule. Licence: CC BY 4.0 (`DATA_LICENSE.md`).
