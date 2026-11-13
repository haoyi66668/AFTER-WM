# Hopper: one-legged hopper with ground-friction, motor-strength and torso-mass changes

Simulated dataset built by us (`datasets/hopper/`, shared core `datasets/simcore/`).

## Content

- Robot: the Gymnasium 1.3.0 `Hopper-v5` model (the model file is read from the installed Gymnasium package and checked by sha256; the file is not modified, the changes below are applied at load time), MuJoCo 3.11.0, frame skip 25, dt = 0.05 s; the joints are driven by position servos with the original torque limit.
- Persistent changes: ground friction (x0.1-3), motor strength (1 or x0.6-1.4) and torso mass (1 or x1.3-2.5).
- Transient disturbances: pulses of 4-12 steps (motor strength or friction) and one-step impacts on the planar root (x, z, pitch). Transient disturbances are not changes.
- Start states: reached after a seeded roll-in, with a contact-state quota; actions are open-loop servo targets around a home posture; they do not depend on the state. The plan draw of a cohort is rejected and redrawn if one of its sequences fails the health check of the generator (`health_check` in `datasets/hopper/gen.py`).
- Structure: parent (start state) -> 2 cohorts (action plans of two of the four kinds) -> 6 condition sequences (`null`, `persistent`, `switch`, `recur`, `recur_nominal`, `drift`) on the segments [0, 40, 120, 190, 260, 300]; the condition values of a parent are shared by its 12 sequences. In `drift` the values change at step 40 and then move on by one small step at each later segment boundary (slow drift).
- The ranges above are sampling ranges. In `drift` sequences the later steps can leave these ranges; the values are clipped at motor strength [0.5, 1.5], torso mass [1, 3] and ground friction [0.1, 3]. The data contain a motor strength up to 1.5 and a torso mass up to x3.0.

## Splits and files

TRAIN 24 parents, VAL 12, TEST 12, TEST_B 12 (the paper's TE-B; second test split, seed offset 1); 12 sequences per parent. The paper uses TEST and TEST_B together as the test set (1,440 queries). Parent ids repeat between TEST and TEST_B, so a group key is (split, parent).

| file | sequences | bytes | sha256 |
|---|---|---|---|
| `data/hopper/hopper_train.npz` | 288 | 2,575,773 | `a692c61f5cbc320178fa5b57eabb830d6f7a331a0c534bebcb94236c9b948dd3` |
| `data/hopper/hopper_val.npz` | 144 | 1,287,591 | `7826c47a7879981ba84d36ae53c58cbdffeb93b1a87c42ea7d57fd412273758f` |
| `data/hopper/hopper_test.npz` | 144 | 1,279,157 | `489862fd07ff0bc2f2ddf0ee4cb204568610ff4e795452c48593242b1a828609` |
| `data/hopper/hopper_testB.npz` | 144 | 1,280,122 | `3d9c2c36ed8e36ed0aad86df52c422222db74e264a11476cc13243596f8aeee3` |

## Arrays (N sequences)

| key | dtype, shape | meaning |
|---|---|---|
| `obs` | float32 [N, 301, 11] | torso height, pitch, 3 joint angles, 6 velocities |
| `act` | float32 [N, 300, 3] | joint-servo targets |
| `friction`, `gain`, `payload` | float32 [N, 300] | persistent values at each transition: ground-friction multiplier, motor-strength factor, torso-mass factor (the array `payload` holds the torso-mass factor) |
| `friction_eff`, `gain_eff` | float32 [N, 300] | effective friction and motor strength including transient pulses |
| `impulse` | float32 [N, 300, 3] | one-step impact on the planar root dofs x, z, pitch (0 when there is none) |
| `seg` | int8 [N, 300] | segment index (0-4) of the condition schedule |
| `sequence` | str [N] | condition sequence type: `null`, `persistent`, `switch`, `recur`, `recur_nominal`, `drift` |
| `parent`, `cohort`, `plan_attempt` | [N] | parent id, cohort (0, 1), accepted action-plan attempt |

## Generation

Environment as for JSHC (`environment/requirements-generate.txt`). `bash datasets/regenerate_all.sh OUT hopper` runs the unit tests and (from `datasets/`)

```
python -B -m hopper.cli collect --design hopper --counts train=24,val=12,test=12 --seed-offset 0 --out OUT/work/hopper/main
python -B -m hopper.cli collect --design hopper --counts test=12 --seed-offset 1 --out OUT/work/hopper/testB
python -B -m hopper.cli assemble --main OUT/work/hopper/main --testb OUT/work/hopper/testB --out OUT/hopper
```

Seeds are derived from a fixed namespace string (the `namespace` entry of `PROTOCOL_HOPPER` in `datasets/hopper/gen.py`) and the position of the draw.
The namespace string is a fixed identifier (`docs/FIXED_IDENTIFIERS.md`); changing it changes every random draw. The protocol constants are in `datasets/hopper/gen.py` (`PROTOCOL_HOPPER`).

## Labels and licence

`datasets/labels.py` computes the change phases (from changes of `friction`, `gain` and `payload`; pulses and impacts do not count as changes), the queries without a change, and the queries with transient disturbances but no change: no change in the context or horizon, and at least one transition in the last 24 context steps or in the horizon inside a friction or motor-strength pulse (`friction_eff != friction` or `gain_eff != gain`) or carrying an impact (`impulse != 0`). Licence: CC BY 4.0 (`DATA_LICENSE.md`).
