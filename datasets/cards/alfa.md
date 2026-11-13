# ALFA: real fixed-wing flights with engine and control-surface failures

Sequences derived by us from the ALFA dataset (licence CC BY 4.0). The original flights are not redistributed here; `datasets/alfa/download_alfa.sh` downloads the archive processed.zip (271,996,798 bytes, sha256 `02562756b9428aa24fbd27105f91687b38541ced8d5432766c14176573d0b4c9`) and checks it. If the download fails, fetch the file manually from the data DOI below and place it in `datasets/alfa/` under the name processed.zip.

- Azarakhsh Keipour, Mohammadreza Mousaei and Sebastian Scherer. ALFA: A Dataset for UAV Fault and Anomaly Detection. The International Journal of Robotics Research 40(2-3), 2021. https://doi.org/10.1177/0278364920966642. Data: https://doi.org/10.1184/R1/12707963.v1 (file processed.zip).
- Azarakhsh Keipour, Mohammadreza Mousaei and Sebastian Scherer. Automatic Real-time Anomaly Detection for Autonomous Aerial Vehicles. IEEE International Conference on Robotics and Automation (ICRA), 2019. https://doi.org/10.1109/ICRA.2019.8794286. Part of the ALFA recordings was published with this paper, and the ALFA authors ask users of those recordings to cite it.

## Changes made to the original data

- Signals are read from the CSV files inside the archive (no extraction) and resampled to a 20 Hz grid (dt = 0.05 s) with plain double-precision arithmetic; values are converted to float32 once at the end.
- Sequences of 301 states are cut every 160 grid steps, starting at the earliest time at which all used signals are present; a sequence is used only if every signal is present without long gaps and the aircraft is armed and in an autonomous mode (GUIDED or AUTO). Failure labels and times are never used to place sequences; they are kept as analysis labels.
- Whole flights are assigned to splits by stratified quotas (engine / control surface / no failure / unlabelled) with a fixed seed (`datasets/alfa/QUOTAS.json`).
- In `datasets/alfa/alfa_windows.py` and in the array `window_k`, a window is one sequence.

## Splits and files

TRAIN 240 sequences (12 flights), VAL 86 (5), TEST 100 (7), TEST_B 122 (7); TEST_B is the paper's TE-B. Every sequence has 5 queries; the paper uses TEST and TEST_B together as the test set (1,110 queries).

| file | sequences | bytes | sha256 |
|---|---|---|---|
| `data/alfa/alfa_train.npz` | 240 | 4,643,110 | `5bd1b322332ffd2938de095637129bff1abb859d66c5cb35443bf37615299015` |
| `data/alfa/alfa_val.npz` | 86 | 1,665,828 | `262021333905658a79ac14999307fc37ddda738d0e8c58b9e8c65ec757706807` |
| `data/alfa/alfa_test.npz` | 100 | 1,936,490 | `ac34426b09c4bb37ea0b72c140eed3f9d8e734b35ccb687a894aca2bd7e626eb` |
| `data/alfa/alfa_testB.npz` | 122 | 2,361,816 | `bf884d364061e9dcf02cbc04ddd48e6870a8c826990bcb883cf028973bdbbcc5` |

## Arrays (N sequences)

| key | dtype, shape | meaning |
|---|---|---|
| `obs` | float32 [N, 301, 12] | `gbx`, `gby` (gravity direction in the body frame), `sin_yaw`, `cos_yaw` (heading), `p`, `q`, `r` (body rates), `vN`, `vE`, `vD` (NED velocity), `h` (altitude), `airspeed` |
| `act` | float32 [N, 300, 3] | `roll_cmd`, `pitch_cmd`, `airspeed_cmd` (roll, pitch and airspeed commands of the autopilot) |
| `parent`, `seq_name` | str [N] | flight (group) and name of the ALFA recording |
| `sequence`, `stratum`, `failure_tail` | str [N] | failure type of the recording (`engine`, `surface`, `none`, `unlabelled`), quota stratum of the flight, failure name |
| `fail_step`, `onset_index` | float64 / int64 [N] | failure onset relative to the start of the sequence in grid steps (NaN without a failure); index of the earliest state at or after the onset (-1 if the onset is not inside the sequence) |
| `seg` | int8 [N, 301] | 1 at states at or after the failure onset |
| `max_gap_s` | float64 [N, 7] | largest sample gap of each signal in the sequence |
| `tau0_ns`, `window_k`, `cohort` | int64 / int64 / str [N] | start time of the sequence, index of the sequence along the recording, `real` |

## Generation

Python 3.11 with numpy (no MuJoCo needed): `python -I datasets/alfa/alfa_windows.py define --zip datasets/alfa/processed.zip --quotas datasets/alfa/QUOTAS.json --out OUT/alfa` (also run by `bash datasets/regenerate_all.sh OUT alfa`).

## Licence and attribution

The ALFA sequences are a derivative of ALFA and are distributed under CC BY 4.0 (`DATA_LICENSE.md`). Please cite both papers above when you use them. No endorsement by the ALFA authors is implied.
