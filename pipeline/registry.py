import argparse
import os
import sys

CLI_OPTIONS = ("--dataset", "--data-root", "--runs-root", "--arm")


def _settings():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for opt in CLI_OPTIONS:
        ap.add_argument(opt, default=None)
    known, _ = ap.parse_known_args([a for a in sys.argv[1:] if a != "-h" and a != "--help"])
    return {"dataset": known.dataset or os.environ.get("AFTERWM_DATASET") or "jshc",
            "data_root": known.data_root or os.environ.get("AFTERWM_DATA_ROOT") or "data",
            "runs_root": known.runs_root or os.environ.get("AFTERWM_RUNS_ROOT") or "runs",
            "arm": known.arm or os.environ.get("AFTERWM_ARM") or None}


def add_cli_options(ap):
    g = ap.add_argument_group("dataset and storage (read when the package is imported)")
    g.add_argument("--dataset", default=None, help="dataset name (default: $AFTERWM_DATASET or jshc)")
    g.add_argument("--data-root", default=None, help="folder with <dataset>/ and prepared/ (default: $AFTERWM_DATA_ROOT or ./data)")
    g.add_argument("--runs-root", default=None, help="folder with bases/, gamma/, runs/, arms/ (default: $AFTERWM_RUNS_ROOT or ./runs)")
    g.add_argument("--arm", default=None, help="sensitivity setting of Fig. 7b (c1ctx32, c1ctx128, c1g005, c1g05)")
    return ap


SETTINGS = _settings()
DATA_ROOT = os.path.abspath(SETTINGS["data_root"])
RUNS_ROOT = os.path.abspath(SETTINGS["runs_root"])

DATASETS = {
    "jshc": {
        "files": {"train": "jshc_train.npz", "val": "jshc_val.npz", "test": "jshc_test.npz", "test_b": "jshc_testB.npz"},
        "sha256": {
            "train": "afeb1fd6f8147d27a79a1afdb8e1c23313d336a2d61db0895d02932663e45679",
            "val": "df1d5f4410229c278393257e105834e17bafbcf757cc1224dea10d1475ab09dc",
            "test": "a0ab8922fb65b83b59475e1689db94bcca22092d9b70bbd88609d17bdff14d26",
            "test_b": "63f764412fc479c994a81728359d2673b2ffdd5f5bdd360abbb3444ea55f5f46",
        },
        "obs_dim": 17, "act_dim": 6,
    },
    "escphc": {
        "files": {"train": "escphc_train.npz", "val": "escphc_val.npz", "test": "escphc_test.npz",
                  "test_b": "escphc_testB.npz"},
        "sha256": {
            "train": "6c49f9ad7c84c65c63ae18ef1e1a92cc0418ac9962df9aea36f3b27adbbf80e9",
            "val": "81680c798e567d37eb9d72839cb7949ac741b766644c3e571b357cc0be8c57d2",
            "test": "1871ac564e79098f1d473c7a051626fcea7a49474f908f7b858468793e6f1b2f",
            "test_b": "e89f69ae256c5fac7cf206b28eaae03b3bd63736179befa5416815987d93a2b0",
        },
        "obs_dim": 17, "act_dim": 6,
        "extra_splits": {
            "beyond_test": {"dir": "escphc_beyond", "file": "escphc_test.npz", "tag": "TEST_BEYOND",
                            "sha256": "e766b990411c88b7cca9d720fa5dbe7adb7ab8e82f68974f539d310ca2707e09"},
            "beyond_test_b": {"dir": "escphc_beyond", "file": "escphc_testB.npz", "tag": "TEST_B_BEYOND",
                              "sha256": "2d50897ac0e77b68d95c20c975b622b4ee0f8b78da08c3907e24721a32ff5a25"},
        },
    },
    "panda": {
        "files": {"train": "panda_train.npz", "val": "panda_val.npz", "test": "panda_test.npz",
                  "test_b": "panda_testB.npz"},
        "sha256": {
            "train": "295a04c2c951717e9ea6934e6cd9b919aef333f097d1a9a3a53b7ef51dc5f730",
            "val": "777d0f5a972cd54438d47516f86470e85b9f8d4604a18e2039aa2ba284537d35",
            "test": "b0e8943d6f6ffd5d1b59f493cea24e210c513ca682e5433023ce515ecb9e8bdb",
            "test_b": "9096e158fd38abf2bb46e6252976f99c2e626c3f1b2574b568b7bfe9a9f8c7de",
        },
        "obs_dim": 14, "act_dim": 7,
    },
    "ur5e": {
        "files": {"train": "ur5e_train.npz", "val": "ur5e_val.npz", "test": "ur5e_test.npz", "test_b": "ur5e_testB.npz"},
        "sha256": {
            "train": "ba258808781e059f5a576b97c83c7f7a288f3febd6e496d5f3a70a8acb7f7879",
            "val": "38502b7563059c75b588c1c38a35bf8e3ce6c72af4d9f4c4879692ffa5e9b9cc",
            "test": "cabee1d1ff6bd77888d185296163c50b99cc2a33450937d5e0f55817963a8e27",
            "test_b": "65d3aa6e3fc6493a984fb9d158487bdb095dba7751376ddfe04e082570adb4ef",
        },
        "obs_dim": 12, "act_dim": 6,
    },
    "hopper": {
        "files": {"train": "hopper_train.npz", "val": "hopper_val.npz", "test": "hopper_test.npz",
                  "test_b": "hopper_testB.npz"},
        "sha256": {
            "train": "a692c61f5cbc320178fa5b57eabb830d6f7a331a0c534bebcb94236c9b948dd3",
            "val": "7826c47a7879981ba84d36ae53c58cbdffeb93b1a87c42ea7d57fd412273758f",
            "test": "489862fd07ff0bc2f2ddf0ee4cb204568610ff4e795452c48593242b1a828609",
            "test_b": "3d9c2c36ed8e36ed0aad86df52c422222db74e264a11476cc13243596f8aeee3",
        },
        "obs_dim": 11, "act_dim": 3,
    },
    "alfa": {
        "files": {"train": "alfa_train.npz", "val": "alfa_val.npz", "test": "alfa_test.npz", "test_b": "alfa_testB.npz"},
        "sha256": {
            "train": "5bd1b322332ffd2938de095637129bff1abb859d66c5cb35443bf37615299015",
            "val": "262021333905658a79ac14999307fc37ddda738d0e8c58b9e8c65ec757706807",
            "test": "ac34426b09c4bb37ea0b72c140eed3f9d8e734b35ccb687a894aca2bd7e626eb",
            "test_b": "bf884d364061e9dcf02cbc04ddd48e6870a8c826990bcb883cf028973bdbbcc5",
        },
        "obs_dim": 12, "act_dim": 3,
    },
}
ARMS = {
    "c1ctx32": {"dataset": "escphc", "context": 32, "eval_starts": [64, 96, 128, 160, 192], "train_t_min": 64,
                "own_prep": True},
    "c1ctx128": {"dataset": "escphc", "context": 128, "eval_starts": [128, 160, 192], "train_t_min": 128,
                 "own_prep": True},
    "c1g005": {"dataset": "escphc", "gamma_fixed": 0.05},
    "c1g05": {"dataset": "escphc", "gamma_fixed": 0.5},
}

DATASET = SETTINGS["dataset"]
if DATASET not in DATASETS:
    raise ValueError(f"dataset {DATASET!r} is not one of {sorted(DATASETS)}")
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
PREP_DIR = os.path.join(DATA_ROOT, "prepared", DATASET)
BASES_DIR = os.path.join(RUNS_ROOT, "bases", DATASET)
RUNS_DIR = os.path.join(RUNS_ROOT, "runs", DATASET)
GAMMA_DIR = os.path.join(RUNS_ROOT, "gamma", DATASET)
SPLIT_SHA256 = dict(DATASETS[DATASET]["sha256"])
SPLIT_FILES = dict(DATASETS[DATASET]["files"])
VISIBLE_KEYS = ("obs", "act")
ANALYSIS_KEYS = ("parent", "cohort", "sequence")

OBS_DIM = int(DATASETS[DATASET]["obs_dim"])
ACT_DIM = int(DATASETS[DATASET]["act_dim"])
T_OBS = 301
T_ACT = 300

CONTEXT = 64
MAX_H = 40
HORIZONS = (1, 5, 10, 20, 40)
EVAL_STARTS = (64, 96, 128, 160, 192)
TRAIN_T_MIN = 64
TRAIN_T_MAX = 260

SEEDS = (20260910, 20270913, 20280916, 20290919, 20300922)
BATCH = 32
MAX_STEPS = 4000
VAL_EVERY = 250
CKPT_EVERY = 100
VAL_SELECTION_METRIC = "mean_prefix_nMSE_H1_5_10_20_40"

BUDGET_BASELINE = 700_000
BUDGET_AFTERWM_EQ = 700_000
BUDGET_FAST_NATIVE = 594_742
BUDGET_TOL = 0.01

ACTIVE_EXTRA_SPLITS = {
    s: {"path": os.path.join(DATA_ROOT, e["dir"], e["file"]), "tag": e["tag"], "sha256": e["sha256"]}
    for s, e in DATASETS[DATASET].get("extra_splits", {}).items()}
EXTRA_PREP_DIR = None
SPLIT_GROUPS = {}

ARM = SETTINGS["arm"]
ARM_SETTINGS = None
ARM_GAMMA = None
if ARM is not None:
    if ARM not in ARMS:
        raise ValueError(f"arm {ARM!r} is not one of {sorted(ARMS)}")
    ARM_SETTINGS = dict(ARMS[ARM], arm=ARM)
    if DATASET != ARM_SETTINGS["dataset"]:
        raise ValueError(f"arm {ARM} needs --dataset {ARM_SETTINGS['dataset']}, got {DATASET!r}")
    _arm_dir = os.path.join(RUNS_ROOT, "arms", ARM)
    if ARM_SETTINGS.get("own_prep"):
        PREP_DIR = os.path.join(DATA_ROOT, "prepared", f"{DATASET}_{ARM}")
    RUNS_DIR = os.path.join(_arm_dir, "runs")
    if ARM_SETTINGS.get("context") is not None:
        CONTEXT = int(ARM_SETTINGS["context"])
        EVAL_STARTS = tuple(int(x) for x in ARM_SETTINGS["eval_starts"])
        TRAIN_T_MIN = int(ARM_SETTINGS["train_t_min"])
        if not (min(EVAL_STARTS) >= CONTEXT and TRAIN_T_MIN >= CONTEXT and max(EVAL_STARTS) + MAX_H <= T_OBS - 1
                and TRAIN_T_MIN <= TRAIN_T_MAX):
            raise ValueError(f"arm {ARM}: inconsistent geometry")
    if ARM_SETTINGS.get("gamma_fixed") is not None:
        ARM_GAMMA = float(ARM_SETTINGS["gamma_fixed"])
