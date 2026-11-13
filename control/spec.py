SEEDS = (20260910, 20270913, 20280916, 20290919, 20300922)
DATASET = "jshc"
MODEL_ROWS = {"afterwm": "afterwm",
              "tdmpc2": "tdmpc2", "tdmpc2_h40": "tdmpc2_h40", "rebal": "rebal", "grbal": "grbal", "mole": "mole",
              "mbcd": "mbcd", "mts3": "mts3", "rssm": "rssm", "gpmm": "gpmm",
              "mlp": "mlp"}
RSSM_EVAL_MODE = {"rssm": "sample16"}
EXTERNALS = ("tdmpc2", "tdmpc2_h40", "rebal", "grbal", "mole", "mbcd", "mts3", "rssm", "gpmm")
GENERIC = ("mlp",)
COMPARATORS = EXTERNALS + GENERIC
LABEL = {"afterwm": "AFTER-WM", "tdmpc2": "TD-MPC2", "tdmpc2_h40": "TD-MPC2-40", "rebal": "ReBAL", "grbal": "GrBAL",
         "mole": "MOLe", "mbcd": "MBCD", "mts3": "MTS3", "rssm": "RSSM", "gpmm": "GPMM", "mlp": "MLP"}
NS = {"test": "Code5.JSHC-MPC-v1"}
DATA_NS = ("Code5.JSHC-v1",)
SEED_OFFSET = {"test": 21}
N_EPISODES = {"test": 48}
T_WARM = 64
K_MPC = 240
K_SWITCH = (40, 140)
CANDIDATE_JOINTS = (1, 2, 3, 4, 5)
VEL_INDEX = 8
CTRL_COST = 0.05
BOX = 0.95
