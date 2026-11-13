from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from . import escp as E

SCHEMA = "escphc_v1"
NAMESPACE = "Code5.ESCP-HalfCheetah-v1"
ENV_ID = "HalfCheetah-v5"
T = 300
OBS_DIM, ACT_DIM = 17, 6
PLAN_LEN = 400
ACTION_LIMIT = 0.95
ACTION_KINDS = ("smooth_ar1", "piecewise", "periodic", "iid")
LOW_AMP_WINDOWS = ((110, 140), (230, 255))
LOW_AMP_MULT = 0.15
OFFSET_MAX_EXCL = 100
EVAL_STARTS = (64, 96, 128, 160, 192)
HORIZON = 40
PARENT_BLOCK = 12
SPLITS = {"train": (288, "train"), "val": (144, "test"), "test": (144, "test"), "testB": (144, "test")}
SPLIT_ORDER = ("train", "val", "test", "testB")
FILE_OF = {"train": "escphc_train.npz", "val": "escphc_val.npz", "test": "escphc_test.npz", "testB": "escphc_testB.npz"}
VISIBLE = ("obs", "act")
WARN_NAMES = ("mjWARN_INERTIA", "mjWARN_CONTACTFULL", "mjWARN_CNSTRFULL", "mjWARN_BADQPOS", "mjWARN_BADQVEL",
              "mjWARN_BADQACC", "mjWARN_BADCTRL")


_LOG_SCALE_LIMIT = [float(E.LOG_SCALE_LIMIT)]


def set_log_scale_limit(L: float) -> None:
    L = float(L)
    if not (0.0 < L <= 4.0):
        raise ValueError(f"log_scale_limit {L} outside (0, 4]")
    _LOG_SCALE_LIMIT[0] = L


def log_scale_limit() -> float:
    return _LOG_SCALE_LIMIT[0]


def derive_seed(namespace, *parts):
    payload = json.dumps([namespace, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def protocol(namespace: str = NAMESPACE) -> dict:
    return {"schema": SCHEMA, "namespace": namespace, "env_id": ENV_ID, "T": T, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
            "escp": {"log_scale_limit": _LOG_SCALE_LIMIT[0], "period": E.NON_STATIONARY_PERIOD,
                     "interval": E.NON_STATIONARY_INTERVAL, "task_num": E.TASK_NUM, "test_task_num": E.TEST_TASK_NUM,
                     "list_len": E.LIST_LEN, "rand_params": list(E.RAND_PARAMS), "torso_body": E.TORSO_BODY,
                     "apply": "absolute writes from stored originals, no mj_setConst (ESCP/DORA set_task)"},
            "pool_seed_parts": ["pools"], "splits": {k: list(v) for k, v in SPLITS.items()},
            "offset_max_exclusive": OFFSET_MAX_EXCL, "plan_len": PLAN_LEN, "action_limit": ACTION_LIMIT,
            "action_kinds": list(ACTION_KINDS), "low_amplitude_windows": [list(w) for w in LOW_AMP_WINDOWS],
            "low_amplitude_multiplier": LOW_AMP_MULT, "eval_starts": list(EVAL_STARTS), "horizon": HORIZON,
            "parent_block": PARENT_BLOCK}


def make_env():
    import gymnasium as gym
    env = gym.make(ENV_ID).unwrapped
    m = env.model
    if (env.frame_skip, float(m.opt.timestep), int(m.opt.integrator)) != (5, 0.01, 0):
        raise RuntimeError("unexpected HalfCheetah-v5 timing")
    if m.nu != ACT_DIM or env.observation_space.shape != (OBS_DIM,):
        raise RuntimeError("unexpected HalfCheetah-v5 dims")
    return env, E.Applier(m)


def warnings_total(env):
    import mujoco
    return {w: int(env.data.warning[int(getattr(mujoco.mjtWarning, w))].number) for w in WARN_NAMES}


def pools(namespace: str = NAMESPACE):
    return E.make_pools(derive_seed(namespace, "pools"), (0.0, 0.0, -9.81), _LOG_SCALE_LIMIT[0])


def episode_design(namespace, split, i):
    rng = np.random.default_rng(derive_seed(namespace, "episode", split, i))
    task_list = E.episode_task_list(rng, E.TASK_NUM if SPLITS[split][1] == "train" else E.TEST_TASK_NUM)
    s0 = int(rng.integers(0, OFFSET_MAX_EXCL))
    kind = ACTION_KINDS[i % len(ACTION_KINDS)]
    plan_seed = derive_seed(namespace, "plan", split, i)
    reset_seed = derive_seed(namespace, "reset", split, i)
    return {"split": split, "index": i, "task_list": task_list, "offset": s0, "kind": kind, "plan_seed": plan_seed,
            "reset_seed": reset_seed, "pool": SPLITS[split][1]}


def episode_actions(d):
    from simcore.vendor.branch_actions import generate_action_plan
    plan = generate_action_plan(d["kind"], PLAN_LEN, ACT_DIM, int(d["plan_seed"]), ACTION_LIMIT).actions.copy()
    s0 = d["offset"]
    acts = plan[: s0 + T].copy()
    for lo, hi in LOW_AMP_WINDOWS:
        acts[s0 + lo: s0 + hi] *= np.float32(LOW_AMP_MULT)
    return acts.astype(np.float32)


def labels(d, pool):
    s0 = d["offset"]
    li = np.array([E.list_index_of_transition(s0 + i) for i in range(T)], np.int8)
    tid = np.array([d["task_list"][k] for k in li], np.int16)
    mass = np.array([pool[k]["body_mass"][0] for k in tid], np.float64)
    damp = np.array([pool[k]["dof_damping_1_dim"][0] for k in tid], np.float64)
    gz = np.array([pool[k]["gravity"][2] for k in tid], np.float64)
    chg_state, chg_eff = [], []
    for i in range(T):
        j = s0 + i
        if j >= 1 and E.list_index_of_transition(j) != E.list_index_of_transition(j - 1):
            chg_state.append(i + 1)
            a, b = E.list_index_of_transition(j - 1), E.list_index_of_transition(j)
            chg_eff.append(bool(d["task_list"][a] != d["task_list"][b]))
    if len(chg_state) != 3:
        raise AssertionError(f"expected 3 scheduled boundaries in the window, got {chg_state}")
    eff = [c for c, e in zip(chg_state, chg_eff) if e]
    since_group, since_steps, chg_k = [], [], []
    for t in EVAL_STARTS:
        ks = [c - t for c in eff if t < c <= t + HORIZON]
        prev = [t - c for c in eff if c <= t]
        k = min(ks) if ks else -1
        s = min(prev) if prev else -1
        if k >= 1:
            g = 1 if k <= 20 else 2
        elif s < 0 or s >= 64:
            g = 5
        else:
            g = 3 if s < 16 else 4
        since_group.append(g)
        since_steps.append(s)
        chg_k.append(k)
    return {"list_index": li, "task_id": tid, "mass_mult": mass.astype(np.float32), "damp_mult": damp.astype(np.float32),
            "grav_z": gz.astype(np.float32), "grav_mult": (gz / -9.81).astype(np.float32),
            "chg_state": np.array(chg_state, np.int16), "chg_eff": np.array(chg_eff, bool),
            "since_group": np.array(since_group, np.int8), "since_steps": np.array(since_steps, np.int16),
            "chg_k": np.array(chg_k, np.int16)}


def simulate(env, applier, d, pool, acts):
    tasks = [pool[k] for k in d["task_list"]]
    applier.set_task(tasks[0])
    obs0, _ = env.reset(seed=int(d["reset_seed"]))
    sched = E.Schedule()
    s0 = d["offset"]
    out = [np.asarray(obs0, np.float64)] if s0 == 0 else []
    applied = np.zeros((s0 + T, 3), np.float64)
    for j in range(s0 + T):
        ind = sched.before_transition()
        if ind is not None:
            applier.set_task(tasks[ind])
        cur = applier.current()
        applied[j] = (cur["torso_mass"] / applier.init_mass[E.TORSO_BODY],
                      cur["dof_damping"][3] / applier.init_damping[3], cur["gravity"][2])
        a = acts[j]
        if np.any(np.abs(a) > ACTION_LIMIT):
            raise ValueError("action outside the limit")
        ob, _, term, trunc, _ = env.step(a)
        if term or trunc:
            raise RuntimeError("HalfCheetah episode ended early")
        ob = np.asarray(ob, np.float64)
        if not np.isfinite(ob).all():
            raise FloatingPointError("non-finite observation")
        if j + 1 >= s0:
            out.append(ob)
    w = warnings_total(env)
    if any(w.values()):
        raise RuntimeError(f"MuJoCo warnings {w}")
    obs = np.asarray(out, np.float64)
    if obs.shape != (T + 1, OBS_DIM):
        raise AssertionError(obs.shape)
    return obs.astype(np.float32), applied[s0:], w


def collect_split(namespace, split, env=None, applier=None, replay=True, log=None):
    if env is None:
        env, applier = make_env()
    train_pool, test_pool = pools(namespace)
    pool = train_pool if SPLITS[split][1] == "train" else test_pool
    n = SPLITS[split][0]
    rows = {k: [] for k in ("obs", "act", "parent", "cohort", "sequence", "kind", "offset", "task_list", "pool",
                            "plan_seed", "reset_seed", "list_index", "task_id", "mass_mult", "damp_mult", "grav_z",
                            "grav_mult", "chg_state", "chg_eff", "since_group", "since_steps", "chg_k")}
    audit = {"replay_exact": 0, "replay_checked": 0, "applied_matches_labels": 0, "warnings": 0}
    t0 = time.time()
    for i in range(n):
        d = episode_design(namespace, split, i)
        acts = episode_actions(d)
        lab = labels(d, pool)
        obs, applied, w = simulate(env, applier, d, pool, acts)
        mass = np.array([pool[k]["body_mass"][0] for k in lab["task_id"]], np.float64)
        damp = np.array([pool[k]["dof_damping_1_dim"][0] for k in lab["task_id"]], np.float64)
        gz = np.array([pool[k]["gravity"][2] for k in lab["task_id"]], np.float64)
        ok = (np.allclose(applied[:, 0], mass, rtol=1e-12, atol=0) and np.allclose(applied[:, 1], damp, rtol=1e-12, atol=0)
              and np.array_equal(applied[:, 2], gz))
        if not ok:
            raise AssertionError(f"{split} {i}: applied parameters differ from labels")
        audit["applied_matches_labels"] += 1
        if replay:
            obs2, _, _ = simulate(env, applier, d, pool, acts)
            audit["replay_checked"] += 1
            if not np.array_equal(obs, obs2):
                raise RuntimeError(f"replay audit failed {split} {i}")
            audit["replay_exact"] += 1
        s0 = d["offset"]
        rows["obs"].append(obs)
        rows["act"].append(acts[s0: s0 + T])
        rows["parent"].append(f"escphc-{split}-{i // PARENT_BLOCK:02d}")
        rows["cohort"].append(0)
        rows["sequence"].append("escp")
        rows["kind"].append(d["kind"])
        rows["offset"].append(s0)
        rows["task_list"].append(d["task_list"])
        rows["pool"].append(d["pool"])
        rows["plan_seed"].append(d["plan_seed"])
        rows["reset_seed"].append(d["reset_seed"])
        for k in ("list_index", "task_id", "mass_mult", "damp_mult", "grav_z", "grav_mult", "chg_state", "chg_eff",
                  "since_group", "since_steps", "chg_k"):
            rows[k].append(lab[k])
        if log and (i + 1) % 48 == 0:
            log(f"{split}: {i + 1}/{n} sequences, {time.time() - t0:.1f} s")
    arrays = {
        "obs": np.asarray(rows["obs"], np.float32), "act": np.asarray(rows["act"], np.float32),
        "parent": np.asarray(rows["parent"]), "cohort": np.asarray(rows["cohort"], np.int8),
        "sequence": np.asarray(rows["sequence"]), "kind": np.asarray(rows["kind"]),
        "offset": np.asarray(rows["offset"], np.int16), "task_list": np.asarray(rows["task_list"], np.int16),
        "pool": np.asarray(rows["pool"]), "plan_seed": np.asarray(rows["plan_seed"], np.int64),
        "reset_seed": np.asarray(rows["reset_seed"], np.int64), "eval_starts": np.asarray(EVAL_STARTS, np.int16)}
    for k in ("list_index", "task_id", "mass_mult", "damp_mult", "grav_z", "grav_mult", "chg_state", "chg_eff",
              "since_group", "since_steps", "chg_k"):
        arrays[k] = np.asarray(rows[k])
    audit["seconds"] = round(time.time() - t0, 2)
    return arrays, audit


def save_npz_det(path, arrays: dict):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            a = np.ascontiguousarray(arrays[name])
            if a.dtype == object:
                raise TypeError(f"object array {name}")
            b = io.BytesIO()
            np.lib.format.write_array(b, a, allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, b.getvalue())
    tmp = path.with_name(".tmp-" + path.name)
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def array_sha256(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def code_files():
    here = Path(__file__).resolve().parent
    root = here.parent
    files = sorted(list(here.glob("*.py")) + [root / "simcore" / "vendor" / "branch_actions.py"])
    return {str(p.relative_to(root)): sha256_file(p) for p in files if "__pycache__" not in p.parts}


def runtime_identity():
    import gymnasium
    import mujoco
    info = {"python": sys.version, "numpy": np.__version__, "mujoco": mujoco.__version__,
            "gymnasium": gymnasium.__version__, "platform": platform.platform(), "machine": platform.machine(),
            "env": {k: os.environ.get(k) for k in ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "OMP_NUM_THREADS",
                                                   "PYTHONHASHSEED")}}
    try:
        import gymnasium as g
        xml = Path(g.__file__).parent / "envs" / "mujoco" / "assets" / "half_cheetah.xml"
        info["half_cheetah_xml_sha256"] = sha256_file(xml)
        info["half_cheetah_v5_py_sha256"] = sha256_file(Path(g.__file__).parent / "envs" / "mujoco" / "half_cheetah_v5.py")
    except Exception as e:
        info["xml_error"] = repr(e)
    return info


def pools_json(namespace=NAMESPACE):
    tr, te = pools(namespace)
    conv = lambda P: [{k: [float(x) for x in v] for k, v in t.items()} for t in P]
    return {"seed": derive_seed(namespace, "pools"), "train": conv(tr), "test": conv(te)}
