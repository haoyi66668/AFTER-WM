from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C

ARRAY_KEYS = ("obs", "act", "fail_joint", "changed", "phase", "parent", "cohort", "sequence")
SPLIT_ORDER = ("train", "val", "test")


def _mgen():
    from simcore import gen as mgen
    return mgen


def code_files() -> dict:
    here = Path(__file__).resolve().parent
    root = here.parent
    files = sorted(list(here.glob("*.py")) + [root / "simcore" / "gen.py", root / "simcore" / "__init__.py"]
                   + sorted((root / "simcore" / "vendor").glob("*.py")))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            if "__pycache__" not in p.parts}


def code_sha256(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def design_parents(protocol, counts, seed_offset=0):
    mgen = _mgen()
    counts = dict(counts)
    unknown = set(counts) - set(SPLIT_ORDER)
    if unknown:
        raise KeyError(f"unknown split names {sorted(unknown)}")
    ns = protocol["namespace"]
    total = sum(counts.values())
    quota = protocol["contact_quota"]
    regimes = []
    for name in ("stable", "transition", "airborne"):
        regimes += [name] * int(round(quota[name] * total))
    while len(regimes) < total:
        regimes.append("stable")
    regimes = regimes[:total]
    np.random.default_rng(mgen.derive_seed(ns, "regimes", seed_offset)).shuffle(regimes)
    designs, index = [], 0
    kinds = protocol["action_kinds"]
    for split in SPLIT_ORDER:
        for j in range(counts.get(split, 0)):
            label = "testB" if (split == "test" and int(seed_offset) == 1) else split
            designs.append({"index": index, "parent_id": f"jshc-{label}-{j:04d}", "split": split,
                            "contact_regime": regimes[index],
                            "cohort_kinds": [kinds[index % len(kinds)], kinds[(index + 1) % len(kinds)]]})
            index += 1
    return designs


def switch_times(protocol, phase):
    js = protocol["joint_switch"]
    T, per = protocol["T"], js["switch_period"]
    return [0] + [phase + per * k for k in range(T // per + 1) if 0 < phase + per * k < T]


def joint_schedule(protocol, parent_index, cohort, seq, seed_offset=0):
    mgen = _mgen()
    js = protocol["joint_switch"]
    seed = mgen.derive_seed(protocol["namespace"], "joints", seed_offset, parent_index, cohort, seq)
    rng = np.random.default_rng(seed)
    p_lo, p_hi = js["phase_range"]
    phase = int(rng.integers(p_lo, p_hi))
    cand = js["candidate_joints"]
    lo, hi = min(cand), max(cand) + 1
    if list(range(lo, hi)) != list(cand):
        raise ValueError("candidate joints must be a contiguous range")
    times = switch_times(protocol, phase)
    draws = [int(x) for x in rng.integers(lo, hi, size=len(times))]
    T = protocol["T"]
    fail = np.empty(T, np.int8)
    bounds = times + [T]
    for k in range(len(draws)):
        fail[bounds[k]:bounds[k + 1]] = draws[k]
    return fail, draws, phase, seed


def changed_flags(fail):
    ch = np.zeros(len(fail), np.int8)
    ch[1:] = (fail[1:] != fail[:-1]).astype(np.int8)
    return ch


def rollout(adapter, snapshot, actions, fail):
    adapter.restore_integration_snapshot(snapshot, parameters={"friction_scale": 1.0, "gain_scale": 1.0})
    obs = [snapshot.flat_observation.copy()]
    for t in range(len(actions)):
        adapter.apply_absolute_dynamics({"friction_scale": 1.0, "gain_scale": 1.0})
        applied = np.array(actions[t], dtype=np.float32, copy=True)
        j = int(fail[t])
        if j >= 0:
            applied[j] = np.float32(0.0)
        res = adapter.step_with_impulse(applied, None)
        if not np.array_equal(res.model_action, applied):
            raise ValueError("executed action differs from the masked plan (clipping is not allowed)")
        if res.episode_end and t < len(actions) - 1:
            raise RuntimeError("episode ended early")
        if not np.isfinite(res.flat_observation).all():
            raise FloatingPointError("non-finite observation")
        obs.append(res.flat_observation.copy())
    return np.asarray(obs, np.float32)


def collect_parent(adapter, protocol, design, seed_offset, replay_all=True):
    mgen = _mgen()
    t0 = time.time()
    snap, anchor = mgen.discover_anchor(adapter, protocol, design, seed_offset)
    rows = {k: [] for k in ARRAY_KEYS}
    rec = {**copy.deepcopy(design), "anchor": anchor, "cohorts": [], "replay_exact": [], "nominal_equals_base_rollout": []}
    T = protocol["T"]
    for cohort in (0, 1):
        acts, plan_seed = mgen.action_plan(protocol, design["cohort_kinds"][cohort], design["index"], cohort, seed_offset)
        nominal = rollout(adapter, snap, acts, np.full(T, -1, np.int8))
        ref = mgen.rollout(adapter, snap, acts, np.ones(T), np.ones(T), np.zeros((T, 3)))
        rec["nominal_equals_base_rollout"].append({"cohort": cohort, "equal": bool(np.array_equal(ref, nominal))})
        seqs = []
        for s, sid in enumerate(C.SEQUENCE_IDS):
            fail, draws, phase, jseed = joint_schedule(protocol, design["index"], cohort, s, seed_offset)
            obs = rollout(adapter, snap, acts, fail)
            if obs.shape != (T + 1, 17):
                raise AssertionError(f"obs shape {obs.shape}")
            if replay_all:
                again = rollout(adapter, snap, acts, fail)
                ok = bool(np.array_equal(obs, again))
                rec["replay_exact"].append({"cohort": cohort, "sequence": sid, "exact": ok})
                if not ok:
                    raise RuntimeError(f"replay audit failed: {design['parent_id']} cohort {cohort} {sid}")
            d = (obs.astype(np.float64) - nominal.astype(np.float64)) ** 2
            bnd = switch_times(protocol, phase) + [T]
            seg_dev = [d[a + 1:b + 1].mean(0).tolist() for a, b in zip(bnd[:-1], bnd[1:])]
            seqs.append({"sequence": sid, "joint_seed": jseed, "phase": phase, "draws": draws, "dev_sq_by_segment": seg_dev,
                         "max_abs_pitch": float(np.abs(obs[:, 1]).max())})
            rows["obs"].append(obs)
            rows["act"].append(acts.astype(np.float32))
            rows["fail_joint"].append(fail.astype(np.int8))
            rows["changed"].append(changed_flags(fail))
            rows["phase"].append(phase)
            rows["parent"].append(design["parent_id"])
            rows["cohort"].append(cohort)
            rows["sequence"].append(sid)
        rec["cohorts"].append({"cohort": cohort, "action_kind": design["cohort_kinds"][cohort], "plan_seed": plan_seed,
                               "nominal_max_abs_pitch": float(np.abs(nominal[:, 1]).max()), "sequences": seqs})
    rec["seconds"] = round(time.time() - t0, 2)
    return rows, rec


def _worker(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    protocol, designs, seed_offset, replay_all = job
    mgen = _mgen()
    adapter, _ = mgen.make_adapter(protocol)
    return [(d["index"],) + collect_parent(adapter, protocol, d, seed_offset, replay_all) for d in designs]


def runtime_identity() -> dict:
    import gymnasium
    import mujoco
    info = {"python": sys.version, "numpy": np.__version__, "mujoco": mujoco.__version__,
            "gymnasium": gymnasium.__version__, "platform": platform.platform(), "machine": platform.machine(),
            "hostname": platform.node(),
            "env": {k: os.environ.get(k) for k in ("NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "OMP_NUM_THREADS",
                                                   "PYTHONHASHSEED")}}
    try:
        xml = Path(gymnasium.__file__).parent / "envs" / "mujoco" / "assets" / "half_cheetah.xml"
        info["half_cheetah_xml_sha256"] = hashlib.sha256(xml.read_bytes()).hexdigest()
    except OSError:
        pass
    return info


def collect(output, protocol, counts, seed_offset=0, workers=1, log=print, replay_all=True, file_prefix="jshc"):
    mgen = _mgen()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    if not C.shared_keys_ok(protocol):
        raise RuntimeError("shared keys differ from simcore.gen.BASE_PROTOCOL")
    output.mkdir(parents=True)
    designs = design_parents(protocol, counts, seed_offset)
    t_start = time.time()
    workers = max(1, min(int(workers), len(designs)))
    jobs = [(protocol, designs[i::workers], seed_offset, replay_all) for i in range(workers)]
    jobs = [j for j in jobs if j[1]]
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
            results = pool.map(_worker, jobs)
    by_index = {idx: (rows, rec) for res in results for idx, rows, rec in res}
    if sorted(by_index) != [d["index"] for d in designs]:
        raise RuntimeError("missing parents after collection")
    manifest = {"protocol": protocol, "counts": counts, "seed_offset": seed_offset, "workers": workers,
                "runtime": runtime_identity(), "code_files": code_files(), "parents": [], "files": {}}
    manifest["code_sha256"] = code_sha256(manifest["code_files"])
    splits = {}
    for d in designs:
        rows, rec = by_index[d["index"]]
        acc = splits.setdefault(d["split"], {k: [] for k in ARRAY_KEYS})
        for k in ARRAY_KEYS:
            acc[k].extend(rows[k])
        manifest["parents"].append(rec)
        log(json.dumps({"stage": "jshc_parent", "parent": d["parent_id"], "regime": d["contact_regime"],
                        "anchor_step": rec["anchor"]["anchor_step"], "seconds": rec["seconds"]}))
    for split, acc in splits.items():
        arrays = {"obs": np.stack(acc["obs"]).astype(np.float32), "act": np.stack(acc["act"]).astype(np.float32),
                  "fail_joint": np.stack(acc["fail_joint"]).astype(np.int8),
                  "changed": np.stack(acc["changed"]).astype(np.int8),
                  "phase": np.asarray(acc["phase"], np.int16),
                  "parent": np.asarray(acc["parent"]), "cohort": np.asarray(acc["cohort"], np.int8),
                  "sequence": np.asarray(acc["sequence"])}
        path = output / f"{file_prefix}_{split}.npz"
        mgen.save_npz(path, **arrays)
        manifest["files"][split] = {"path": path.name, "sha256": mgen.sha256_file(path), "episodes": int(len(acc["obs"])),
                                    "array_sha256": {k: mgen.array_sha256(v) for k, v in arrays.items()}}
    manifest["elapsed_s"] = round(time.time() - t_start, 1)
    manifest["checks"] = {
        "replay_exact_all": bool(all(r["exact"] for p in manifest["parents"] for r in p["replay_exact"])),
        "nominal_equals_base_rollout_all": bool(all(r["equal"] for p in manifest["parents"]
                                                   for r in p["nominal_equals_base_rollout"])),
        "finite_all": True, "full_length_all": True, "executed_action_equals_masked_plan_all": True}
    mgen.write_json(output / "JSHC_MANIFEST.json", manifest)
    return manifest


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
