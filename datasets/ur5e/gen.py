from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np

from simcore.gen import _jsonable, array_sha256, derive_seed, save_npz, sha256_file, write_json

SEQUENCES = ("null", "persistent", "switch", "recur", "recur_nominal", "drift")
ARRAY_KEYS = ("obs", "act", "friction", "gain", "payload", "friction_eff", "gain_eff", "wrench", "impulse", "seg",
              "parent", "cohort", "sequence", "plan_attempt")

PROTOCOL_UR5E = {
    "schema": "ur5e_v1",
    "namespace": "Code5.TSD.MTX-UR5e-v1",
    "plant": "ur5e/env.py PLANT (Menagerie universal_robots_ur5e @ 6da8b971, compliant joint impedance 0.03 x Menagerie kp, same damping ratio)",
    "obs_dim": 12,
    "act_dim": 6,
    "dt": 0.05,
    "T": 300,
    "anchor_step_min": 16,
    "anchor_step_max": 80,
    "max_parent_attempts": 4000,
    "anchor_posture_rad": 0.3,
    "rollin_amp_rad": 0.3,
    "action_limit": 0.75,
    "action_kinds": ["smooth_ar1", "piecewise", "periodic", "iid"],
    "plan": {"posture_rad": 0.3, "amp_rad": 0.3},
    "low_amplitude_windows": [[110, 140], [230, 255]],
    "low_amplitude_multiplier": 0.15,
    "segments": [0, 40, 120, 190, 260, 300],
    "sequences": list(SEQUENCES),
    "friction_prior": {"low": [0.1, 0.35, 0.4], "high": [2.0, 3.0, 0.2], "mid": [0.35, 2.0, 0.4]},
    "gain_prior": {"nominal_prob": 0.25, "range": [0.8, 1.2]},
    "payload_prior": {"nominal_prob": 0.25, "range": [1.0, 4.0]},
    "min_condition_distance": {"log_friction": 0.2, "gain": 0.05, "payload": 0.25},
    "drift_step": {"log_friction": 0.15, "gain": 0.02, "payload": 0.4},
    "drift_clip": {"friction": [0.1, 3.0], "gain": [0.7, 1.3], "payload": [0.0, 5.0]},
    "pulse": {"start": 20, "rate": 1.0 / 25.0, "duration": [4, 12], "channel_probs": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
              "channels": ["wrench", "gain", "friction"], "wrench_newton": [5.0, 25.0], "gain_factor": [0.875, 1.125],
              "log_friction_delta": [-1.0, 1.0]},
    "impulse": {"start": 45, "rate": 1.0 / 30.0, "sd_newton": [15.0, 15.0, 15.0]},
    "parents": {"train": 24, "val": 12, "test": 12},
    "max_plan_attempts": 25,
    "max_anchor_redraws": 10,
    "parent_prefix": "ur5e",
}
DESIGNS = ("ur5e", "unittest")


def default_config(design="ur5e"):
    if design not in DESIGNS:
        raise KeyError(f"unknown design {design!r}; known: {DESIGNS}")
    p = copy.deepcopy(PROTOCOL_UR5E)
    if design == "unittest":
        p["namespace"] = PROTOCOL_UR5E["namespace"].rsplit("-", 1)[0] + "-unittest"
        p["parent_prefix"] = "ur5eunit"
    return {"design": design, "protocol": p}


def make_adapter(protocol=None, plant=None):
    from .env import UR5eAdapter
    ad = UR5eAdapter(plant)
    if (ad.obs_dim, ad.act_dim) != (12, 6) or abs(ad.dt - 0.05) > 1e-12:
        raise RuntimeError("UR5e plant does not have the expected dims / dt")
    return ad


def frame(adapter):
    return (np.asarray(adapter.q_nom, np.float64), np.asarray(adapter.mid, np.float64), np.asarray(adapter.half, np.float64))


def _draw_condition(rng, protocol):
    fp, gp, pp = protocol["friction_prior"], protocol["gain_prior"], protocol["payload_prior"]
    md = protocol["min_condition_distance"]
    for _ in range(1000):
        u = rng.random()
        if u < fp["low"][2]:
            lo, hi = fp["low"][:2]
        elif u < fp["low"][2] + fp["high"][2]:
            lo, hi = fp["high"][:2]
        else:
            lo, hi = fp["mid"][:2]
        f = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        g = 1.0 if rng.random() < gp["nominal_prob"] else float(rng.uniform(*gp["range"]))
        m = 0.0 if rng.random() < pp["nominal_prob"] else float(rng.uniform(*pp["range"]))
        if abs(np.log(f)) >= md["log_friction"] or abs(g - 1.0) >= md["gain"] or m >= md["payload"]:
            return (f, g, m)
    raise RuntimeError("could not draw a non-nominal condition")


def design_parents(protocol, counts=None, seed_offset=0):
    counts = dict(protocol["parents"] if counts is None else counts)
    ns = protocol["namespace"]
    designs, index = [], 0
    kinds = protocol["action_kinds"]
    for split in ("train", "val", "test"):
        for j in range(counts.get(split, 0)):
            rng = np.random.default_rng(derive_seed(ns, "values", seed_offset, index))
            A, B, C = (_draw_condition(rng, protocol) for _ in range(3))
            designs.append({
                "index": index, "parent_id": f"{protocol['parent_prefix']}-{split}-{j:04d}", "split": split,
                "cohort_kinds": [kinds[index % len(kinds)], kinds[(index + 1) % len(kinds)]],
                "values": {"A": A, "B": B, "C": C},
            })
            index += 1
    return designs


def _plan_value(name, i, values, protocol):
    N = (1.0, 1.0, 0.0)
    A, B, C = values["A"], values["B"], values["C"]
    ds, dc = protocol["drift_step"], protocol["drift_clip"]
    if i == 0 or name == "null":
        return N
    if name == "persistent":
        return A
    if name == "switch":
        return (A, B, C)[min(i, 3) - 1]
    if name == "recur":
        return A if i % 2 == 1 else B
    if name == "recur_nominal":
        return A if i % 2 == 1 else N
    if name == "drift":
        if i == 1:
            return A
        k = i - 1
        return (float(np.clip(A[0] * np.exp(ds["log_friction"] * k), *dc["friction"])),
                float(np.clip(A[1] + ds["gain"] * k, *dc["gain"])),
                float(np.clip(A[2] + ds["payload"] * k, *dc["payload"])))
    raise KeyError(name)


def condition_sequences(values, protocol):
    T = protocol["T"]
    segs = list(protocol["segments"])
    out = {}
    for name in protocol["sequences"]:
        fr, gn, pl = np.ones(T), np.ones(T), np.zeros(T)
        seg = np.zeros(T, np.int8)
        for i, (lo, hi) in enumerate(zip(segs[:-1], segs[1:])):
            fr[lo:hi], gn[lo:hi], pl[lo:hi] = _plan_value(name, i, values, protocol)
            seg[lo:hi] = i
        out[name] = (fr, gn, pl, seg)
    return out


def pulse_schedule(protocol, parent_index, cohort, attempt, seed_offset=0):
    pp, T = protocol["pulse"], protocol["T"]
    rng = np.random.default_rng(derive_seed(protocol["namespace"], "pulse", seed_offset, parent_index, cohort, attempt))
    gain_f = np.ones(T)
    logf_d = np.zeros(T)
    wrench = np.zeros((T, 3))
    p_w, p_g = pp["channel_probs"][0], pp["channel_probs"][0] + pp["channel_probs"][1]
    t = pp["start"]
    while t < T:
        if rng.random() < pp["rate"]:
            L = int(rng.integers(pp["duration"][0], pp["duration"][1] + 1))
            u = rng.random()
            if u < p_w:
                v = rng.normal(size=3)
                v = v / np.linalg.norm(v)
                wrench[t:t + L] = v * rng.uniform(*pp["wrench_newton"])
            elif u < p_g:
                gain_f[t:t + L] = rng.uniform(*pp["gain_factor"])
            else:
                logf_d[t:t + L] = rng.uniform(*pp["log_friction_delta"])
            t += L
        else:
            t += 1
    return gain_f, logf_d, wrench


def impulse_schedule(protocol, parent_index, cohort, attempt, seed_offset=0):
    imp, T = protocol["impulse"], protocol["T"]
    rng = np.random.default_rng(derive_seed(protocol["namespace"], "impulse", seed_offset, parent_index, cohort, attempt))
    forces = np.zeros((T, 3), np.float32)
    for t in range(imp["start"], T):
        if rng.random() < imp["rate"]:
            forces[t] = rng.normal(size=3) * np.asarray(imp["sd_newton"])
    return forces


def disturbance_labels(protocol, design, cohort, attempt, seed_offset=0):
    seqs = condition_sequences(design["values"], protocol)
    gain_f, logf_d, wrench = pulse_schedule(protocol, design["index"], cohort, attempt, seed_offset)
    imps = impulse_schedule(protocol, design["index"], cohort, attempt, seed_offset)
    out = {}
    for name in protocol["sequences"]:
        fr, gn, pl, seg = seqs[name]
        out[name] = {"friction": fr, "gain": gn, "payload": pl, "friction_eff": fr * np.exp(logf_d),
                     "gain_eff": gn * gain_f, "seg": seg}
    return out, wrench, imps


def action_plan(protocol, kind, parent_index, cohort, attempt, frm, seed_offset=0):
    q_nom, mid, half = (np.asarray(x, np.float64) for x in frm)
    from simcore.vendor.branch_actions import generate_action_plan
    ns, T, pl = protocol["namespace"], protocol["T"], protocol["plan"]
    bseed = derive_seed(ns, "posture", seed_offset, parent_index, cohort, attempt)
    b = np.random.default_rng(bseed).uniform(-pl["posture_rad"], pl["posture_rad"], size=6)
    pseed = derive_seed(ns, "plan", seed_offset, parent_index, cohort, attempt)
    p = generate_action_plan(kind, T, 6, pseed, 1.0).actions.astype(np.float64) * float(pl["amp_rad"])
    w = np.ones(T)
    for lo, hi in protocol["low_amplitude_windows"]:
        w[lo:hi] = protocol["low_amplitude_multiplier"]
    acts = ((q_nom[None, :] + b[None, :] + w[:, None] * p - mid[None, :]) / half[None, :]).astype(np.float32)
    if np.abs(acts).max() > protocol["action_limit"]:
        raise RuntimeError("plan exceeds the action limit (never clipped)")
    return acts, {"kind": kind, "attempt": int(attempt), "posture_seed": bseed, "plan_seed": pseed,
                  "posture_rad": b.tolist()}


def discover_anchor(adapter, protocol, design, seed_offset=0, start_attempt=0):
    from simcore.vendor.branch_collection_min import _build_parent_rollin_actions
    ns = protocol["namespace"]
    skipped = []
    for attempt in range(int(start_attempt), protocol["max_parent_attempts"]):
        parent_seed = derive_seed(ns, "parent", seed_offset, design["index"], attempt)
        rng = np.random.default_rng(parent_seed)
        off = rng.uniform(-protocol["anchor_posture_rad"], protocol["anchor_posture_rad"], size=6)
        t_anchor = int(rng.integers(protocol["anchor_step_min"], protocol["anchor_step_max"] + 1))
        adapter.reset_to(adapter.q_nom + off)
        ro = _build_parent_rollin_actions(action_dim=6, steps=protocol["anchor_step_max"],
                                          seed=derive_seed(ns, "rollin", parent_seed), action_limit=1.0)
        ro = ((adapter.q_nom[None, :] + off[None, :] + protocol["rollin_amp_rad"] * ro.astype(np.float64) - adapter.mid[None, :])
              / adapter.half[None, :]).astype(np.float32)
        for t in range(t_anchor):
            adapter.step(ro[t], None, t)
            if adapter.violation is not None:
                break
        if adapter.violation is not None:
            skipped.append({"attempt": attempt, **adapter.violation})
            continue
        snap = adapter.capture()
        return snap, {"parent_seed": parent_seed, "attempt": attempt, "anchor_step": t_anchor,
                      "posture_offset_rad": off.tolist(), "state_sha256": snap["state_sha256"],
                      "unhealthy_rollins_skipped": skipped}
    raise RuntimeError(f"anchor search exhausted for {design['parent_id']}")


def rollout(adapter, snap, actions, friction, gain, payload, wrench, impulses):
    adapter.restore(snap)
    obs = [snap["obs"].copy()]
    for t in range(len(actions)):
        adapter.set_dynamics(float(friction[t]), float(gain[t]), float(payload[t]))
        f = np.asarray(wrench[t], np.float64) + np.asarray(impulses[t], np.float64)
        obs.append(adapter.step(actions[t], f if np.any(f != 0) else None, t))
    adapter.set_dynamics(1.0, 1.0, 0.0)
    return np.asarray(obs, np.float32), adapter.violation


def _simulate_parent(adapter, protocol, d, snap, anchor, seed_offset, manifest, log):
    out = []
    for cohort in (0, 1):
        accepted = None
        for attempt in range(protocol["max_plan_attempts"]):
            labels, wrench, imps = disturbance_labels(protocol, d, cohort, attempt, seed_offset)
            acts, info = action_plan(protocol, d["cohort_kinds"][cohort], d["index"], cohort, attempt, frame(adapter),
                                     seed_offset)
            episodes, failed = [], None
            for name in protocol["sequences"]:
                lab = labels[name]
                obs, bad = rollout(adapter, snap, acts, lab["friction_eff"], lab["gain_eff"], lab["payload"], wrench, imps)
                if bad is not None:
                    failed = {"parent_id": d["parent_id"], "cohort": cohort, "attempt": attempt,
                              "kind": d["cohort_kinds"][cohort], "sequence": name, "step": bad["t"],
                              "reason": bad["reason"], "detail": bad.get("bodies"),
                              "anchor_attempt": int(anchor["attempt"])}
                    break
                episodes.append((name, obs))
            if failed is None:
                accepted = (attempt, acts, info, episodes, labels, wrench, imps)
                break
            manifest["rejections"].append(failed)
            log("ur5e_reject", **failed)
        if accepted is None:
            return None, cohort
        out.append((cohort, accepted))
    return out, None


def collect(output, config=None, counts=None, log=print, seed_offset=0):
    config = default_config("ur5e") if config is None else config
    protocol = config["protocol"]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    adapter = make_adapter(protocol)
    designs = design_parents(protocol, counts, seed_offset)
    manifest = {"design": config.get("design"), "protocol": copy.deepcopy(protocol),
                "versions": {"mujoco": adapter.mujoco.__version__, "numpy": np.__version__},
                "env": adapter.identity(), "parents": [], "replay_audit": [], "rejections": [],
                "anchor_rejections": [], "seed_offset": seed_offset, "counts": counts}
    max_redraws = int(protocol.get("max_anchor_redraws", 0))
    splits = {}
    t_start = time.time()
    for d in designs:
        rows = splits.setdefault(d["split"], {k: [] for k in ARRAY_KEYS})
        parent_anchor_rejections = []
        next_anchor_attempt = 0
        result = None
        for _ in range(max_redraws + 1):
            snap, anchor = discover_anchor(adapter, protocol, d, seed_offset, start_attempt=next_anchor_attempt)
            result, exhausted = _simulate_parent(adapter, protocol, d, snap, anchor, seed_offset, manifest, log)
            if result is not None:
                break
            rec = {"parent_id": d["parent_id"], "anchor_attempt": int(anchor["attempt"]),
                   "anchor_step": int(anchor["anchor_step"]), "state_sha256": anchor["state_sha256"],
                   "exhausted_cohort": int(exhausted), "plan_attempts": int(protocol["max_plan_attempts"])}
            parent_anchor_rejections.append(rec)
            manifest["anchor_rejections"].append(rec)
            log("ur5e_anchor_reject", **rec)
            next_anchor_attempt = int(anchor["attempt"]) + 1
        if result is None:
            last = parent_anchor_rejections[-1]
            raise RuntimeError(f"plan attempts exhausted for {d['parent_id']} cohort {last['exhausted_cohort']} "
                               f"under {len(parent_anchor_rejections)} anchor(s)")
        first = None
        cohorts = []
        for cohort, (attempt, acts, info, episodes, labels, wrench, imps) in result:
            cohorts.append({"cohort": cohort, **info})
            for name, obs in episodes:
                lab = labels[name]
                if first is None:
                    first = (obs, acts, lab, wrench, imps)
                rows["obs"].append(obs)
                rows["act"].append(acts)
                for k in ("friction", "gain", "payload", "friction_eff", "gain_eff"):
                    rows[k].append(lab[k].astype(np.float32))
                rows["wrench"].append(wrench.astype(np.float32))
                rows["impulse"].append(imps)
                rows["seg"].append(lab["seg"])
                rows["parent"].append(d["parent_id"])
                rows["cohort"].append(cohort)
                rows["sequence"].append(name)
                rows["plan_attempt"].append(attempt)
        o, ac, lab, wr, im = first
        again, _ = rollout(adapter, snap, ac, lab["friction_eff"], lab["gain_eff"], lab["payload"], wr, im)
        ok = bool(np.array_equal(o, again))
        manifest["replay_audit"].append({"parent_id": d["parent_id"], "exact": ok})
        if not ok:
            raise RuntimeError(f"replay audit failed for {d['parent_id']}")
        manifest["parents"].append({**d, "anchor": anchor, "cohorts": cohorts,
                                    "anchor_rejections": parent_anchor_rejections})
        log("ur5e_parent", parent=d["parent_id"], split=d["split"], values=d["values"],
            anchor_step=anchor["anchor_step"], anchor_attempt=anchor["attempt"],
            attempts=[c["attempt"] for c in cohorts], anchor_redraws=len(parent_anchor_rejections),
            elapsed_s=round(time.time() - t_start, 1))
    files = {}
    for split, rows in splits.items():
        path = output / f"ur5e_{split}.npz"
        arrays = {k: np.stack(rows[k]) for k in ("obs", "act", "friction", "gain", "payload", "friction_eff",
                                                 "gain_eff", "wrench", "impulse", "seg")}
        arrays.update(parent=np.asarray(rows["parent"]), cohort=np.asarray(rows["cohort"], np.int8),
                      sequence=np.asarray(rows["sequence"]), plan_attempt=np.asarray(rows["plan_attempt"], np.int16))
        save_npz(path, **arrays)
        files[split] = {"path": path.name, "sha256": sha256_file(path), "episodes": len(rows["obs"]),
                        "array_sha256": {k: array_sha256(v) for k, v in arrays.items()}}
    manifest["files"] = files
    manifest["wall_s"] = round(time.time() - t_start, 1)
    write_json(output / "UR5E_MANIFEST.json", manifest)
    return manifest


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
