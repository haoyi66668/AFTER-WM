from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np

from simcore.gen import _jsonable, array_sha256, derive_seed, save_npz, sha256_file, write_json

SEQUENCES = ("null", "persistent", "switch", "recur", "recur_nominal", "drift")
ENV_ID = "ServoHopper-v1"
ENV_ENTRY_POINT = "hopper.env:ServoHopperEnv"
JOINTS = ("thigh", "leg", "foot")

PROTOCOL_HOPPER = {
    "schema": "hopper_v1",
    "namespace": "Code5.TSD.MTX-Hopper-v1",
    "env_id": ENV_ID,
    "env_entry_point": ENV_ENTRY_POINT,
    "env_kwargs": {"frame_skip": 25, "reset_noise_scale": 0.005, "servo_kp": 800.0, "servo_kv": 60.0,
                   "home": [-0.1, -0.1, 0.17], "cone": "elliptic"},
    "max_episode_steps": 1000,
    "obs_dim": 11,
    "act_dim": 3,
    "dt": 0.05,
    "T": 300,
    "anchor_step_min": 16,
    "anchor_step_max": 80,
    "max_parent_attempts": 4000,
    "rollin_rad": [0.05, 0.05, 0.05],
    "action_kinds": ["smooth_ar1", "piecewise", "periodic", "iid"],
    "bounce": {"freq_hz": [0.5, 2.0], "amp_rel": [0.5, 1.0], "amp_rad": [0.03, 0.03, 0.03]},
    "perturbation_rad": [0.03, 0.03, 0.03],
    "offset_box_rad": [0.3, 0.3, 0.3],
    "low_amplitude_windows": [[110, 140], [230, 255]],
    "low_amplitude_multiplier": 0.15,
    "segments": [0, 40, 120, 190, 260, 300],
    "sequences": list(SEQUENCES),
    "friction_prior": {"low": [0.1, 0.35, 0.4], "high": [2.0, 3.0, 0.2], "mid": [0.35, 2.0, 0.4]},
    "friction_eff_max": 3.0,
    "gain_prior": {"nominal_prob": 0.25, "range": [0.6, 1.4]},
    "payload_prior": {"nominal_prob": 0.25, "range": [1.3, 2.5]},
    "min_condition_distance": {"log_friction": 0.2, "gain": 0.1, "log_payload": 0.2},
    "drift_step": {"log_friction": 0.15, "gain": 0.04, "log_payload": 0.1},
    "drift_clip": {"friction": [0.1, 3.0], "gain": [0.5, 1.5], "payload": [1.0, 3.0]},
    "impulse": {"start": 45, "rate": 1.0 / 30.0, "dofs": [0, 1, 2], "sd": [30.0, 30.0, 6.0]},
    "pulse": {"start": 20, "rate": 1.0 / 25.0, "duration": [4, 12], "gain_factor": [0.75, 1.25],
              "log_friction_delta": [-1.0, 1.0], "gain_prob": 0.5},
    "contact_quota": {"flat": 0.6, "transition": 0.2, "rocked": 0.2},
    "parents": {"train": 24, "val": 12, "test": 12},
    "health": {"z_min": 0.7, "angle_max": 0.2, "state_abs_max": 100.0},
    "max_plan_attempts": 25,
    "max_anchor_redraws": 10,
    "parent_prefix": "hopper",
}
ARRAY_KEYS = ("obs", "act", "friction", "gain", "payload", "friction_eff", "gain_eff", "impulse", "seg", "parent",
              "cohort", "sequence", "plan_attempt")
DESIGNS = ("hopper",)


def default_config(design="hopper"):
    if design not in DESIGNS:
        raise KeyError(f"unknown design {design!r}; known: {DESIGNS}")
    p = copy.deepcopy(PROTOCOL_HOPPER)
    return {"design": design, "protocol": p}


def register_env(protocol):
    import gymnasium as gym
    env_id = protocol["env_id"]
    kwargs = copy.deepcopy(protocol["env_kwargs"])
    if env_id in gym.registry:
        spec = gym.registry[env_id]
        if spec.entry_point != protocol["env_entry_point"] or dict(spec.kwargs) != kwargs:
            raise RuntimeError(f"{env_id} is already in the Gymnasium registry with a different spec: {spec}")
        return
    gym.register(id=env_id, entry_point=protocol["env_entry_point"], kwargs=kwargs)


def make_adapter(protocol):
    import mujoco
    from simcore.vendor.gym_mujoco import GymMujocoAdapter

    register_env(protocol)

    class HopperAdapter(GymMujocoAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            m = self._model
            self._base_gainprm = m.actuator_gainprm.copy()
            self._base_biasprm = m.actuator_biasprm.copy()
            self._base_forcerange = m.actuator_forcerange.copy()
            self._foot = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom"))
            if self._foot < 0 or self._floor_geom_id < 0:
                raise RuntimeError("Hopper model lacks the foot geom or the floor geom")
            self._torso = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso"))
            if self._torso < 0:
                raise RuntimeError("Hopper model has no body named 'torso'")
            self._base_body_mass = m.body_mass.copy()
            self._base_body_inertia = m.body_inertia.copy()
            self._current_gain_scale = 1.0
            self._current_payload = None
            self._apply_payload(1.0)

        def _apply_payload(self, payload):
            m, d = self._model, self._data
            state = np.empty(self._integration_state_size, dtype=np.float64)
            mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            m.body_mass[...] = self._base_body_mass
            m.body_inertia[...] = self._base_body_inertia
            m.body_mass[self._torso] = self._base_body_mass[self._torso] * payload
            m.body_inertia[self._torso] = self._base_body_inertia[self._torso] * payload
            mujoco.mj_setConst(m, d)
            mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            self._current_payload = float(payload)

        def _set_absolute_dynamics_without_forward(self, parameters):
            params = dict(parameters)
            gain = float(params.pop("gain_scale", 1.0))
            payload = float(params.pop("payload_scale", 1.0))
            if not np.isfinite(gain) or gain <= 0:
                raise ValueError("gain_scale must be finite and positive")
            if not np.isfinite(payload) or payload <= 0:
                raise ValueError("payload_scale must be finite and positive")
            super()._set_absolute_dynamics_without_forward(params)
            m = self._model
            m.actuator_gainprm[...] = self._base_gainprm
            m.actuator_biasprm[...] = self._base_biasprm
            m.actuator_forcerange[...] = self._base_forcerange
            m.actuator_gainprm[:, 0] = self._base_gainprm[:, 0] * gain
            m.actuator_biasprm[:, 1] = self._base_biasprm[:, 1] * gain
            m.actuator_biasprm[:, 2] = self._base_biasprm[:, 2] * gain
            m.actuator_forcerange[...] = self._base_forcerange * gain
            self._current_gain_scale = gain
            if payload != self._current_payload:
                self._apply_payload(payload)

        def step_with_impulse(self, model_action, impulse=None):
            data = self._data
            data.qfrc_applied[:] = 0.0
            if impulse is not None:
                data.qfrc_applied[protocol["impulse"]["dofs"]] = np.asarray(impulse, np.float64)
            try:
                return self.step(model_action)
            finally:
                data.qfrc_applied[:] = 0.0

        def foot_contacts(self):
            d = self._data
            pairs = []
            n_foot = 0
            for i in range(int(d.ncon)):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                pairs.append(tuple(sorted((g1, g2))))
                if {g1, g2} == {self._floor_geom_id, self._foot}:
                    n_foot += 1
            return tuple(sorted(pairs)), n_foot

        def dynamics_now(self):
            return {"friction_scale": float(self._current_friction_scale),
                    "gain_scale": float(self._current_gain_scale),
                    "payload_scale": float(self._current_payload),
                    "torso_mass": float(self._model.body_mass[self._torso])}

        def env_identity(self):
            import gymnasium as gym
            from . import env as henv
            u = self._env.unwrapped
            spec = gym.registry[protocol["env_id"]]
            m = self._model
            return {"env_id": protocol["env_id"], "entry_point": spec.entry_point, "kwargs": dict(spec.kwargs),
                    "class": type(u).__module__ + "." + type(u).__name__, "dt": float(u.dt),
                    "timestep": float(m.opt.timestep), "frame_skip": int(u.frame_skip),
                    "integrator": int(m.opt.integrator),
                    "obs_dim": int(self.observation_dim), "action_dim": int(self.action_dim),
                    "model_sha256": self._model_sha256, "model_file_sha256": henv.check_model_file(),
                    "body_mass": self._base_body_mass.tolist(), "total_mass": float(self._base_body_mass.sum()),
                    "torso_mass": float(self._base_body_mass[self._torso]),
                    "actuator_ctrlrange": m.actuator_ctrlrange.tolist(),
                    "actuator_kp": self._base_gainprm[:, 0].tolist(),
                    "actuator_kv": (-self._base_biasprm[:, 2]).tolist(),
                    "actuator_forcerange": self._base_forcerange.tolist(),
                    "geom_friction": m.geom_friction.tolist(),
                    "home_ctrl": u.home_ctrl.tolist(), "init_qpos": u.init_qpos.tolist()}

    adapter = HopperAdapter(protocol["env_id"], max_episode_steps=protocol["max_episode_steps"])
    if adapter.observation_dim != protocol["obs_dim"] or adapter.action_dim != protocol["act_dim"]:
        raise RuntimeError(f"Hopper dims {adapter.observation_dim}/{adapter.action_dim} != expected "
                           f"{protocol['obs_dim']}/{protocol['act_dim']}")
    if abs(adapter._env.unwrapped.dt - protocol["dt"]) > 1e-12:
        raise RuntimeError("Hopper control step is not the protocol dt")
    return adapter, mujoco


def ctrl_range(adapter):
    lo = np.asarray(adapter._action_low, np.float64)
    hi = np.asarray(adapter._action_high, np.float64)
    return lo, hi


def to_model_action(q_target, lo, hi):
    return 2.0 * (np.asarray(q_target, np.float64) - lo) / (hi - lo) - 1.0


def home_pose(protocol):
    return np.asarray(protocol["env_kwargs"]["home"], np.float64)


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
        m = 1.0 if rng.random() < pp["nominal_prob"] else float(rng.uniform(*pp["range"]))
        if abs(np.log(f)) >= md["log_friction"] or abs(g - 1.0) >= md["gain"] or abs(np.log(m)) >= md["log_payload"]:
            return (f, g, m)
    raise RuntimeError("could not draw a non-nominal condition")


def design_parents(protocol, counts=None, seed_offset=0):
    counts = dict(protocol["parents"] if counts is None else counts)
    ns = protocol["namespace"]
    total = sum(counts.values())
    quota = protocol["contact_quota"]
    regimes = []
    for name in ("flat", "transition", "rocked"):
        regimes += [name] * int(round(quota[name] * total))
    while len(regimes) < total:
        regimes.append("flat")
    regimes = regimes[:total]
    np.random.default_rng(derive_seed(ns, "regimes", seed_offset)).shuffle(regimes)
    designs, index = [], 0
    kinds = protocol["action_kinds"]
    for split in ("train", "val", "test"):
        for j in range(counts.get(split, 0)):
            rng = np.random.default_rng(derive_seed(ns, "values", seed_offset, index))
            A, B, C = (_draw_condition(rng, protocol) for _ in range(3))
            designs.append({
                "index": index, "parent_id": f"{protocol['parent_prefix']}-{split}-{j:04d}", "split": split,
                "contact_regime": regimes[index],
                "cohort_kinds": [kinds[index % len(kinds)], kinds[(index + 1) % len(kinds)]],
                "values": {"A": A, "B": B, "C": C},
            })
            index += 1
    return designs


def _plan_value(name, i, values, protocol):
    N = (1.0, 1.0, 1.0)
    A, B, C = (tuple(values[k]) for k in ("A", "B", "C"))
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
                float(np.clip(A[2] * np.exp(ds["log_payload"] * k), *dc["payload"])))
    raise KeyError(name)


def condition_sequences(values, protocol):
    T = protocol["T"]
    segs = list(protocol["segments"])
    out = {}
    for name in protocol["sequences"]:
        fr, gn, pl = np.ones(T), np.ones(T), np.ones(T)
        seg = np.zeros(T, np.int8)
        for i, (lo, hi) in enumerate(zip(segs[:-1], segs[1:])):
            f, g, m = _plan_value(name, i, values, protocol)
            fr[lo:hi], gn[lo:hi], pl[lo:hi] = f, g, m
            seg[lo:hi] = i
        out[name] = (fr, gn, pl, seg)
    return out


def pulse_schedule(protocol, parent_index, cohort, attempt, seed_offset=0):
    pp, T = protocol["pulse"], protocol["T"]
    rng = np.random.default_rng(derive_seed(protocol["namespace"], "pulse", seed_offset, parent_index, cohort, attempt))
    gain_f = np.ones(T)
    logf_d = np.zeros(T)
    t = pp["start"]
    while t < T:
        if rng.random() < pp["rate"]:
            L = int(rng.integers(pp["duration"][0], pp["duration"][1] + 1))
            if rng.random() < pp["gain_prob"]:
                gain_f[t:t + L] = rng.uniform(*pp["gain_factor"])
            else:
                logf_d[t:t + L] = rng.uniform(*pp["log_friction_delta"])
            t += L
        else:
            t += 1
    return gain_f, logf_d


def impulse_schedule(protocol, parent_index, cohort, attempt, seed_offset=0):
    imp, T = protocol["impulse"], protocol["T"]
    rng = np.random.default_rng(derive_seed(protocol["namespace"], "impulse", seed_offset, parent_index, cohort, attempt))
    forces = np.zeros((T, len(imp["dofs"])), np.float32)
    for t in range(imp["start"], T):
        if rng.random() < imp["rate"]:
            forces[t] = rng.normal(size=len(imp["dofs"])) * np.asarray(imp["sd"])
    return forces


def disturbance_labels(protocol, design, cohort, attempt, seed_offset=0):
    seqs = condition_sequences(design["values"], protocol)
    gain_f, logf_d = pulse_schedule(protocol, design["index"], cohort, attempt, seed_offset)
    imps = impulse_schedule(protocol, design["index"], cohort, attempt, seed_offset)
    cap = float(protocol["friction_eff_max"])
    out = {}
    for name in protocol["sequences"]:
        fr, gn, pl, seg = seqs[name]
        out[name] = {"friction": fr, "gain": gn, "payload": pl,
                     "friction_eff": np.minimum(fr * np.exp(logf_d), cap),
                     "gain_eff": gn * gain_f, "seg": seg}
    return out, imps


def bounce_offsets(rng, T, bounce, dt):
    f = float(rng.uniform(*bounce["freq_hz"]))
    amps = np.asarray([float(rng.uniform(*bounce["amp_rel"])) * a for a in bounce["amp_rad"]], np.float64)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=3)
    t = np.arange(T, dtype=np.float64) * dt
    out = amps[None, :] * np.sin(2.0 * math.pi * f * t[:, None] + phases[None, :])
    return out, {"freq_hz": f, "amp_rad": amps.tolist(), "phases": phases.tolist()}


def action_plan(protocol, kind, parent_index, cohort, attempt, lo, hi, seed_offset=0):
    from simcore.vendor.branch_actions import generate_action_plan
    ns, T = protocol["namespace"], protocol["T"]
    home = home_pose(protocol)
    bseed = derive_seed(ns, "bounce", seed_offset, parent_index, cohort, attempt)
    off, bparams = bounce_offsets(np.random.default_rng(bseed), T, protocol["bounce"], protocol["dt"])
    pseed = derive_seed(ns, "plan", seed_offset, parent_index, cohort, attempt)
    scale = np.asarray(protocol["perturbation_rad"], np.float64)
    if np.any(scale > 0):
        pert = generate_action_plan(kind, T, 3, pseed, 1.0).actions.astype(np.float64) * scale
        off = off + pert
    box = np.asarray(protocol["offset_box_rad"], np.float64)
    off = np.clip(off, -box, box)
    for a, b in protocol["low_amplitude_windows"]:
        off[a:b] *= float(protocol["low_amplitude_multiplier"])
    q = np.clip(home + off, lo, hi)
    acts = to_model_action(q, lo, hi).astype(np.float32)
    acts = np.clip(acts, -1.0, 1.0)
    return acts, {"kind": kind, "attempt": int(attempt), "bounce_seed": bseed, "plan_seed": pseed, "bounce": bparams}


def rollin_actions(protocol, parent_seed, steps, lo, hi):
    from simcore.vendor.branch_collection_min import _build_parent_rollin_actions
    r = _build_parent_rollin_actions(action_dim=3, steps=steps,
                                     seed=derive_seed(protocol["namespace"], "rollin", parent_seed),
                                     action_limit=1.0).astype(np.float64)
    q = np.clip(home_pose(protocol) + r * np.asarray(protocol["rollin_rad"], np.float64), lo, hi)
    return np.clip(to_model_action(q, lo, hi), -1.0, 1.0).astype(np.float32)


def health_check(obs, protocol):
    obs = np.atleast_2d(np.asarray(obs, np.float64))
    h = protocol["health"]
    bad_z = ~(obs[:, 0] > h["z_min"])
    bad_ang = ~(np.abs(obs[:, 1]) < h["angle_max"])
    bad_state = ~(np.abs(obs[:, 1:]) < h["state_abs_max"]).all(axis=1)
    bad = bad_z | bad_ang | bad_state
    if not bad.any():
        return None
    t = int(np.flatnonzero(bad)[0])
    reason = "z_low" if bad_z[t] else ("angle" if bad_ang[t] else "state")
    return t, reason, float(obs[t, 0] if reason == "z_low" else obs[t, 1] if reason == "angle" else np.abs(obs[t, 1:]).max())


def classify_regime(current, previous):
    sig, n_foot = current
    if n_foot < 2:
        return "rocked"
    if previous is not None and previous[0] != sig:
        return "transition"
    return "flat"


NOMINAL = {"friction_scale": 1.0, "gain_scale": 1.0, "payload_scale": 1.0}


def discover_anchor(adapter, protocol, design, seed_offset=0, start_attempt=0):
    first, last = protocol["anchor_step_min"], protocol["anchor_step_max"]
    lo, hi = ctrl_range(adapter)
    for attempt in range(int(start_attempt), protocol["max_parent_attempts"]):
        parent_seed = derive_seed(protocol["namespace"], "parent", seed_offset, design["index"], attempt)
        res = adapter.reset(parent_seed)
        rollin = rollin_actions(protocol, parent_seed, last, lo, hi)
        previous = None
        obs = res.flat_observation
        for t in range(last + 1):
            adapter.apply_absolute_dynamics(NOMINAL)
            contact = adapter.foot_contacts()
            regime = classify_regime(contact, previous)
            if (t >= first and regime == design["contact_regime"]
                    and health_check(obs[None, :], protocol) is None):
                snap = adapter.capture_integration_snapshot()
                return snap, {"parent_seed": parent_seed, "attempt": attempt, "anchor_step": t,
                              "state_sha256": snap.state_sha256}
            if t < last:
                r = adapter.step(rollin[t])
                if r.episode_end:
                    break
                obs = r.flat_observation
            previous = contact
    raise RuntimeError(f"anchor search exhausted for {design['parent_id']}")


def rollout(adapter, snapshot, actions, friction, gain, payload, impulses):
    adapter.restore_integration_snapshot(snapshot, parameters=dict(NOMINAL))
    obs = [snapshot.flat_observation.copy()]
    for t in range(len(actions)):
        want = {"friction_scale": float(friction[t]), "gain_scale": float(gain[t]), "payload_scale": float(payload[t])}
        adapter.apply_absolute_dynamics(want)
        dyn = adapter.dynamics_now()
        if any(abs(dyn[k] - v) > 1e-12 for k, v in want.items()):
            raise ValueError("requested dynamics not applied")
        imp = impulses[t] if np.any(impulses[t] != 0) else None
        res = adapter.step_with_impulse(actions[t], imp)
        if not np.array_equal(res.model_action, actions[t]):
            raise ValueError("executed action differs from plan (clipping is not allowed)")
        if res.episode_end and t < len(actions) - 1:
            raise RuntimeError("episode ended early")
        if not np.isfinite(res.flat_observation).all():
            raise FloatingPointError("non-finite observation")
        obs.append(res.flat_observation.copy())
    return np.asarray(obs, np.float32)


def _simulate_parent(adapter, protocol, d, snap, anchor, seed_offset, manifest, log):
    lo, hi = ctrl_range(adapter)
    out = []
    for cohort in (0, 1):
        accepted = None
        for attempt in range(protocol["max_plan_attempts"]):
            labels, imps = disturbance_labels(protocol, d, cohort, attempt, seed_offset)
            acts, info = action_plan(protocol, d["cohort_kinds"][cohort], d["index"], cohort, attempt, lo, hi,
                                     seed_offset)
            episodes = []
            failed = None
            for name in protocol["sequences"]:
                lab = labels[name]
                obs = rollout(adapter, snap, acts, lab["friction_eff"], lab["gain_eff"], lab["payload"], imps)
                bad = health_check(obs, protocol)
                if bad is not None:
                    failed = {"parent_id": d["parent_id"], "cohort": cohort, "attempt": attempt,
                              "kind": d["cohort_kinds"][cohort], "sequence": name, "step": bad[0],
                              "reason": bad[1], "value": bad[2], "anchor_attempt": int(anchor["attempt"])}
                    break
                episodes.append((name, obs))
            if failed is None:
                accepted = (attempt, acts, info, episodes, labels, imps)
                break
            manifest["rejections"].append(failed)
            log("hopper_reject", **failed)
        if accepted is None:
            return None, cohort
        out.append((cohort, accepted))
    return out, None


def collect(output, config=None, counts=None, log=print, seed_offset=0):
    config = default_config("hopper") if config is None else config
    protocol = config["protocol"]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    adapter, mujoco = make_adapter(protocol)
    import gymnasium
    designs = design_parents(protocol, counts, seed_offset)
    manifest = {"design": config.get("design"), "protocol": copy.deepcopy(protocol),
                "versions": {"gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__},
                "env": adapter.env_identity(), "parents": [], "replay_audit": [], "rejections": [],
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
            log("hopper_anchor_reject", **rec)
            next_anchor_attempt = int(anchor["attempt"]) + 1
        if result is None:
            last = parent_anchor_rejections[-1]
            raise RuntimeError(f"plan attempts exhausted for {d['parent_id']} cohort {last['exhausted_cohort']} "
                               f"under {len(parent_anchor_rejections)} anchor(s)")
        first = None
        cohorts = []
        for cohort, (attempt, acts, info, episodes, labels, imps) in result:
            cohorts.append({"cohort": cohort, **info})
            for name, obs in episodes:
                lab = labels[name]
                if first is None:
                    first = (obs, acts, lab, imps)
                rows["obs"].append(obs)
                rows["act"].append(acts)
                rows["friction"].append(lab["friction"].astype(np.float32))
                rows["gain"].append(lab["gain"].astype(np.float32))
                rows["payload"].append(lab["payload"].astype(np.float32))
                rows["friction_eff"].append(lab["friction_eff"].astype(np.float32))
                rows["gain_eff"].append(lab["gain_eff"].astype(np.float32))
                rows["impulse"].append(imps)
                rows["seg"].append(lab["seg"])
                rows["parent"].append(d["parent_id"])
                rows["cohort"].append(cohort)
                rows["sequence"].append(name)
                rows["plan_attempt"].append(attempt)
        o, ac, lab, im = first
        again = rollout(adapter, snap, ac, lab["friction_eff"], lab["gain_eff"], lab["payload"], im)
        ok = bool(np.array_equal(o, again))
        manifest["replay_audit"].append({"parent_id": d["parent_id"], "exact": ok})
        if not ok:
            raise RuntimeError(f"replay audit failed for {d['parent_id']}")
        manifest["parents"].append({**d, "anchor": anchor, "cohorts": cohorts,
                                    "anchor_rejections": parent_anchor_rejections})
        log("hopper_parent", parent=d["parent_id"], split=d["split"], regime=d["contact_regime"], values=d["values"],
            anchor_step=anchor["anchor_step"], anchor_attempt=anchor["attempt"],
            attempts=[c["attempt"] for c in cohorts], anchor_redraws=len(parent_anchor_rejections),
            elapsed_s=round(time.time() - t_start, 1))
    files = {}
    for split, rows in splits.items():
        path = output / f"hopper_{split}.npz"
        arrays = dict(obs=np.stack(rows["obs"]), act=np.stack(rows["act"]),
                      friction=np.stack(rows["friction"]), gain=np.stack(rows["gain"]), payload=np.stack(rows["payload"]),
                      friction_eff=np.stack(rows["friction_eff"]), gain_eff=np.stack(rows["gain_eff"]),
                      impulse=np.stack(rows["impulse"]), seg=np.stack(rows["seg"]),
                      parent=np.asarray(rows["parent"]), cohort=np.asarray(rows["cohort"], np.int8),
                      sequence=np.asarray(rows["sequence"]), plan_attempt=np.asarray(rows["plan_attempt"], np.int16))
        save_npz(path, **arrays)
        files[split] = {"path": path.name, "sha256": sha256_file(path), "episodes": len(rows["obs"]),
                        "array_sha256": {k: array_sha256(v) for k, v in arrays.items()}}
    manifest["files"] = files
    manifest["wall_s"] = round(time.time() - t_start, 1)
    write_json(output / "HOPPER_MANIFEST.json", manifest)
    return manifest


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
