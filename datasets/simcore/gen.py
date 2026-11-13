from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

BASE_PROTOCOL = {
    "env_id": "HalfCheetah-v5",
    "max_episode_steps": 1000,
    "T": 300,
    "anchor_step_min": 16,
    "anchor_step_max": 80,
    "max_parent_attempts": 4000,
    "action_limit": 0.95,
    "action_kinds": ["smooth_ar1", "piecewise", "periodic", "iid"],
    "low_amplitude_windows": [[110, 140], [230, 255]],
    "low_amplitude_multiplier": 0.15,
    "contact_quota": {"stable": 0.6, "transition": 0.2, "airborne": 0.2},
    "parents": {"train": 24, "val": 12, "test": 12},
}


def derive_seed(namespace, *parts):
    payload = json.dumps([namespace, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def save_npz(path, **arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def make_adapter(protocol):
    import mujoco
    from .vendor.gym_mujoco import GymMujocoAdapter

    class AfterWMAdapter(GymMujocoAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._base_gear = self._model.actuator_gear.copy()
            self._current_gain_scale = 1.0

        def _set_absolute_dynamics_without_forward(self, parameters):
            params = dict(parameters)
            gain = float(params.pop("gain_scale", 1.0))
            if not np.isfinite(gain) or gain <= 0:
                raise ValueError("gain_scale must be finite and positive")
            super()._set_absolute_dynamics_without_forward(params)
            self._model.actuator_gear[...] = self._base_gear
            self._model.actuator_gear[:, 0] = self._base_gear[:, 0] * gain
            self._current_gain_scale = gain

        def step_with_impulse(self, model_action, impulse=None):
            data = self._data
            data.qfrc_applied[:] = 0.0
            if impulse is not None:
                dofs = protocol["impulse"]["dofs"]
                data.qfrc_applied[dofs] = np.asarray(impulse, np.float64)
            try:
                return self.step(model_action)
            finally:
                data.qfrc_applied[:] = 0.0

        def dynamics_now(self):
            return {"friction_scale": float(self._current_friction_scale), "gain_scale": float(self._current_gain_scale),
                    "gear0": self._model.actuator_gear[:, 0].copy()}

    adapter = AfterWMAdapter(protocol["env_id"], max_episode_steps=protocol["max_episode_steps"])
    return adapter, mujoco


def action_plan(protocol, kind, parent_index, cohort, seed_offset=0):
    from .vendor.branch_actions import generate_action_plan
    seed = derive_seed(protocol["namespace"], "plan", seed_offset, parent_index, cohort)
    acts = generate_action_plan(kind, protocol["T"], 6, seed, protocol["action_limit"]).actions.copy()
    for lo, hi in protocol["low_amplitude_windows"]:
        acts[lo:hi] *= np.float32(protocol["low_amplitude_multiplier"])
    return acts.astype(np.float32), seed


def discover_anchor(adapter, protocol, design, seed_offset=0):
    from .vendor.branch_collection_min import _build_parent_rollin_actions, _classify_contact_regime
    first, last = protocol["anchor_step_min"], protocol["anchor_step_max"]
    for attempt in range(protocol["max_parent_attempts"]):
        parent_seed = derive_seed(protocol["namespace"], "parent", seed_offset, design["index"], attempt)
        adapter.reset(parent_seed)
        rollin = _build_parent_rollin_actions(action_dim=adapter.action_dim, steps=last,
                                              seed=derive_seed(protocol["namespace"], "rollin", parent_seed),
                                              action_limit=protocol["action_limit"])
        previous = None
        for t in range(last + 1):
            adapter.apply_absolute_dynamics({"friction_scale": 1.0, "gain_scale": 1.0})
            contact = adapter.get_contact_summary()
            regime = _classify_contact_regime(contact, previous)
            if t >= first and regime == design["contact_regime"]:
                snap = adapter.capture_integration_snapshot()
                return snap, {"parent_seed": parent_seed, "attempt": attempt, "anchor_step": t,
                              "state_sha256": snap.state_sha256}
            if t < last:
                if adapter.step(rollin[t]).episode_end:
                    break
            previous = contact
    raise RuntimeError(f"anchor search exhausted for {design['parent_id']}")


def rollout(adapter, snapshot, actions, friction, gain, impulses):
    adapter.restore_integration_snapshot(snapshot, parameters={"friction_scale": 1.0, "gain_scale": 1.0})
    obs = [snapshot.flat_observation.copy()]
    for t in range(len(actions)):
        adapter.apply_absolute_dynamics({"friction_scale": float(friction[t]), "gain_scale": float(gain[t])})
        dyn = adapter.dynamics_now()
        if abs(dyn["friction_scale"] - friction[t]) > 1e-12 or abs(dyn["gain_scale"] - gain[t]) > 1e-12:
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
