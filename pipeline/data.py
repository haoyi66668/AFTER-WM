from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from . import registry as P

_SHA_CACHE: dict = {}


def _sha256(path: str) -> str:
    if path not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[path] = h.hexdigest()
    return _SHA_CACHE[path]


def split_path(split: str, data_dir: str = P.DATA_DIR) -> str:
    extra = getattr(P, "ACTIVE_EXTRA_SPLITS", {})
    if split not in P.SPLIT_FILES and split in extra:
        if data_dir != P.DATA_DIR:
            raise ValueError(f"{split} is an evaluation-only split of dataset {P.DATASET!r} given by absolute "
                             f"path; a data_dir override ({data_dir}) is not supported")
        return extra[split]["path"]
    return os.path.join(data_dir, P.SPLIT_FILES[split])


def expected_sha256(split: str) -> str:
    if split in P.SPLIT_SHA256:
        return P.SPLIT_SHA256[split]
    return P.ACTIVE_EXTRA_SPLITS[split]["sha256"]


def _checked_npz(split: str, data_dir: str):
    path = split_path(split, data_dir)
    got = _sha256(path)
    exp = expected_sha256(split)
    if got != exp:
        raise RuntimeError(f"{split}: sha256 mismatch {got} != {exp}")
    return np.load(path, allow_pickle=False)


def load_visible(split: str, data_dir: str = P.DATA_DIR):
    with _checked_npz(split, data_dir) as z:
        obs = np.asarray(z["obs"], dtype=np.float32)
        act = np.asarray(z["act"], dtype=np.float32)
    assert obs.ndim == 3 and obs.shape[1:] == (P.T_OBS, P.OBS_DIM), obs.shape
    assert act.ndim == 3 and act.shape[1:] == (P.T_ACT, P.ACT_DIM), act.shape
    assert obs.shape[0] == act.shape[0]
    return obs, act


def load_analysis_labels(split: str, data_dir: str = P.DATA_DIR) -> dict:
    out = {}
    with _checked_npz(split, data_dir) as z:
        for k in P.ANALYSIS_KEYS:
            if k in z.files:
                out[k] = np.asarray(z[k]).astype(str)
    return out


@dataclass
class TrainStats:
    obs_mean: list
    obs_std: list
    act_mean: list
    act_std: list

    @staticmethod
    def fit(obs: np.ndarray, act: np.ndarray) -> "TrainStats":
        o = obs.reshape(-1, P.OBS_DIM).astype(np.float64)
        a = act.reshape(-1, P.ACT_DIM).astype(np.float64)
        return TrainStats(
            obs_mean=o.mean(0).tolist(),
            obs_std=np.maximum(o.std(0), 1e-6).tolist(),
            act_mean=a.mean(0).tolist(),
            act_std=np.maximum(a.std(0), 1e-6).tolist(),
        )

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def save(self, path: str) -> None:
        d = asdict(self)
        d["sha256"] = self.sha256()
        with open(path, "w") as f:
            json.dump(d, f, indent=1)

    @staticmethod
    def load(path: str) -> "TrainStats":
        with open(path) as f:
            d = json.load(f)
        sha = d.pop("sha256", None)
        s = TrainStats(**d)
        if sha is not None and sha != s.sha256():
            raise RuntimeError("TrainStats sha mismatch")
        return s

    def arrays(self):
        f = lambda x: np.asarray(x, dtype=np.float32)
        return f(self.obs_mean), f(self.obs_std), f(self.act_mean), f(self.act_std)

    def norm_obs(self, obs):
        m, s, _, _ = self.arrays()
        return ((obs - m) / s).astype(np.float32)

    def norm_act(self, act):
        _, _, m, s = self.arrays()
        return ((act - m) / s).astype(np.float32)

    def denorm_obs(self, z):
        m, s, _, _ = self.arrays()
        return z * s + m


def train_stats(data_dir: str = P.DATA_DIR) -> TrainStats:
    obs, act = load_visible("train", data_dir)
    return TrainStats.fit(obs, act)


def eval_queries(n_episodes: int):
    e = np.repeat(np.arange(n_episodes), len(P.EVAL_STARTS))
    t = np.tile(np.asarray(P.EVAL_STARTS), n_episodes)
    return e.astype(np.int64), t.astype(np.int64)


def windows(obs_n: np.ndarray, act_n: np.ndarray, e: np.ndarray, t: np.ndarray, horizon: int = P.MAX_H) -> dict:

    e = np.asarray(e)
    t = np.asarray(t)
    assert np.all(t >= P.CONTEXT) and np.all(t + horizon <= P.T_OBS - 1)
    o_idx = (t[:, None] - P.CONTEXT) + np.arange(P.CONTEXT + 1 + horizon)[None, :]
    a_idx = (t[:, None] - P.CONTEXT) + np.arange(P.CONTEXT + horizon)[None, :]
    win_obs = obs_n[e[:, None], o_idx]
    win_act = act_n[e[:, None], a_idx]
    return {
        "ctx_obs": win_obs[:, : P.CONTEXT + 1],
        "ctx_act": win_act[:, : P.CONTEXT],
        "fut_act": win_act[:, P.CONTEXT:],
        "fut_obs": win_obs[:, P.CONTEXT + 1:],
        "win_obs": win_obs,
        "win_act": win_act,
    }


def context_keys(ctx_obs: np.ndarray, ctx_act: np.ndarray) -> np.ndarray:
    keys = np.empty(ctx_obs.shape[0], dtype=np.int64)
    for i in range(ctx_obs.shape[0]):
        h = hashlib.sha256(np.ascontiguousarray(ctx_obs[i]).tobytes() + np.ascontiguousarray(ctx_act[i]).tobytes())
        keys[i] = int.from_bytes(h.digest()[:8], "little", signed=True)
    return keys


def make_schedule(seed: int, n_train: int, steps: int = P.MAX_STEPS, batch: int = P.BATCH):
    rng = np.random.default_rng(seed)
    ep = np.stack([rng.choice(n_train, size=batch, replace=False) for _ in range(steps)]).astype(np.int64)
    tt = rng.integers(P.TRAIN_T_MIN, P.TRAIN_T_MAX + 1, size=(steps, batch)).astype(np.int64)
    return ep, tt
