from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import core as C
from . import legacy as LEGACY

BASE_MODEL_CONFIG = dict(observation_dim=C.OBS, action_dim=C.ACT, num_experts=3, router_hidden_dims=(128, 128),
                         expert_hidden_dims=(256, 256, 256), alpha_q=0.5, route_probability_floor=1e-3,
                         epsilon=1e-8)
UQ_GEOMETRY = dict(observation_dim=C.OBS, action_dim=C.ACT, num_experts=3, fast_dim=32, slow_dim=32,
                   uq_hidden_dims=(128, 128), logit_step_limit=1.0, epsilon=1e-8)
UF_GEOMETRY = dict(observation_dim=C.OBS, action_dim=C.ACT, num_experts=3, fast_dim=32, fast_hidden_dims=(128, 128),
                   fast_retention=0.8, fast_innovation_limit=1.0, epsilon=1e-8)
BASE_RULE = "update_equivalent"
BASE_VARIANT = "own"
PINS_FILE = "PINS.json"
GAMMA_FILE = "GAMMA.json"
GAMMA_MAX = 1.4
CHECKPOINT_FORMAT = LEGACY.STAGE1_CHECKPOINT_FORMAT


def _registry():
    from pipeline import registry as P
    return P


def _active_dataset() -> str:
    return _registry().DATASET


def bases_dir() -> str:
    return _registry().BASES_DIR


def gamma_file(ds: str | None = None, seed=None) -> str:
    P = _registry()
    ds = ds or P.DATASET
    if ds != P.DATASET:
        raise ValueError("gamma files are resolved for the active dataset only (--dataset)")
    return os.path.join(P.GAMMA_DIR, GAMMA_FILE)


def pinned_gamma(seed: int) -> tuple:
    P = _registry()
    ds = P.DATASET
    if P.ARM_GAMMA is not None:
        val = float(P.ARM_GAMMA)
        if not 0.0 < val <= GAMMA_MAX:
            raise RuntimeError(f"invalid fixed gamma {val}")
        return val, {"gamma_fixed": val, "arm": P.ARM, "rule": "fixed by the sensitivity setting"}
    path = gamma_file(ds, seed)
    with open(path) as f:
        g = json.load(f)
    if not g.get("frozen") or g.get("dataset") != ds:
        raise RuntimeError(f"GAMMA.json is not a frozen selection for dataset {ds!r}: {path}")
    val = float(g["gamma_by_seed"][str(int(seed))])
    if not 0.0 < val <= GAMMA_MAX:
        raise RuntimeError(f"invalid pinned gamma {val}")
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return val, {"gamma_file": path, "gamma_file_sha256": sha, "rule": g.get("rule")}


def _load_pins(ds: str) -> dict:
    pins = {}
    path = os.path.join(bases_dir(), PINS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            rec = json.load(f)
        if rec.get("dataset") != ds or rec.get("rule") != BASE_RULE:
            raise RuntimeError(f"{path}: dataset/rule {rec.get('dataset')}/{rec.get('rule')} != {ds}/{BASE_RULE}")
        for k, v in rec["sha256"].items():
            pins[int(k)] = str(v)
    return pins


BASE_SHA256 = _load_pins(_active_dataset())
_SHA_CACHE: dict = {}


def pinned_base_sha256(seed, ds: str | None = None) -> str:
    ds = ds or _active_dataset()
    if ds != _active_dataset():
        raise ValueError("pins are loaded for the active dataset only (--dataset)")
    expected = BASE_SHA256.get(int(seed))
    if expected is None:
        raise ValueError(f"no pinned base for seed {seed} in dataset {_active_dataset()!r} "
                         f"(pinned: {sorted(BASE_SHA256)}; pins file: {os.path.join(bases_dir(), PINS_FILE)})")
    return expected


def _sha(path: str) -> str:
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = C.sha256_file(path)
    return _SHA_CACHE[key]


@dataclass
class FastParent:
    variant: str
    seed: int
    fast_model: object
    gamma: float
    c_e: float
    native: dict
    identity: dict = field(default_factory=dict)


def _load_base_model(path: str, expected_sha: str, device="cpu"):
    from .lib.experts.models.stage1a import Stage1ABaseWorldModel, Stage1AModelConfig
    got = _sha(path)
    if got != expected_sha:
        raise RuntimeError(f"base checkpoint sha256 mismatch {path}: {got} != {expected_sha}")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format_name") != CHECKPOINT_FORMAT or ck.get("format_version") not in (1, 2):
        raise RuntimeError("not a stage-1 base checkpoint")
    mc = {k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["model_config"].items()}
    if mc != BASE_MODEL_CONFIG:
        raise RuntimeError(f"base model_config differs from the expected geometry: {mc}")
    model = Stage1ABaseWorldModel(Stage1AModelConfig(**mc))
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.requires_grad_(False).eval()
    n = sum(p.numel() for p in model.parameters())
    if n != C.BASE_PARAMS:
        raise RuntimeError(f"base params {n} != {C.BASE_PARAMS}")
    return model.to(device), ck


def _state_sha(model) -> str:
    state = model.state_dict()
    d = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().to("cpu").contiguous()
        nb, db = name.encode(), str(t.dtype).encode()
        d.update(len(nb).to_bytes(8, "big")); d.update(nb)
        d.update(len(db).to_bytes(8, "big")); d.update(db)
        d.update(len(t.shape).to_bytes(8, "big"))
        for s in t.shape:
            d.update(int(s).to_bytes(8, "big", signed=True))
        d.update(t.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return d.hexdigest()


def _wrap_fast(base, uq_cfg: dict, uf_cfg: dict, gf_hidden=None):
    from .lib.experts.models.stage1b0 import Stage1B0ModelConfig, Stage1B0WorldModel
    from .lib.experts.models.stage1b1 import Stage1B1FastModelConfig, Stage1B1FastWorldModel
    from .lib.experts.models.stage1b2_fast_readout import (Stage1B2FastReadoutModelConfig,
                                                           Stage1B2FastReadoutWorldModel)
    q = Stage1B0WorldModel(base, Stage1B0ModelConfig(**uq_cfg))
    q.requires_grad_(False).eval()
    f = Stage1B1FastWorldModel(q, Stage1B1FastModelConfig(**uf_cfg))
    f.requires_grad_(False).eval()
    gf_kw = {} if gf_hidden is None else {"fast_readout_hidden_dims": tuple(int(h) for h in gf_hidden)}
    m = Stage1B2FastReadoutWorldModel(f, Stage1B2FastReadoutModelConfig.from_q_plus_f(f, **gf_kw))
    m.requires_grad_(False).eval()
    return m


def _check_geometry(uq_cfg: dict, uf_cfg: dict):
    for k, v in UQ_GEOMETRY.items():
        if uq_cfg[k] != v:
            raise RuntimeError(f"Uq geometry {k}={uq_cfg[k]} != {v}")
    for k, v in UF_GEOMETRY.items():
        if uf_cfg[k] != v:
            raise RuntimeError(f"UF geometry {k}={uf_cfg[k]} != {v}")
    if uq_cfg["residual_clip_norm"] != uf_cfg["residual_clip_norm"]:
        raise RuntimeError("Uq/UF residual_clip_norm differ")


def build_fast_parent(base_variant: str, seed: int, device="cpu", *, stats=None, gf_hidden=None,
                      router_gamma=None) -> FastParent:
    if base_variant != BASE_VARIANT:
        raise ValueError(f"base_variant must be {BASE_VARIANT!r}, got {base_variant!r}")
    d = os.path.join(bases_dir(), str(int(seed)))
    meta_path = os.path.join(d, "BASE.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"stage-1 base for seed {seed} is not complete: {d}")
    expected = pinned_base_sha256(seed)
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("debug"):
        raise RuntimeError("refusing a debug base")
    if meta.get("sha256") != expected:
        raise RuntimeError(f"base of seed {seed} is not the pinned base {expected}: BASE.json {meta.get('sha256')}")
    rule = (meta.get("selection_rule") or {}).get("rule")
    if rule != BASE_RULE:
        raise RuntimeError(f"base selection rule {rule!r} != {BASE_RULE!r}")
    base, ck = _load_base_model(os.path.join(d, meta["file"]), expected)
    if int(ck.get("epoch", -1)) != int(meta["epoch"]) or ck.get("validation_score") != meta["val_score"]:
        raise RuntimeError("base checkpoint epoch/VAL differ from BASE.json")
    c_e = float(meta["c_e"])
    gamma = float(meta.get("gamma", 1.0))
    if gamma != 1.0:
        raise RuntimeError("stage-1 bases use gamma 1.0")
    gamma_prov = None
    if router_gamma == "pinned":
        gamma, gamma_prov = pinned_gamma(seed)
    elif router_gamma is not None:
        raise ValueError(f"router_gamma must be None or 'pinned', got {router_gamma!r}")
    uq_cfg = dict(UQ_GEOMETRY, residual_clip_norm=c_e)
    uf_cfg = dict(UF_GEOMETRY, residual_clip_norm=c_e)
    from .lib.experts.models.stage1b0 import Stage1B0ModelConfig
    from .lib.experts.models.stage1a import Stage1AModelConfig
    native_q = Stage1B0ModelConfig.from_stage1a(Stage1AModelConfig(**BASE_MODEL_CONFIG), residual_clip_norm=c_e)
    for k, v in uq_cfg.items():
        if getattr(native_q, k) != v:
            raise RuntimeError(f"from_stage1a geometry differs at {k}")
    nb = ck["normalizer"]
    native = {"obs_mean": np.asarray(nb["observation_mean"], np.float64),
              "obs_std": np.asarray(nb["observation_std"], np.float64),
              "act_mean": np.asarray(nb["action_mean"], np.float64),
              "act_std": np.asarray(nb["action_std"], np.float64)}
    if stats is not None:
        for k, sk in (("obs_mean", "obs_mean"), ("obs_std", "obs_std"), ("act_mean", "act_mean"), ("act_std", "act_std")):
            if not np.array_equal(native[k], np.asarray(getattr(stats, sk), np.float64)):
                raise RuntimeError(f"base normaliser {k} differs from the TRAIN stats")
    identity = {"variant": BASE_VARIANT, "dataset": _active_dataset(), "seed": int(seed),
                "file": os.path.join(d, meta["file"]), "sha256": expected, "pinned": True,
                "stage1a_state_sha256": meta.get("stage1a_state_sha256"),
                "base_epoch": meta.get("epoch"), "base_val_score": meta.get("val_score"),
                "selection_rule": rule, "epochs_run": meta.get("epochs_run"),
                "c_e_calibration": meta.get("c_e_calibration"), "router_gamma": gamma_prov}
    fast = _wrap_fast(base, uq_cfg, uf_cfg, gf_hidden)
    identity.update(state_sha256=_state_sha(base), gamma=gamma, c_e=c_e, uq_config=uq_cfg, uf_config=uf_cfg,
                    base_params=C.BASE_PARAMS, gf_hidden=None if gf_hidden is None else [int(h) for h in gf_hidden])
    if identity["state_sha256"] != identity["stage1a_state_sha256"]:
        raise RuntimeError("base state sha256 differs from BASE.json stage1a_state_sha256")
    fast = fast.to(device)
    return FastParent(BASE_VARIANT, int(seed), fast, gamma, c_e, native, identity)


class FrameMap(nn.Module):

    def __init__(self, parent: FastParent, stats):
        super().__init__()
        tm = {k: np.asarray(getattr(stats, k), np.float64) for k in ("obs_mean", "obs_std", "act_mean", "act_std")}
        nm = parent.native
        self.identity = bool(all(np.array_equal(tm[k], nm[k]) for k in tm))
        f64 = lambda x: torch.as_tensor(np.asarray(x, np.float64))
        self.register_buffer("obs_scale", f64(tm["obs_std"] / nm["obs_std"]))
        self.register_buffer("obs_shift", f64((tm["obs_mean"] - nm["obs_mean"]) / nm["obs_std"]))
        self.register_buffer("act_scale", f64(tm["act_std"] / nm["act_std"]))
        self.register_buffer("act_shift", f64((tm["act_mean"] - nm["act_mean"]) / nm["act_std"]))
        self.register_buffer("own_std", f64(nm["obs_std"]))
        self.register_buffer("common_std", f64(tm["obs_std"]))
        self.raw_scale = float(np.mean(tm["obs_std"] ** 2))

    def obs_to_native(self, z):
        if self.identity:
            return z
        return (z.double() * self.obs_scale + self.obs_shift).to(z.dtype)

    def act_to_native(self, a):
        if self.identity:
            return a
        return (a.double() * self.act_scale + self.act_shift).to(a.dtype)

    def obs_to_train(self, zn):
        if self.identity:
            return zn
        return ((zn.double() - self.obs_shift) / self.obs_scale).to(zn.dtype)


@contextlib.contextmanager
def gamma_hook(model, gamma: float):
    from .lib.experts.models.stage1a import Stage1ABaseWorldModel
    if gamma == 1.0:
        yield
        return
    if not 0 <= gamma <= 1:
        raise ValueError("invalid fixed gamma")
    bases = [m for m in model.modules() if isinstance(m, Stage1ABaseWorldModel)]
    if len(bases) != 1:
        raise ValueError("exactly one native Router required")
    router = bases[0].router
    ref = next(router.parameters())
    factor = torch.tensor(gamma, device=ref.device, dtype=ref.dtype)
    handle = router.register_forward_hook(lambda _m, _a, out: out * factor)
    try:
        yield
    finally:
        handle.remove()
