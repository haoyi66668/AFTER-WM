from __future__ import annotations

import torch

from .. import core as C

NAME = "fast"
FAMILY = "afterwm"
GF_HIDDEN_BY_DIMS = {(17, 6): {"fast_native": (128, 128), "afterwm_eq": (213, 213)},
                     (14, 7): {"fast_native": (128, 128), "afterwm_eq": (219, 219)},
                     (12, 6): {"fast_native": (128, 128), "afterwm_eq": (225, 225)},
                     (11, 3): {"fast_native": (128, 128), "afterwm_eq": (231, 231)},
                     (12, 3): {"fast_native": (128, 128), "afterwm_eq": (229, 229)}}
GF_HIDDEN = GF_HIDDEN_BY_DIMS[(C.OBS, C.ACT)]
BUDGET_TARGET = {"fast_native": C.BASE_PARAMS + C.FAST_ACTIVE, "afterwm_eq": 700_000}
ARCH = "native_F_A_pure_fast_control"


def gf_params(h: int) -> int:
    return C.gf_param_count(h)


def solve_gf_hidden(target: int = 700_000, lo: int = 16, hi: int = 1024) -> int:
    return min(range(lo, hi + 1), key=lambda h: (abs(budget(h)["total"] - int(target)), h))


def budget(h: int) -> dict:
    gf = gf_params(h)
    pre = C.FAST_PRE - gf_params(128) + gf
    active = C.FAST_ACTIVE - gf_params(128) + gf
    return {"gf": gf, "original_target": pre, "target": active, "total": C.BASE_PARAMS + active}


def default_config(budget_class: str = "fast_native", **variant) -> dict:
    unknown = set(variant) - {"base", "cuda_graph", "construct_device", "router_gamma"}
    if unknown:
        raise ValueError(f"unknown {NAME} variant keys: {sorted(unknown)}")
    base = variant.get("base", "own")
    if base != "own":
        raise ValueError("base must be own")
    if budget_class not in GF_HIDDEN:
        raise ValueError(f"{NAME}: budget_class must be one of {sorted(GF_HIDDEN)}")
    h = GF_HIDDEN[budget_class][0]
    b = budget(h)
    target = BUDGET_TARGET[budget_class]
    assert abs(b["total"] - target) <= 0.01 * target, b
    tag = {"fast_native": "fast_native", "afterwm_eq": "afterwm_fast_only"}[budget_class]
    cfg = {"name": NAME, "family": FAMILY, "base": base, "budget_class": budget_class,
           "gf_hidden": list(GF_HIDDEN[budget_class]), "budget_total": b["total"], "budget_target": target,
           "original_target": b["original_target"], "active_target": b["target"], "lr": 1e-4,
           "betas": (0.9, 0.999), "grad_clip": 1.0, "optimizer": "Adam", "sigma": 0.5,
           "horizons": list(C.HORIZONS), "context": 64, "cuda_graph": True, "construct_device": "cpu",
           "model_id": f"{tag}{'' if variant.get('router_gamma') else '_gamma1'}"}
    if variant.get("router_gamma") is not None:
        if variant["router_gamma"] != "pinned":
            raise ValueError("router_gamma must be 'pinned'")
        cfg["router_gamma"] = "pinned"
    cfg.update({k: v for k, v in variant.items() if k in ("cuda_graph", "construct_device")})
    return cfg


def fast_restriction(model, names):
    if getattr(model, "family", None) != "F":
        raise TypeError("fast-only restriction requires the native F family")
    groups = model.learning_modules()
    if set(groups) != {"Uq", "UF", "GF"}:
        raise AssertionError(f"fast-only model must expose exactly Uq/UF/GF, got {sorted(groups)}")
    if len(model.core.slow_readouts) or len(model.core.consolidators) or len(model.core.slow_updaters):
        raise AssertionError("fast-only model must not keep slow modules")
    if len(model.static_readouts):
        raise AssertionError("fast-only model must not keep static heads")
    return ARCH, []


class FastModel(C.MemoryModel):
    family = "fast"

    def init_state(self, B, ref):
        K, D = C.K, C.FAST_DIM
        q = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=ref.device)
        F = torch.zeros((B, K, D), dtype=torch.float32, device=ref.device)
        return (q, F)

    def observe_fast(self, state, z, a, zn):
        q, F = state
        fp = self.fp
        joint = torch.cat((z, a), dim=-1)
        log_p, p = fp.route(joint, q)
        z_hat = fp.experts_zhat(z, a, F, joint)
        return fp.fast_update(z, a, zn, p, log_p, F, z_hat)

    def predict_fast(self, state, z, af):
        q, F = state
        fp = self.fp
        log_p, _ = fp.route(torch.cat((z, af[:, 0]), dim=-1), q)
        paths = []
        for k in range(C.K):
            st, Fk = z, F[:, k]
            path = []
            for h in range(af.shape[1]):
                st = fp.expert_zhat(st, af[:, h], Fk, k)
                path.append(st)
                Fk = fp.retention * Fk
            paths.append(torch.stack(path, 1))
        return torch.stack(paths, -2), log_p

    def state_dims(self):
        return {"q": 3, "F": 96, "H_S_identically_zero": 192, "total_live": 99}


def build(cfg: dict, seed: int, device, stats) -> FastModel:
    from ..lib.residual import model as residual_model
    from ..lib import slow_core
    from ..lib.experts.models.stage1a import BaseExpertPrediction
    parent, fmap = C.build_parent_and_frame(cfg, seed, stats)
    gf = tuple(int(h) for h in cfg["gf_hidden"])
    got = tuple(parent.fast_model.config.fast_readout_hidden_dims)
    if got != gf:
        raise AssertionError(f"G^F hidden dims {got} != {gf}")

    def core_factory(fast):
        return slow_core.SlowCore(fast, slow_core.SlowConfig())

    def factory(s):
        return residual_model.Candidate(parent.fast_model, core_factory, BaseExpertPrediction, "F_A", int(s))

    core, report = C.construct_restricted(factory, int(seed), NAME, C.normalizers_for(parent, stats),
                                          original_target=int(cfg["original_target"]),
                                          target=int(cfg["active_target"]), slow_restriction=fast_restriction)
    report["pure_fast_control"] = {"gf_hidden": list(gf), "centred_gf": True, "static_head": False,
                                   "slow_state": False}
    model = FastModel(core, parent, fmap, cfg, report)
    model.to(device)
    rep = model.parameter_report()
    if rep["total"] != cfg["budget_total"]:
        raise AssertionError(f"total params {rep['total']} != {cfg['budget_total']}")
    return model


def make_optimizer(model, cfg):
    return C.adam_optimizer(model, cfg)


def train_step(model, batch: dict, step: int, cfg: dict):
    return C.generic_train_step(model, batch, step, cfg)


def forecaster(model, cfg) -> C.MemoryForecaster:
    return C.MemoryForecaster(model, cfg.get("model_id", NAME), cfg["budget_class"])


C.register_family(NAME, __import__(__name__, fromlist=["*"]))
