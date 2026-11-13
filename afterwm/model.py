from __future__ import annotations

import torch

from . import core as C
from . import legacy as LEGACY

NAME = "afterwm"
FAMILY = "afterwm"
NATIVE_WIDTHS = (149, 202, 128, 130, 32)
WIDTHS_BY_DIMS = {(17, 6): {"afterwm_eq": (86, 116, 73, 75, 18)},
                  (14, 7): {"afterwm_eq": (94, 127, 80, 82, 20)},
                  (12, 6): {"afterwm_eq": (101, 137, 87, 88, 22)},
                  (11, 3): {"afterwm_eq": (108, 146, 92, 94, 23)},
                  (12, 3): {"afterwm_eq": (105, 142, 90, 91, 22)}}
WIDTHS = WIDTHS_BY_DIMS[(C.OBS, C.ACT)]
_NATIVE_SIGNATURES = ((55, (149, 202), 68), (58, (128, 130), 17), (55, (32,), 3))


def slow_pre(m1, m2, r1, r2, f1, obs: int = C.OBS, act: int = C.ACT) -> int:
    d_in = obs + act + C.FAST_DIM
    meas = (d_in + 1) * m1 + (m1 + 1) * m2 + 4 * obs * (m2 + 1)
    read = (d_in + 4) * r1 + (r1 + 1) * r2 + obs * (r2 + 1)
    filt = (d_in + 1) * f1 + 3 * (f1 + 1)
    return 3 * (meas + read + filt)


def budget(widths) -> dict:
    sp = slow_pre(*widths)
    removed = 3 * C.OBS
    active = C.FAST_ACTIVE + sp - removed
    return {"slow_pre": sp, "slow_active": sp - removed, "original_target": C.FAST_PRE + sp, "target": active,
            "total": C.BASE_PARAMS + active}


def solve_uniform_widths(total_fn, target: int, native=NATIVE_WIDTHS) -> tuple:
    best = None
    for i in range(100, 3001):
        sc = i / 1000
        w = tuple(int(round(x * sc)) for x in native)
        t = int(total_fn(w))
        key = (abs(t - int(target)), sc)
        if best is None or key < best[0]:
            best = (key, w, t, sc)
    return best[1], best[2], best[3]


def default_config(budget_class: str = "afterwm_eq", **variant) -> dict:
    unknown = set(variant) - {"base", "cuda_graph", "construct_device", "router_gamma"}
    if unknown:
        raise ValueError(f"unknown {NAME} variant keys: {sorted(unknown)}")
    base = variant.get("base", "own")
    if base != "own":
        raise ValueError("base must be own")
    if budget_class not in WIDTHS:
        raise ValueError("budget_class must be afterwm_eq")
    w = WIDTHS[budget_class]
    b = budget(w)
    target = 700_000
    assert abs(b["total"] - target) <= 0.01 * target, b
    cfg = {"name": NAME, "family": FAMILY, "base": base, "budget_class": budget_class, "widths": list(w),
           "budget_total": b["total"], "budget_target": target, "original_target": b["original_target"],
           "active_target": b["target"], "lr": 1e-4, "betas": (0.9, 0.999), "grad_clip": 1.0,
           "optimizer": "Adam", "sigma": 0.5, "horizons": list(C.HORIZONS), "context": 64, "cuda_graph": True,
           "construct_device": "cpu", "model_id": f"afterwm{'' if variant.get('router_gamma') else '_gamma1'}"}
    if variant.get("router_gamma") is not None:
        if variant["router_gamma"] != "pinned":
            raise ValueError("router_gamma must be 'pinned'")
        cfg["router_gamma"] = "pinned"
    cfg.update({k: v for k, v in variant.items() if k in ("cuda_graph", "construct_device")})
    return cfg


def afterwm_class(widths):
    from .lib.memory import networks as T
    widths = tuple(int(x) for x in widths)
    if C.NATIVE_DIMS and widths == NATIVE_WIDTHS:
        return T.AfterWMNativeDifferentiable, None
    sp = slow_pre(*widths)
    m1, m2, r1, r2, f1 = widths
    d_in = C.OBS + C.ACT + C.FAST_DIM
    mapping = {(55, (149, 202), 68): (d_in, (m1, m2), 4 * C.OBS),
               (58, (128, 130), C.OBS): (d_in + 3, (r1, r2), C.OBS),
               (55, (32,), 3): (d_in, (f1,), 3)}
    orig = T._mlp

    def _mlp(din, hidden, dout):
        a, h, b = mapping[(din, tuple(hidden), dout)]
        return orig(a, h, b)

    attrs = {"TRAINABLE_SLOW": sp, "TRAINABLE_TOTAL": C.FAST_PRE + sp, "WIDTHS": widths}
    if not C.NATIVE_DIMS:
        attrs.update(OBS=C.OBS, ACT=C.ACT)
    cls = type("AfterWMNativeDifferentiableWidths", (T.AfterWMNativeDifferentiable,), attrs)
    return cls, (T, _mlp)


class AfterWMModel(C.MemoryModel):
    family = "afterwm"

    def init_state(self, B, ref):
        K, D = C.K, C.FAST_DIM
        q = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=ref.device)
        F = torch.zeros((B, K, D), dtype=torch.float32, device=ref.device)
        c = torch.zeros(B, K, 3, device=ref.device, dtype=ref.dtype)
        return (q, F, c, torch.ones_like(c))

    def observe_fast(self, state, z, a, zn):
        q, F, c, P = state
        fp, m = self.fp, self.core
        joint = torch.cat((z, a), dim=-1)
        log_p, p = fp.route(joint, q)
        z_hat = fp.experts_zhat(z, a, F, joint)
        q_new, F_new = fp.fast_update(z, a, zn, p, log_p, F, z_hat)
        err = zn[:, None, :] - z_hat
        cs, ps = [], []
        for k in range(C.K):
            Fk = F[:, k]
            out = m.measurement[k](torch.cat((z, a, Fk), dim=-1))
            J = torch.tanh(out[..., :3 * C.OBS].reshape(*out.shape[:-1], C.OBS, 3))
            bias = out[..., 3 * C.OBS:]
            y = err[:, k] - bias
            r = 0.05 + torch.nn.functional.softplus(m.filter[k](torch.cat((z, a, Fk), dim=-1)))
            Pk = P[:, k].clamp_min(1e-4)
            JPJ = (J.square() * Pk[:, None, :]).sum(dim=1)
            S = JPJ + r + 1e-3
            innov = (J * y[:, :, None]).sum(dim=1)
            gain = Pk / S.clamp_min(1e-3)
            c_next = c[:, k] + gain * innov / float(C.OBS)
            c_next = torch.tanh(c_next)
            p_next = ((1.0 - (Pk[:, None, :] * J.square() / S[:, None, :]).mean(dim=1)).clamp(0.02, 1.0) * Pk + 0.01)
            cs.append(c_next)
            ps.append(p_next)
        return (q_new, F_new, torch.stack(cs, dim=1), torch.stack(ps, dim=1))

    def predict_fast(self, state, z, af):
        q, F, c, P = state
        fp, m = self.fp, self.core
        log_p, _ = fp.route(torch.cat((z, af[:, 0]), dim=-1), q)
        paths = []
        for k in range(C.K):
            st, Fk, ck = z, F[:, k], c[:, k]
            zero_c = torch.zeros_like(ck)
            pp = []
            for h in range(af.shape[1]):
                a_h = af[:, h]
                z_hat = fp.expert_zhat(st, a_h, Fk, k)
                x = torch.cat((st, a_h, Fk, ck), dim=-1)
                x0 = torch.cat((st, a_h, Fk, zero_c), dim=-1)
                delta = m.readout[k](x) - m.readout[k](x0)
                st = z_hat + delta
                pp.append(st)
                Fk = fp.retention * Fk
            paths.append(torch.stack(pp, 1))
        return torch.stack(paths, -2), log_p

    def state_dims(self):
        return {"q": 3, "F": 96, "c": 9, "P": 9, "total_live": 117}


def build(cfg: dict, seed: int, device, stats) -> AfterWMModel:
    parent, fmap = C.build_parent_and_frame(cfg, seed, stats)
    cls, override = afterwm_class(cfg["widths"])

    def factory(s):
        if override is None:
            return cls(parent.fast_model, int(s))
        T, _mlp = override
        with C.override_global(T, "_mlp", _mlp):
            return cls(parent.fast_model, int(s))

    core, report = C.construct_restricted(factory, int(seed), LEGACY.CONSTRUCTOR_STREAM, C.normalizers_for(parent, stats),
                                          original_target=int(cfg["original_target"]),
                                          target=int(cfg["active_target"]))
    model = AfterWMModel(core, parent, fmap, cfg, report)
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
