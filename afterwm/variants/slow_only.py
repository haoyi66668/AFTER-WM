from __future__ import annotations

import torch

from .. import core as C
from .. import legacy as LEGACY
from .. import model as T

NAME = "slow"
FAMILY = "afterwm"
WIDTHS_BY_DIMS = {(17, 6): {"afterwm_eq": (154, 209, 133, 135, 33)},
                  (14, 7): {"afterwm_eq": (161, 218, 138, 141, 35)},
                  (12, 6): {"afterwm_eq": (167, 226, 143, 146, 36)},
                  (11, 3): {"afterwm_eq": (172, 234, 148, 150, 37)},
                  (12, 3): {"afterwm_eq": (169, 230, 146, 148, 36)}}
WIDTHS = WIDTHS_BY_DIMS[(C.OBS, C.ACT)]
BUDGET_TARGET = {"afterwm_eq": 700_000}
CONSTRUCTOR_STREAM = LEGACY.CONSTRUCTOR_STREAM
ARCH = "neural_LF_AFTERWM_slow_only"
OBS, ACT, K, FAST, SLOW = C.OBS, C.ACT, 3, 32, 32
UQ_HIDDEN = 128
UQ_PRE = C.uq_pre_count()
UQ_F_START = OBS + ACT + K
UQ_REMOVED_COLS = list(range(UQ_F_START, UQ_F_START + 3 * K * FAST))
UQ_ACTIVE = UQ_PRE - UQ_HIDDEN * len(UQ_REMOVED_COLS) - K
F_COLS = list(range(OBS + ACT, OBS + ACT + FAST))
KEPT = {"measurement": list(range(OBS + ACT)), "filter": list(range(OBS + ACT)),
        "readout": list(range(OBS + ACT)) + [OBS + ACT + FAST + i for i in range(3)],
        "Uq": list(range(UQ_F_START)) + list(range(UQ_F_START + 3 * K * FAST, UQ_F_START + 3 * K * FAST + K * OBS))}


def budget(widths) -> dict:
    m1, m2, r1, r2, f1 = (int(x) for x in widths)
    sp = T.slow_pre(m1, m2, r1, r2, f1)
    slow_active = sp - 3 * OBS - 3 * FAST * (m1 + r1 + f1)
    active = UQ_ACTIVE + slow_active
    return {"slow_pre": sp, "slow_active": slow_active, "uq_active": UQ_ACTIVE,
            "original_target": C.FAST_PRE + sp, "target": active, "total": C.BASE_PARAMS + active}


def config_for_widths(widths, budget_class: str = "afterwm_eq", base: str = "own", check_budget: bool = True, **extra):
    w = [int(x) for x in widths]
    b = budget(w)
    target = BUDGET_TARGET.get(budget_class)
    if check_budget and (target is None or abs(b["total"] - target) > 0.01 * target):
        raise AssertionError(f"slow: total {b['total']} outside the {budget_class} budget")
    tag = "afterwm_slow_only"
    cfg = {"name": NAME, "family": FAMILY, "base": base, "budget_class": budget_class, "widths": w,
           "budget_total": b["total"], "budget_target": target, "original_target": b["original_target"],
           "active_target": b["target"], "lr": 1e-4, "betas": (0.9, 0.999), "grad_clip": 1.0, "optimizer": "Adam",
           "sigma": 0.5, "horizons": list(C.HORIZONS), "context": 64, "cuda_graph": True, "construct_device": "cpu",
           "model_id": f"{tag}_gamma1", "architecture": ARCH}
    cfg.update(extra)
    return cfg


def default_config(budget_class: str = "afterwm_eq", **variant) -> dict:
    unknown = set(variant) - {"base", "cuda_graph", "construct_device", "router_gamma"}
    if unknown:
        raise ValueError(f"unknown {NAME} variant keys: {sorted(unknown)}")
    base = variant.get("base", "own")
    if base != "own":
        raise ValueError("base must be own")
    if budget_class not in WIDTHS:
        raise ValueError(f"{NAME}: budget_class must be one of {sorted(WIDTHS)}")
    extra = {k: v for k, v in variant.items() if k in ("cuda_graph", "construct_device")}
    cfg = config_for_widths(WIDTHS[budget_class], budget_class, base, **extra)
    if variant.get("router_gamma") is not None:
        if variant["router_gamma"] != "pinned":
            raise ValueError("router_gamma must be 'pinned'")
        cfg["router_gamma"] = "pinned"
        cfg["model_id"] = "afterwm_slow_only"
    return cfg


class RemovedFastBranch(torch.nn.Module):

    def __init__(self, what: str):
        super().__init__()
        self.what = what

    def forward(self, *args, **kwargs):
        raise RuntimeError(f"{self.what} was removed: the pure-slow control has no fast branch")


def slow_class(widths):
    cls, override = T.afterwm_class(widths)
    if override is None:
        raise AssertionError("the pure-slow control must not use the native widths")

    class AfterWMSlowOnly(cls):
        _fast_removed = False

        def learning_modules(self):
            groups = super().learning_modules()
            if self._fast_removed:
                groups = {g: m for g, m in groups.items() if g not in ("UF", "GF")}
            return groups

    AfterWMSlowOnly.__name__ = "AfterWMSlowOnly"
    return AfterWMSlowOnly, override


def slow_restriction(model, names):
    fc = C._fc()
    arch, rows = C.default_slow_restriction(model, names)
    if arch != "neural_LF_AFTERWM":
        raise AssertionError(f"expected the AFTER-WM architecture, got {arch}")
    groups = model.learning_modules()
    for g in ("UF", "GF"):
        if g not in groups:
            raise AssertionError(f"missing {g} before the fast-branch removal")
    n_uf = sum(p.numel() for p in groups["UF"].parameters())
    n_gf = sum(p.numel() for p in groups["GF"].parameters())
    fm = model.core
    uf_name, gf_name = names[id(groups["UF"])], names[id(groups["GF"])]
    fm.accepted_model.fast_updater = RemovedFastBranch("U_F")
    fm.fast_readout_experts = torch.nn.ModuleList()
    model._fast_removed = True
    rows.append({"group": "UF", "layer": uf_name, "method": "delete the fast writer U_F (the model has no fast state)",
                 "removed_trainable_scalars": n_uf, "removal_forward_invariant": True})
    rows.append({"group": "GF", "layer": gf_name, "method": "delete the fast readout G^F (the model has no fast state)",
                 "removed_trainable_scalars": n_gf, "removal_forward_invariant": True})
    for kind in ("measurement", "filter", "readout"):
        nets = getattr(model, kind)
        for k in range(len(nets)):
            first = nets[k][0]
            if type(first) is not torch.nn.Linear:
                raise AssertionError(f"{kind}[{k}] first layer must be an ordinary Linear before packing")
            packed = fc.pack_linear_input(first, fixed_zero_indices=F_COLS)
            if packed.input_kept_indices.tolist() != KEPT[kind]:
                raise AssertionError(f"unexpected kept columns for {kind}")
            nets[k][0] = packed
            rows.append({"expert": k, "layer": names[id(first)], "method": "remove weights on the absent F inputs",
                         "zero_input_coordinates": F_COLS, "removed_trainable_scalars": first.out_features * FAST,
                         "removal_forward_invariant": True})
    return ARCH, rows


def slow_common_nulls(model):
    fc = C._fc()
    uq = model.learning_modules()["Uq"]
    cfg = uq.config
    network = fc._writer_network(uq)
    first = network[0]
    geom = (cfg.observation_dim, cfg.action_dim, cfg.num_experts, cfg.fast_dim, cfg.slow_dim)
    expected_in = OBS + ACT + K + 3 * K * FAST + K * OBS
    if type(first) is not torch.nn.Linear or first.in_features != expected_in or geom != (OBS, ACT, K, FAST, SLOW) \
            or first.out_features != UQ_HIDDEN:
        raise AssertionError("U_q input geometry/order differs from the reviewed z,a,p,F,S,H,e")
    packed = fc.pack_linear_input(first, fixed_zero_indices=UQ_REMOVED_COLS)
    if packed.input_kept_indices.tolist() != KEPT["Uq"]:
        raise AssertionError("unexpected U_q kept columns")
    network[0] = packed
    rows = [{"group": "Uq", "layer": "Uq first", "method": "remove weights on permanent zero F/S/H inputs",
             "original_input_width": first.in_features, "active_input_width": packed.weight.shape[1],
             "zero_input_coordinates": UQ_REMOVED_COLS, "removed_trainable_scalars": UQ_HIDDEN * len(UQ_REMOVED_COLS),
             "removal_forward_invariant": True}]
    last = network[-1]
    if type(last) is not torch.nn.Linear or last.bias is None or last.bias.numel() != K:
        raise AssertionError("unexpected centred U_q final bias")
    last.register_parameter("bias", None)
    rows.append({"group": "Uq", "layer": "Uq last", "method": "delete centered writer final bias",
                 "removed_trainable_scalars": K, "removal_forward_invariant": True,
                 "reason": "raw_with_error - raw_without_error cancels final bias"})
    return rows


def _kept(net, x_kept):
    first = net[0]
    h = torch.nn.functional.linear(x_kept, first.weight, first.bias)
    for layer in list(net)[1:]:
        h = layer(h)
    return h


class SlowPath(C.FastPath):

    def __init__(self, fm, gamma: float):
        self.fm = fm
        self.base = fm.base_model
        bc = self.base.config
        self.K, self.floor, self.alpha_q, self.eps_base = bc.num_experts, bc.route_probability_floor, bc.alpha_q, bc.epsilon
        self.router = self.base.router
        self.g0 = list(self.base.g0_experts)
        self.gf = []
        qm = fm.q_only_model
        self.qcfg = qm.config
        seq = qm.belief_updater.network.network
        first = seq[0]
        if type(first).__name__ != "PackedInputLinear" or first.input_kept_indices.tolist() != KEPT["Uq"]:
            raise AssertionError("pure-slow U_q must be packed to its z,a,p,e columns")
        self.uq_first = first
        self.uq_rest = list(seq)[1:]
        self.uf_module = None
        self.retention = None
        self.write = None
        self.gamma = float(gamma)
        self._gamma_t = None

    def base_zhat(self, state, joint, k: int):
        return state + self.g0[k](joint)

    def experts_base_zhat(self, z, joint):
        return torch.stack([self.base_zhat(z, joint, k) for k in range(self.K)], dim=-2)

    def q_update(self, z, a, zn, p, log_p, z_hat):
        e_raw = z_hat - zn.unsqueeze(-2)
        e_safe = self.clip(e_raw, self.qcfg.residual_clip_norm, self.qcfg.epsilon)
        x_with = torch.cat((z, a, p, e_safe.flatten(start_dim=-2)), dim=-1)
        x_without = torch.cat((z, a, p, torch.zeros_like(e_safe).flatten(start_dim=-2)), dim=-1)
        q_delta = self.qcfg.logit_step_limit * torch.tanh(self.uq_raw(x_with) - self.uq_raw(x_without))
        posterior_logits = log_p.to(dtype=torch.float32) + q_delta
        return torch.log_softmax(posterior_logits, dim=-1).exp()

    def expert_zhat(self, *args, **kwargs):
        raise RuntimeError("pure-slow control: use base_zhat (there is no fast state)")

    def fast_update(self, *args, **kwargs):
        raise RuntimeError("pure-slow control: use q_update (there is no fast state)")


class SlowModel(C.MemoryModel):
    family = "slow"

    def make_fast_path(self) -> SlowPath:
        return SlowPath(self.core.fast_model, self.gamma)

    def init_state(self, B, ref):
        q = torch.full((B, K), 1.0 / K, dtype=torch.float32, device=ref.device)
        c = torch.zeros(B, K, 3, device=ref.device, dtype=ref.dtype)
        return (q, c, torch.ones_like(c))

    def observe_fast(self, state, z, a, zn):
        q, c, P = state
        fp, m = self.fp, self.core
        joint = torch.cat((z, a), dim=-1)
        log_p, p = fp.route(joint, q)
        z_hat = fp.experts_base_zhat(z, joint)
        q_new = fp.q_update(z, a, zn, p, log_p, z_hat)
        err = zn[:, None, :] - z_hat
        cs, ps = [], []
        for k in range(K):
            out = _kept(m.measurement[k], joint)
            J = torch.tanh(out[..., :3 * OBS].reshape(*out.shape[:-1], OBS, 3))
            bias = out[..., 3 * OBS:]
            y = err[:, k] - bias
            r = 0.05 + torch.nn.functional.softplus(_kept(m.filter[k], joint))
            Pk = P[:, k].clamp_min(1e-4)
            JPJ = (J.square() * Pk[:, None, :]).sum(dim=1)
            S = JPJ + r + 1e-3
            innov = (J * y[:, :, None]).sum(dim=1)
            gain = Pk / S.clamp_min(1e-3)
            c_next = c[:, k] + gain * innov / float(OBS)
            c_next = torch.tanh(c_next)
            p_next = ((1.0 - (Pk[:, None, :] * J.square() / S[:, None, :]).mean(dim=1)).clamp(0.02, 1.0) * Pk + 0.01)
            cs.append(c_next)
            ps.append(p_next)
        return (q_new, torch.stack(cs, dim=1), torch.stack(ps, dim=1))

    def predict_fast(self, state, z, af):
        q, c, P = state
        fp, m = self.fp, self.core
        log_p, _ = fp.route(torch.cat((z, af[:, 0]), dim=-1), q)
        paths = []
        for k in range(K):
            st, ck = z, c[:, k]
            zero_c = torch.zeros_like(ck)
            pp = []
            for h in range(af.shape[1]):
                a_h = af[:, h]
                joint = torch.cat((st, a_h), dim=-1)
                z_hat = fp.base_zhat(st, joint, k)
                delta = (_kept(m.readout[k], torch.cat((st, a_h, ck), dim=-1))
                         - _kept(m.readout[k], torch.cat((st, a_h, zero_c), dim=-1)))
                st = z_hat + delta
                pp.append(st)
            paths.append(torch.stack(pp, 1))
        return torch.stack(paths, -2), log_p

    def rollout_original(self, zc, ac, af):
        raise NotImplementedError("pure-slow control: this path needs U_F / G^F")

    def state_dims(self):
        return {"q": 3, "c": 9, "P": 9, "total_live": 21}


def check_layout(model) -> dict:
    groups = model.core.learning_modules()
    if sorted(groups) != ["Uq", "filter", "measurement", "readout"]:
        raise AssertionError(f"pure-slow groups {sorted(groups)}")
    fm = model.core.fast_model
    if len(fm.fast_readout_experts) or not isinstance(fm.accepted_model.fast_updater, RemovedFastBranch):
        raise AssertionError("fast branch still present")
    for kind in ("measurement", "filter", "readout"):
        for net in getattr(model.core, kind):
            if type(net[0]).__name__ != "PackedInputLinear" or net[0].input_kept_indices.tolist() != KEPT[kind]:
                raise AssertionError(f"{kind} first layer not packed to {KEPT[kind]}")
    _ = model.fp
    return {"groups": sorted(groups), "uf_removed": True, "gf_removed": True,
            "kept_columns": {k: len(v) for k, v in KEPT.items()}}


def build(cfg: dict, seed: int, device, stats) -> SlowModel:
    parent, fmap = C.build_parent_and_frame(cfg, seed, stats)
    cls, override = slow_class(cfg["widths"])
    mod, _mlp = override

    def factory(s):
        with C.override_global(mod, "_mlp", _mlp):
            return cls(parent.fast_model, int(s))

    core, report = C.construct_restricted(factory, int(seed), CONSTRUCTOR_STREAM, C.normalizers_for(parent, stats),
                                          original_target=int(cfg["original_target"]),
                                          target=int(cfg["active_target"]), slow_restriction=slow_restriction,
                                          common_nulls=slow_common_nulls)
    report["pure_slow_control"] = {"widths": list(cfg["widths"]), "fast_state": False, "fast_writer": False,
                                   "fast_readout": False, "routing_update_Uq": True,
                                   "constructor_stream": CONSTRUCTOR_STREAM}
    model = SlowModel(core, parent, fmap, cfg, report)
    model.to(device)
    report["layout"] = check_layout(model)
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
