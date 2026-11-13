from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from pipeline import registry as _P

NATIVE_OBS, NATIVE_ACT = 17, 6
OBS, ACT, K, FAST_DIM, SLOW_DIM = int(_P.OBS_DIM), int(_P.ACT_DIM), 3, 32, 32
NATIVE_DIMS = (OBS, ACT) == (NATIVE_OBS, NATIVE_ACT)


def base_param_count(obs: int = OBS, act: int = ACT, k: int = 3) -> int:
    d = obs + act
    router = (d * 128 + 128) + 256 + (128 * 128 + 128) + 256 + (128 * k + k)
    g0 = (d * 256 + 256) + 2 * (256 * 256 + 256) + (256 * obs + obs)
    return router + k * g0


def uq_pre_count(obs: int = OBS, act: int = ACT) -> int:
    d_in = obs + act + K + 3 * K * FAST_DIM + K * obs
    return (d_in * 128 + 128) + 256 + (128 * 128 + 128) + 256 + (128 * K + K)


def uf_pre_count(obs: int = OBS, act: int = ACT) -> int:
    d_in = obs + act + 1 + 1 + FAST_DIM + obs
    return (d_in * 128 + 128) + 256 + (128 * 128 + 128) + 256 + (128 * FAST_DIM + FAST_DIM)


def gf_param_count(h: int = 128, obs: int = OBS, act: int = ACT) -> int:
    return 3 * (h * h + (2 * obs + act + FAST_DIM + 2) * h)


BASE_PARAMS = base_param_count()
HORIZONS = (1, 5, 10, 20, 40)
SIGMA = 0.5


def configure_runtime(threads: int = 1, deterministic: bool = True) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(int(threads))
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False, default=_json_default) + "\n")
    os.replace(tmp, path)


def atomic_torch_save(path, value) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (tuple, set)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def code_manifest(extra=()) -> dict:
    files = [HERE / n for n in ("core.py", "base.py", "model.py", "stage1.py")]
    files += [Path(p) for p in extra]
    out = {str(p.relative_to(HERE.parent)): sha256_file(p) for p in files if p.is_file()}
    return out


FAST_PRE = uq_pre_count() + uf_pre_count() + gf_param_count()
COMMON_NULLS = 24_611
FAST_ACTIVE = FAST_PRE - COMMON_NULLS
if NATIVE_DIMS and (BASE_PARAMS, FAST_PRE, FAST_ACTIVE) != (446_774, 172_579, 147_968):
    raise AssertionError("17/6 parameter counts differ from the reference values")


def _fc():
    from .lib.memory import budget as fairness_checks
    return fairness_checks


def default_slow_restriction(model, names):
    import torch
    fc = _fc()
    groups = model.learning_modules()
    rows = []
    if all(hasattr(model, n) for n in ("measurement", "filter", "readout")):
        arch = "neural_LF_AFTERWM"
        for k, net in enumerate(model.readout):
            last = net[-1]
            if type(last) is not torch.nn.Linear or last.bias is None or last.bias.numel() != OBS:
                raise AssertionError(f"expected {OBS} centered-readout final biases per AFTER-WM expert")
            last.register_parameter("bias", None)
            rows.append({"expert": k, "layer": names[id(last)], "method": "delete final centered readout bias",
                         "removed_trainable_scalars": OBS, "removal_forward_invariant": True})
    elif "GS" in groups or "GS_basis" in groups:
        group = "GS" if "GS" in groups else "GS_basis"
        arch = "slow_wide_readout" if group == "GS" else "slow_basis_readout"
        for k, readout in enumerate(groups[group]):
            first = readout.network[0]
            before = sum(p.numel() for p in first.parameters())
            packed = fc.pack_linear_bias(first, fixed_last=OBS)
            readout.network[0] = packed
            if before - sum(p.numel() for p in packed.parameters()) != OBS:
                raise AssertionError(f"packed bias did not remove exactly {OBS} genuine scalars")
            rows.append({"expert": k, "layer": names[id(first)], "method": "packed first-layer slow readout bias",
                         "removed_trainable_scalars": OBS, "removal_forward_invariant": False})
    else:
        raise TypeError("unrecognized strict dual architecture; pass slow_restriction=")
    return arch, rows


def remove_common_nulls_dims(model):
    import torch
    fc = _fc()
    groups = model.learning_modules()
    names = {id(module): name for name, module in model.named_modules()}
    uq, uf = groups["Uq"], groups["UF"]
    config = uq.config
    network = fc._writer_network(uq)
    first = network[0]
    z, a, k, f, s = (config.observation_dim, config.action_dim,
                     config.num_experts, config.fast_dim, config.slow_dim)
    boundary = z + a + k + k * f
    width = k * s + k * f
    expected_input = z + a + k + k * f + k * s + k * f + k * z
    if type(first) is not torch.nn.Linear or first.in_features != expected_input:
        raise AssertionError("Uq input geometry/order differs from reviewed z,a,p,F,S,H,e")
    if (z, a, k, f, s, first.out_features, width) != (OBS, ACT, 3, 32, 32, 128, 192):
        raise AssertionError("Uq geometry differs from the active-dataset geometry")
    excluded = list(range(boundary, boundary + width))
    packed = fc.pack_linear_input(first, fixed_zero_indices=excluded)
    network[0] = packed
    rows = [{"group": "Uq", "layer": names[id(first)], "method": "remove weights on permanent zero S/H inputs",
             "original_input_width": first.in_features, "active_input_width": packed.weight.shape[1],
             "feature_order": ["z", "a", "p", "F", "S", "H", "e"],
             "zero_input_coordinates": excluded,
             "removed_trainable_scalars": first.out_features * width,
             "removal_forward_invariant": True, "every_forward_zero_premise_asserted": True}]
    for group, writer in (("Uq", uq), ("UF", uf)):
        last = fc._writer_network(writer)[-1]
        expected = 3 if group == "Uq" else 32
        if type(last) is not torch.nn.Linear or last.bias is None or last.bias.numel() != expected:
            raise AssertionError("unexpected centered common-writer final bias: " + group)
        last.register_parameter("bias", None)
        rows.append({"group": group, "layer": names[id(last)], "method": "delete centered writer final bias",
                     "removed_trainable_scalars": expected, "removal_forward_invariant": True,
                     "reason": "raw_with_error - raw_without_error cancels final bias"})
    if sum(row["removed_trainable_scalars"] for row in rows) != COMMON_NULLS:
        raise AssertionError("common exact null removal count changed")
    return rows


def apply_active_budget_param(model, model_name, original_target, target, slow_restriction=None,
                              common_nulls=None):
    fc = _fc()
    if getattr(model, "_strict_active_budget_receipt", None) is not None:
        raise AssertionError("active budget already applied")
    original = sum(p.numel() for _, p in model.declared_trainable_named())
    if original != original_target:
        raise AssertionError(f"unexpected pre-restriction count {original} != {original_target}")
    names = {id(m): n for n, m in model.named_modules()}
    arch, rows = (slow_restriction or default_slow_restriction)(model, names)
    rows.extend((common_nulls or (fc.remove_common_nulls if NATIVE_DIMS else remove_common_nulls_dims))(model))
    actual = sum(p.numel() for _, p in model.declared_trainable_named())
    removed = sum(r["removed_trainable_scalars"] for r in rows)
    if actual != target or removed != original_target - target:
        raise AssertionError(f"restricted active budget {actual} != {target} (removed {removed})")
    receipt = {"schema": "afterwm_active_budget_v1(parameterised)", "model": model_name,
               "architecture": arch, "original_declared_scalars": original, "removed_scalars": removed,
               "target_scalars": target, "actual_declared_scalars": actual, "restrictions": rows}
    model._strict_active_budget_receipt = receipt
    return receipt


def construct_restricted(factory, seed: int, model_name: str, normalizers: dict, *, original_target: int,
                         target: int, slow_restriction=None, common_nulls=None, joint_gf_init_std: float = .001):

    fc = _fc()
    with fc.seeded_rng(fc.seed_for(seed, "architecture-constructor", model_name)):
        model = factory(seed)
    if int(getattr(model, "seed", seed)) != seed:
        raise AssertionError("factory did not pass init_seed to the constructor")
    model.set_training(True)
    before = fc.capture_invariants(model, normalizers)
    slow_fp = lambda: {g: fc._mapping_fingerprint({n: fc._tensor_description(v) for n, v in m.state_dict().items()})
                       for g, m in model.learning_modules().items() if g not in fc.SHARED_GROUPS}
    slow_before = slow_fp()
    reset = fc.reset_common_fast(model, seed, joint_gf_init_std=joint_gf_init_std)
    fc.assert_invariants_unchanged(before, fc.capture_invariants(model, normalizers))
    if slow_fp() != slow_before:
        raise AssertionError("common reset modified constructor-initialized slow modules")
    budget = apply_active_budget_param(model, model_name, original_target, target, slow_restriction, common_nulls)
    after = fc.capture_invariants(model, normalizers)
    for key in ("base_sha256", "legacy_base_sha256", "normalizer_sha256"):
        if before[key] != after[key]:
            raise AssertionError("budget restriction altered frozen base/normalizer: " + key)
    manifest = fc.parameter_manifest(model, target=target)
    nulls = fc.structural_null_parameters(model)
    if nulls:
        raise AssertionError("known structurally null trainable parameters: " + repr(nulls))
    group_sha = {g: fc._mapping_fingerprint({n: fc._tensor_description(p) for n, p in m.named_parameters()})
                 for g, m in model.learning_modules().items()}
    report = {"model": model_name, "seed": seed, "base_sha256": after["base_sha256"],
              "legacy_base_sha256": after["legacy_base_sha256"], "normalizer_sha256": after["normalizer_sha256"],
              "groups": manifest["groups"], "trainable": manifest["total_scalars"],
              "group_sha256": group_sha, "reset": reset, "active_budget": budget}
    return model, report


class FastPath:


    def __init__(self, fm, gamma: float):
        import torch
        self.fm = fm
        self.base = fm.base_model
        bc = self.base.config
        self.K, self.floor, self.alpha_q, self.eps_base = bc.num_experts, bc.route_probability_floor, bc.alpha_q, bc.epsilon
        self.router = self.base.router
        self.g0 = list(self.base.g0_experts)
        self.gf = list(fm.fast_readout_experts)
        qm = fm.q_only_model
        self.qcfg = qm.config
        self.ufcfg = fm.accepted_model.config
        seq = qm.belief_updater.network.network
        first = seq[0]
        if type(first).__name__ == "PackedInputLinear":
            kept = first.input_kept_indices.tolist()
            c = self.qcfg
            boundary = c.observation_dim + c.action_dim + c.num_experts + c.num_experts * c.fast_dim
            width = c.num_experts * (c.slow_dim + c.fast_dim)
            if kept != list(range(boundary)) + list(range(boundary + width, first.in_features)):
                raise AssertionError("unexpected Uq kept-column layout")
            self.uq_first = first
        elif isinstance(first, torch.nn.Linear):
            raise AssertionError("Uq must be restricted (PackedInputLinear) before building the fast path")
        else:
            raise AssertionError(f"unexpected Uq first layer {type(first)}")
        self.uq_rest = list(seq)[1:]
        self.uf_module = fm.fast_updater
        self.retention = self.ufcfg.fast_retention
        self.write = (1.0 - self.ufcfg.fast_retention) * self.ufcfg.fast_innovation_limit
        self.gamma = float(gamma)
        self._gamma_t = None

    def gamma_tensor(self, ref):
        import torch
        if self._gamma_t is None or self._gamma_t.device != ref.device:
            self._gamma_t = torch.tensor(self.gamma, device=ref.device, dtype=ref.dtype)
        return self._gamma_t

    def route(self, joint, q):
        import torch
        router_logits = self.router(joint)
        if self.gamma != 1.0:
            router_logits = router_logits * self.gamma_tensor(router_logits)
        q_memory = (1.0 - self.floor) * q.to(dtype=torch.float32) + (self.floor / self.K)
        memory_logits = self.alpha_q * torch.log(q_memory.clamp_min(self.eps_base))
        raw_logits = router_logits.to(dtype=torch.float32) + memory_logits
        raw_log_p = torch.log_softmax(raw_logits, dim=-1)
        raw_probability = raw_log_p.exp()
        probability = (1.0 - self.floor) * raw_probability + (self.floor / self.K)
        return torch.log(probability), probability

    def expert_zhat(self, state, action, fast_state, k: int, joint=None):
        import torch
        if joint is None:
            joint = torch.cat((state, action), dim=-1)
        d_base = self.g0[k](joint)
        expert = self.gf[k]
        with_fast = torch.cat((state, action, fast_state), dim=-1)
        without_fast = torch.cat((state, action, torch.zeros_like(fast_state)), dim=-1)
        d_fast = expert(with_fast) - expert(without_fast)
        d_hat = d_base + d_fast
        return state + d_hat

    def experts_zhat(self, z, a, F, joint):
        import torch
        return torch.stack([self.expert_zhat(z, a, F[..., k, :], k, joint) for k in range(self.K)], dim=-2)

    @staticmethod
    def clip(values, max_norm, epsilon):
        import torch
        norms = torch.linalg.vector_norm(values, ord=2, dim=-1, keepdim=True)
        scale = torch.clamp(float(max_norm) / (norms + float(epsilon)), max=1.0)
        return values * scale

    def uq_raw(self, x_kept):
        import torch
        h = torch.nn.functional.linear(x_kept, self.uq_first.weight, self.uq_first.bias)
        for layer in self.uq_rest:
            h = layer(h)
        return h

    def uf_input(self, z, a, p, q, F, e):
        import torch
        K = self.K
        return torch.cat((z.unsqueeze(-2).expand(*z.shape[:-1], K, -1),
                          a.unsqueeze(-2).expand(*a.shape[:-1], K, -1),
                          p.unsqueeze(-1), q.unsqueeze(-1), F, e), dim=-1)

    def uf_innovation(self, x_with, x_without):
        import torch
        net = self.uf_module.network
        return torch.tanh(net(x_with) - net(x_without))

    def fast_update(self, z, a, zn, p, log_p, F, z_hat):
        import torch
        e_raw = z_hat - zn.unsqueeze(-2)
        e_safe = self.clip(e_raw, self.qcfg.residual_clip_norm, self.qcfg.epsilon)
        x_with = torch.cat((z, a, p, F.flatten(start_dim=-2), e_safe.flatten(start_dim=-2)), dim=-1)
        x_without = torch.cat((z, a, p, F.flatten(start_dim=-2),
                               torch.zeros_like(e_safe).flatten(start_dim=-2)), dim=-1)
        q_delta = self.qcfg.logit_step_limit * torch.tanh(self.uq_raw(x_with) - self.uq_raw(x_without))
        posterior_logits = log_p.to(dtype=torch.float32) + q_delta
        log_q = torch.log_softmax(posterior_logits, dim=-1)
        q = log_q.exp()
        evidence_safe = self.clip(e_raw, self.ufcfg.residual_clip_norm, self.ufcfg.epsilon)
        uf_with = self.uf_input(z, a, p, q, F, evidence_safe)
        uf_without = self.uf_input(z, a, p, q, F, torch.zeros_like(evidence_safe))
        innovation = self.uf_innovation(uf_with, uf_without)
        F_next = (self.retention * F) + (self.write * innovation)
        return q, F_next


def residual_step_state(state, basis, probabilities, residual, *, eta=.05, ridge=.01, radius=4., step_limit=.1):
    import torch
    b, k, d = state.shape
    o = basis.shape[2]
    s = state.double().reshape(b, k * d)
    A = (basis.double() * probabilities.double()[:, :, None, None]).permute(0, 2, 1, 3).reshape(b, o, k * d)
    r = residual.double()
    gram = A @ A.transpose(-1, -2)
    upper = gram.abs().sum(-1).amax(-1, keepdim=True) + ridge
    step = (eta / upper) * (A.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)
    inward = 1 - 16 * torch.finfo(state.dtype).eps
    norm = step.norm(dim=-1, keepdim=True)
    fraction = torch.clamp((step_limit * inward) / norm.clamp_min(1e-30), max=1.)
    proposed = s + fraction * step
    size = proposed.norm(dim=-1, keepdim=True)
    projected = proposed * torch.clamp((radius * inward) / size.clamp_min(1e-30), max=1.)
    updated = projected.reshape_as(state).to(state.dtype)
    return torch.where((norm == 0).reshape(b, 1, 1), state, updated)


NATIVE_CONST = .5 * np.log(2 * np.pi * .5 ** 2)
METRIC_NAMES = tuple(f"H{h}_{m}" for h in HORIZONS for m in ("native_NLL", "J", "MSE_raw", "nMSE_train"))


def strict_loss(predicted_states, route_log_p, target, own_std, common_std, raw_scale: float):

    import torch
    mu = predicted_states.double()
    y = target.double()
    logp = torch.log_softmax(route_log_p.double(), -1)
    if mu.ndim != 4 or y.shape != (mu.shape[0], mu.shape[1], mu.shape[-1]) or mu.shape[1] < max(HORIZONS):
        raise ValueError("complete H40 query required for the strict loss")
    own = own_std.to(device=mu.device, dtype=torch.float64)
    common = common_std.to(device=mu.device, dtype=torch.float64)
    mean = (logp.exp()[:, None, :, None] * mu).sum(2)
    raw = (mean - y) * own
    square_component = (mu - y[:, :, None]).square().sum(-1)
    losses, recs = [], []
    for h in HORIZONS:
        nll = -torch.logsumexp(logp - square_component[:, :h].sum(1) / (2 * .5 ** 2), -1) / (h * mu.shape[-1]) + NATIVE_CONST
        rmse = raw[:, :h].square().mean((1, 2))
        nmse = (raw[:, :h] / common).square().mean((1, 2))
        J = .5 * (nmse + rmse / raw_scale)
        losses.append(nll.mean() + .5 * J.mean())
        recs.extend((nll.detach().mean(), J.detach().mean(), rmse.detach().mean(), nmse.detach().mean()))
    return torch.stack(losses).mean(), torch.stack(recs)


import torch as _torch
from torch import nn as _nn


class MemoryModel(_nn.Module):

    family = "afterwm"

    def __init__(self, core, parent, fmap, cfg: dict, init_report: dict):
        super().__init__()
        self.core = core
        self.fmap = fmap
        self.cfg = dict(cfg)
        self.variant = parent.variant
        self.gamma = float(parent.gamma)
        self.c_e = float(parent.c_e)
        self.__dict__["parent_identity"] = parent.identity
        self.__dict__["init_report"] = init_report
        self.__dict__["_fp"] = None
        self.__dict__["_graph_cache"] = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_graph_cache"] = {}
        d["_fp"] = None
        return d

    @property
    def fp(self) -> FastPath:
        if self._fp is None:
            self.__dict__["_fp"] = self.make_fast_path()
        return self._fp

    def make_fast_path(self) -> FastPath:
        return FastPath(self.core.fast_model, self.gamma)

    def init_state(self, B, ref):
        raise NotImplementedError

    def observe_fast(self, state, z, a, zn):
        raise NotImplementedError

    def predict_fast(self, state, z, af):
        raise NotImplementedError

    def native_inputs(self, ctx_obs, ctx_act, fut_act, fut_obs=None):
        f = self.fmap
        y = None if fut_obs is None else f.obs_to_native(fut_obs)
        return f.obs_to_native(ctx_obs), f.act_to_native(ctx_act), f.act_to_native(fut_act), y

    def rollout_fast(self, zc, ac, af):
        state = self.init_state(zc.shape[0], zc)
        for j in range(ac.shape[1]):
            state = self.observe_fast(state, zc[:, j], ac[:, j], zc[:, j + 1])
        return self.predict_fast(state, zc[:, -1], af)

    def rollout_original(self, zc, ac, af):
        from .base import gamma_hook
        core = self.core
        with gamma_hook(core, self.gamma), core.intervention(error=True, slow=True):
            b = core.initialize_belief(zc[:, 0])
            for j in range(ac.shape[1]):
                b = core.observe_transition(zc[:, j], ac[:, j], zc[:, j + 1], b, mode="full").belief
            q = core.predict_future(zc[:, -1], af, b, mode="full")
        return q.predicted_states, q.route_log_p

    def loss_and_metrics(self, zc, ac, af, y, original=False):
        pred, logp = (self.rollout_original if original else self.rollout_fast)(zc, ac, af)
        return strict_loss(pred, logp, y, self.fmap.own_std, self.fmap.common_std, self.fmap.raw_scale)

    def forward(self, zc, ac, af, y):
        return self.loss_and_metrics(zc, ac, af, y)

    def train_loss(self, batch: dict):

        zc, ac, af, y = self.native_inputs(batch["ctx_obs"], batch["ctx_act"], batch["fut_act"], batch["fut_obs"])
        if graph_enabled(self.cfg) and zc.is_cuda:
            key = (tuple(zc.shape), tuple(ac.shape), tuple(af.shape), str(zc.device), zc.dtype)
            runner = self._graph_cache.get("runner")
            if runner is None or runner.key != key:
                self._graph_cache.clear()
                t0 = time.time()
                runner = GraphRunner(self, (zc, ac, af, y), key)
                self._graph_cache.update(runner=runner, status="ok", capture_seconds=time.time() - t0)
            loss, metrics = _GraphedLoss.apply(runner, zc, ac, af, y, *runner.params)
            return loss, metrics, 1.0
        loss, metrics = self.loss_and_metrics(zc, ac, af, y)
        return loss, metrics, 0.0

    def parameter_report(self) -> dict:
        groups = {"base_router": sum(p.numel() for p in self.core.fast_model.base_model.router.parameters()),
                  "base_G0": sum(p.numel() for p in self.core.fast_model.base_model.g0_experts.parameters())}
        for g, m in self.core.learning_modules().items():
            groups[g] = sum(p.numel() for p in m.parameters())
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        declared = sum(p.numel() for _, p in self.core.declared_trainable_named())
        if sum(groups.values()) != total:
            raise AssertionError(f"parameter groups {sum(groups.values())} != total {total}")
        buffers = sum(b.numel() for b in self.buffers())
        return {"total": total, "trainable": declared, "trainable_requires_grad_now": trainable,
                "frozen": total - declared, "stored_nongrad": 0, "buffers": buffers, "groups": groups,
                "budget_class": self.cfg.get("budget_class"), "budget": self.cfg.get("budget_total"),
                "state_dims_per_query": self.state_dims()}

    def state_dims(self) -> dict:
        return {}


def graph_enabled(cfg) -> bool:
    v = cfg.get("cuda_graph", True)
    if isinstance(v, str):
        return v.strip().lower() not in ("0", "false", "no", "off", "")
    return bool(v)


class GraphRunner:


    def __init__(self, model, sample, key):
        self.key = key
        named = list(model.core.declared_trainable_named())
        self.params = [p for _, p in named]
        ref = self.params[0]
        full = {id(p): n for n, p in model.named_parameters(remove_duplicate=False)}
        if len(full) != len(list(model.named_parameters(remove_duplicate=False))):
            raise AssertionError("tied parameters are not supported by the graph runner")
        self.alias = {full[id(p)]: p.detach().requires_grad_(True) for p in self.params}
        alias_list = list(self.alias.values())
        model.fp.gamma_tensor(ref)
        self.static = [t.detach().clone() for t in sample]
        from torch.func import functional_call

        def fwd():
            return functional_call(model, self.alias, tuple(self.static))

        stream = _torch.cuda.Stream(device=ref.device)
        stream.wait_stream(_torch.cuda.current_stream(ref.device))
        with _torch.cuda.stream(stream):
            for _ in range(3):
                loss, met = fwd()
                grads = _torch.autograd.grad(loss, alias_list, allow_unused=True)
                del loss, met, grads
        _torch.cuda.current_stream(ref.device).wait_stream(stream)
        _torch.cuda.synchronize(ref.device)
        self.graph = _torch.cuda.CUDAGraph()
        with _torch.cuda.graph(self.graph, stream=stream):
            self.loss, self.metrics = fwd()
            self.grads = _torch.autograd.grad(self.loss, alias_list, allow_unused=True)
        _torch.cuda.synchronize(ref.device)
        self.stream = stream

    def run(self, inputs):
        for s, x in zip(self.static, inputs):
            s.copy_(x)
        self.graph.replay()


class _GraphedLoss(_torch.autograd.Function):
    @staticmethod
    def forward(ctx, runner, zc, ac, af, y, *params):
        runner.run((zc, ac, af, y))
        ctx.runner = runner
        metrics = runner.metrics.detach().clone()
        ctx.mark_non_differentiable(metrics)
        return runner.loss.detach().clone(), metrics

    @staticmethod
    def backward(ctx, g_loss, g_metrics):
        grads = tuple(None if g is None else g * g_loss.to(g.dtype) for g in ctx.runner.grads)
        return (None, None, None, None, None) + grads


def metrics_dict(metrics, graphed: float) -> dict:
    vals = metrics.detach().double().cpu().tolist()
    out = dict(zip(METRIC_NAMES, vals))
    out["cuda_graph"] = graphed
    return out


class MemoryForecaster:
    family = "afterwm"
    needs_grad_at_eval = False
    stochastic = False

    def __init__(self, model: MemoryModel, name: str, budget_class: str, audit_descent: bool = False):
        self.model = model
        self.name = name
        self.budget_class = budget_class
        self.audit_descent = audit_descent

    def forecast(self, q):
        from pipeline.eval_api import Forecast
        m = self.model
        with _torch.no_grad():
            zc, ac, af, _ = m.native_inputs(q.ctx_obs, q.ctx_act, q.fut_act)
            if self.audit_descent:
                pred, logp = m.rollout_original(zc, ac, af)
            else:
                pred, logp = m.rollout_fast(zc, ac, af)
            logw = _torch.log_softmax(logp.double(), -1)
            mean_native = (logw.exp()[:, None, :, None] * pred.double()).sum(2)
            comp = m.fmap.obs_to_train(pred)
            mean = m.fmap.obs_to_train(mean_native).float()
        return Forecast(mean=mean, comp_mean=comp.float(), comp_logw=logw.float())

    def parameter_report(self) -> dict:
        return self.model.parameter_report()


FAMILIES: dict = {}


def register_family(name: str, module) -> None:
    FAMILIES[name] = module


def adam_optimizer(model, cfg):
    params = [p for _, p in model.core.declared_trainable_named()]
    return _torch.optim.Adam(params, lr=cfg.get("lr", 1e-4), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def generic_train_step(model: MemoryModel, batch: dict, step: int, cfg: dict):
    loss, metrics, graphed = model.train_loss(batch)
    return loss, metrics_dict(metrics, graphed)


@contextlib.contextmanager
def override_global(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def build_parent_and_frame(cfg, seed, stats):
    from .base import build_fast_parent, FrameMap
    if stats is None:
        raise ValueError("build() needs the TRAIN stats (pipeline.data.TrainStats)")
    gf = cfg.get("gf_hidden")
    gf = None if gf is None or tuple(gf) == (128, 128) else tuple(gf)
    parent = build_fast_parent(cfg["base"], seed, cfg.get("construct_device", "cpu"), stats=stats, gf_hidden=gf,
                               router_gamma=cfg.get("router_gamma"))
    return parent, FrameMap(parent, stats)


def normalizers_for(parent, stats) -> dict:
    return {"native": {k: np.asarray(v) for k, v in parent.native.items()},
            "train_stats_sha256": stats.sha256(), "variant": parent.variant}
