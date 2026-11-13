from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import copy
import torch
from torch import Tensor, nn

try:
    from afterwm.lib.experts.models.stage1a import BaseExpertPrediction
except Exception:
    BaseExpertPrediction = None


def _mlp(d_in: int, hidden: tuple[int, ...], d_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = d_in
    for h in hidden:
        layers.extend((nn.Linear(d, h), nn.Tanh()))
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class AfterWMPrediction:
    d_base: Tensor
    d_fast: Tensor
    d_slow: Tensor
    d_hat: Tensor
    z_hat: Tensor


@dataclass(frozen=True)
class AfterWMTransition:
    belief: Any
    route: Any
    prediction: AfterWMPrediction
    fast_update: Any
    diagnostics: dict


@dataclass(frozen=True)
class AfterWMQuery:
    predicted_states: Tensor
    fast_states: Tensor
    slow_states: Tensor
    slow_corrections: Tensor
    route_log_p: Tensor


@dataclass(frozen=True)
class AfterWMBelief:
    fast: Any
    H: Tensor
    S: Tensor

    def detach(self):
        return AfterWMBelief(self.fast.detach(), self.H.detach(), self.S.detach())

    def clone(self):
        return AfterWMBelief(self.fast.clone(), self.H.clone(), self.S.clone())


class AfterWMNativeDifferentiable(nn.Module):


    OBS = 17
    ACT = 6
    EXPERTS = 3
    FAST = 32
    CONDITION = 3
    TRAINABLE_SLOW = 242_664
    TRAINABLE_TOTAL = 415_243

    def __init__(self, parent_fast: nn.Module, seed: int = 20260910):
        super().__init__()
        if any(p.requires_grad or p.grad is not None for p in parent_fast.parameters()):
            raise ValueError("parent fast model must be frozen and gradient-free")
        self.seed = int(seed)
        self.core = copy.deepcopy(parent_fast).requires_grad_(False).eval()
        self._slow_enabled, self._error_enabled = True, True
        self.measurement = nn.ModuleList(
            [_mlp(55, (149, 202), 68) for _ in range(self.EXPERTS)]
        )
        self.readout = nn.ModuleList(
            [_mlp(58, (128, 130), self.OBS) for _ in range(self.EXPERTS)]
        )
        self.filter = nn.ModuleList(
            [_mlp(55, (32,), self.CONDITION) for _ in range(self.EXPERTS)]
        )
        for net in self.readout:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.to(next(self.core.parameters()))
        self.set_training(False)
        slow = self.slow_parameter_count()
        if slow != self.TRAINABLE_SLOW:
            print('AFTER-WM parameter counts', [(k,sum(p.numel() for p in m.parameters())) for k,m in self.learning_modules().items()], 'slow=',slow, flush=True)
            raise RuntimeError(f"AFTER-WM slow count {slow} != {self.TRAINABLE_SLOW}")

    @property
    def fast_model(self):
        return self.core

    @property
    def config(self):
        return self.core.config

    @property
    def query_mode(self):
        return "full"

    def learning_modules(self):
        fast = {
            "Uq": self.core.accepted_model.q_only_model.belief_updater,
            "UF": self.core.fast_updater,
            "GF": self.core.fast_readout_experts,
        }
        return {**fast, "measurement": self.measurement, "filter": self.filter, "readout": self.readout}

    def declared_trainable_named(self):
        allowed = {id(p) for m in self.learning_modules().values() for p in m.parameters()}
        return [(n, p) for n, p in self.named_parameters() if id(p) in allowed]

    def slow_parameter_count(self):
        return sum(p.numel() for module in (self.measurement, self.filter, self.readout)
                   for p in module.parameters())

    def set_training(self, mode: bool):
        self.requires_grad_(False)
        self.train(bool(mode))
        self.core.eval()
        for module in self.learning_modules().values():
            module.requires_grad_(bool(mode))
            module.train(bool(mode))
        if not mode:
            self.zero_grad(set_to_none=True)
        return self

    def initialize_belief(self, state: Tensor):
        fast = self.core.initialize_belief(state)
        B = state.shape[0]
        device, dtype = state.device, state.dtype
        c = torch.zeros(B, self.EXPERTS, self.CONDITION, device=device, dtype=dtype)
        covariance = torch.ones_like(c)
        return AfterWMBelief(fast, c, covariance)

    def validate_belief(self, belief: AfterWMBelief, state: Tensor):
        if not isinstance(belief, AfterWMBelief):
            raise TypeError("belief must be AfterWMBelief")
        fb = belief.fast
        for name in ("q_previous", "F"):
            value = getattr(fb, name, None)
            if value is None or not torch.isfinite(value).all():
                raise FloatingPointError(f"invalid fast belief field {name}")
        expected = (state.shape[0], self.EXPERTS, self.CONDITION)
        if tuple(belief.H.shape) != expected or tuple(belief.S.shape) != expected:
            raise ValueError(f"invalid AFTER-WM state shape: {tuple(belief.H.shape)}")
        if not torch.isfinite(belief.H).all() or not torch.isfinite(belief.S).all():
            raise FloatingPointError("nonfinite AFTER-WM state")

    def intervention(self, *, slow: bool = True, error: bool = True):
        adapter = self
        class _Ctx:
            def __enter__(self):
                self.prev = adapter._slow_enabled, adapter._error_enabled
                adapter._slow_enabled, adapter._error_enabled = bool(slow), bool(error)
                return adapter
            def __exit__(self, *_):
                adapter._slow_enabled, adapter._error_enabled = self.prev
        return _Ctx()

    def _feature(self, z: Tensor, a: Tensor, F: Tensor) -> Tensor:
        return torch.cat((z, a, F), dim=-1)

    def _measurement(self, z: Tensor, a: Tensor, F: Tensor, k: int):
        out = self.measurement[k](self._feature(z, a, F))
        J = out[..., : self.OBS * self.CONDITION].reshape(*out.shape[:-1], self.OBS, self.CONDITION)
        b = out[..., self.OBS * self.CONDITION :]
        return torch.tanh(J), b

    def correction(self, z: Tensor, a: Tensor, F: Tensor, c: Tensor, k: int, *, enabled=True):
        if not enabled or not self._slow_enabled:
            return torch.zeros_like(z)
        x = torch.cat((z, a, F, c), dim=-1)
        x0 = torch.cat((z, a, F, torch.zeros_like(c)), dim=-1)
        return self.readout[k](x) - self.readout[k](x0)

    def _filter_update(self, z, a, next_z, fast_belief, c, covariance):
        F = fast_belief.F
        f = self.core.predict_experts(z, a, fast_belief)
        err = next_z[:, None, :] - f.z_hat
        cs, ps = [], []
        eps = torch.finfo(z.dtype).eps
        for k in range(self.EXPERTS):
            J, bias = self._measurement(z, a, F[:, k], k)
            y = err[:, k] - bias
            r = 0.05 + torch.nn.functional.softplus(self.filter[k](self._feature(z, a, F[:, k])))
            P = covariance[:, k].clamp_min(1e-4)
            JPJ = (J.square() * P[:, None, :]).sum(dim=1)
            S = JPJ + r + 1e-3
            innov = (J * y[:, :, None]).sum(dim=1)
            gain = P / S.clamp_min(1e-3)
            c_next = c[:, k] + gain * innov / float(self.OBS)
            c_next = torch.tanh(c_next)
            p_next = ((1.0 - (P[:, None, :] * J.square() / S[:, None, :]).mean(dim=1)).clamp(0.02, 1.0) * P + 0.01)
            cs.append(c_next)
            ps.append(p_next)
        return torch.stack(cs, dim=1), torch.stack(ps, dim=1), f

    def predict_experts(self, z, a, belief: AfterWMBelief, *, mode="full"):
        if mode not in ("off", "shadow", "full"):
            raise ValueError("invalid AFTER-WM mode")
        self.validate_belief(belief, z)
        f = self.core.predict_experts(z, a, belief.fast)
        if mode == "full" and self._slow_enabled:
            d_slow = torch.stack([
                self.correction(z, a, belief.fast.F[:, k], belief.H[:, k], k, enabled=True)
                for k in range(self.EXPERTS)
            ], dim=-2)
        else:
            d_slow = torch.zeros_like(f.d_hat)
        return AfterWMPrediction(f.d_base, f.d_fast, d_slow, f.d_hat + d_slow, f.z_hat + d_slow)

    def observe_transition(self, z, a, next_z, belief: AfterWMBelief, *, mode="full"):
        if mode not in ("off", "shadow", "full"):
            raise ValueError("invalid AFTER-WM mode")
        self.validate_belief(belief, z)
        route = self.core.prior_routing(z, a, belief.fast)
        f = self.core.predict_experts(z, a, belief.fast)
        if self._error_enabled:
            global BaseExpertPrediction
            if BaseExpertPrediction is None:
                from afterwm.lib.experts.models.stage1a import BaseExpertPrediction as _BEP
                BaseExpertPrediction = _BEP
            native = BaseExpertPrediction(f.d_base, f.d_hat, f.z_hat)
            fast_update = self.core.post_action_update(z, a, next_z, route, native, belief.fast)
            fast = fast_update.belief
        else:
            fast_update = None
            fast = type(belief.fast)(route.p, self.core.fast_retention * belief.fast.F)
        if mode == "off" or not self._slow_enabled:
            c_next, p_next = belief.H, belief.S
        else:
            c_next, p_next, _ = self._filter_update(z, a, next_z, belief.fast, belief.H, belief.S)
        after = AfterWMBelief(fast, c_next, p_next)
        self.validate_belief(after, z)
        pred = self.predict_experts(z, a, belief, mode=mode)
        diag = {"slow_correction": pred.d_slow.detach(), "measurement_state": c_next.detach()}
        return AfterWMTransition(after, route, pred, fast_update, diag)

    @torch.no_grad()
    def natural_no_current_error(self, z, a, belief):
        with self.intervention(slow=self._slow_enabled, error=False):
            return self.observe_transition(z, a, z, belief).belief.detach()

    @torch.no_grad()
    def burn_in(self, observations, actions, *, mode="full"):
        b = self.initialize_belief(observations[:, 0])
        for t in range(actions.shape[1]):
            b = self.observe_transition(observations[:, t], actions[:, t], observations[:, t + 1], b, mode=mode).belief
        return b.detach()

    def predict_future(self, z, actions, belief: AfterWMBelief, *, mode="full"):
        self.validate_belief(belief, z)
        route = self.core.prior_routing(z, actions[:, 0], belief.fast)
        paths, fs, ss, cs = [], [], [], []
        for k in range(self.EXPERTS):
            state, F, c = z, belief.fast.F[:, k], belief.H[:, k]
            pp, ff, cc = [], [], []
            for h in range(actions.shape[1]):
                base = self.core.predict_expert(state, actions[:, h], F, k)
                delta = self.correction(state, actions[:, h], F, c, k, enabled=(mode == "full"))
                state = base.z_hat + delta
                pp.append(state); ff.append(F); cc.append(delta)
                F = self.core.fast_retention * F
            paths.append(torch.stack(pp, 1)); fs.append(torch.stack(ff, 1)); cs.append(torch.stack(cc, 1))
            ss.append(c[:, None, :].expand(-1, actions.shape[1], -1))
        return AfterWMQuery(torch.stack(paths, -2), torch.stack(fs, -2), torch.stack(ss, -2), torch.stack(cs, -2), route.log_p)


if __name__ == "__main__":
    m = AfterWMNativeDifferentiable.__dict__
    print("expected slow parameters:", AfterWMNativeDifferentiable.TRAINABLE_SLOW)
