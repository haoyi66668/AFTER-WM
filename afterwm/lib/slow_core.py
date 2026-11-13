from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from afterwm.lib.experts.models.stage1a import MLP, BaseExpertPrediction, PriorRoutingOutput
from afterwm.lib.experts.models.stage1b0 import clip_vectors_by_norm
from afterwm.lib.experts.models.stage1b1 import (
    Stage1B1FastBelief,
    Stage1B1FastPostActionOutput,
    validate_stage1b1_fast_belief,
)
from afterwm.lib.experts.models.stage1b2_fast_readout import Stage1B2FastReadoutWorldModel
from torch import Tensor, nn

Mode = Literal["off", "shadow", "full"]


@dataclass(frozen=True)
class SlowConfig:
    slow_dim: int = 32
    rho: float = 0.98
    slow_step_limit: float = 0.05
    slow_max_norm: float = 1.0
    fixed_slow_input_gain: float = 1.0
    fixed_ignore: float = 0.0
    readout_hidden: tuple[int, ...] = (128, 128)
    consolidation_hidden: tuple[int, ...] = (128,)
    readout_init_std: float = 0.001
    consolidation_init_std: float = 0.01
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("slow_dim",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("readout_hidden", "consolidation_hidden"):
            values = tuple(getattr(self, name))
            if not values or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, values)
        for name in (
            "rho",
            "slow_step_limit",
            "slow_max_norm",
            "fixed_slow_input_gain",
            "readout_init_std",
            "consolidation_init_std",
            "epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.rho < 1 or not 0 < self.fixed_slow_input_gain <= 1:
            raise ValueError("rho must be in (0,1); fixed_slow_input_gain in (0,1]")
        if isinstance(self.fixed_ignore, bool) or not 0 <= self.fixed_ignore <= 1:
            raise ValueError("fixed_ignore must be in [0,1]")


@dataclass(frozen=True)
class SlowBelief:
    fast: Stage1B1FastBelief
    H: Tensor
    S: Tensor

    def detach(self) -> SlowBelief:
        return SlowBelief(self.fast.detach(), self.H.detach(), self.S.detach())

    def clone(self) -> SlowBelief:
        return SlowBelief(
            Stage1B1FastBelief(self.fast.q_previous.clone(), self.fast.F.clone()),
            self.H.clone(),
            self.S.clone(),
        )


@dataclass(frozen=True)
class SlowPrediction:
    d_base: Tensor
    d_fast: Tensor
    d_slow: Tensor
    d_hat: Tensor
    z_hat: Tensor

    def as_base_prediction(self) -> BaseExpertPrediction:
        return BaseExpertPrediction(self.d_base, self.d_hat, self.z_hat)


@dataclass(frozen=True)
class SlowWrite:
    H: Tensor
    S: Tensor
    F_prior: Tensor
    actual_delta_F: Tensor
    H_prior: Tensor
    actual_delta_H: Tensor
    actual_delta_S: Tensor
    consolidation_code: Tensor


@dataclass(frozen=True)
class TransitionResult:
    belief: SlowBelief
    route: PriorRoutingOutput
    prediction: SlowPrediction
    fast_update: Stage1B1FastPostActionOutput
    slow_update: SlowWrite | None


@dataclass(frozen=True)
class QueryPrediction:
    predicted_states: Tensor
    fast_states: Tensor
    slow_states: Tensor
    slow_corrections: Tensor
    route_log_p: Tensor


def _mode(mode: str) -> None:
    if mode not in ("off", "shadow", "full"):
        raise ValueError("mode must be off, shadow or full")


def _tensor(
    value: Tensor, shape: tuple[int, ...], reference: Tensor, name: str
) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != torch.float32 or value.device != reference.device:
        raise TypeError(f"{name} must be float32 on {reference.device}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")


def _small_last_layer(module: MLP, std: float) -> None:
    previous = module.network[-1]
    if not isinstance(previous, nn.Linear):
        raise TypeError("MLP must end in Linear")
    output = nn.Linear(previous.in_features, previous.out_features, bias=False)
    nn.init.normal_(output.weight, mean=0.0, std=std)
    module.network[-1] = output


class SlowCore(nn.Module):


    def __init__(self, fast_model: Stage1B2FastReadoutWorldModel, config: SlowConfig):
        super().__init__()
        if not isinstance(fast_model, Stage1B2FastReadoutWorldModel):
            raise TypeError("fast_model must be the existing Stage1B2 model")
        if any(p.requires_grad or p.grad is not None for p in fast_model.parameters()):
            raise ValueError("B parameters must already be frozen and gradient-free")
        self.fast_model = fast_model
        self.config = config
        c = fast_model.config
        self.slow_readouts = nn.ModuleList(
            MLP(
                c.observation_dim + c.action_dim + config.slow_dim,
                config.readout_hidden,
                c.observation_dim,
                layer_norm=False,
            )
            for _ in range(c.num_experts)
        )
        self.consolidators = nn.ModuleList(
            MLP(
                2 * c.fast_dim,
                config.consolidation_hidden,
                c.fast_dim,
                layer_norm=False,
            )
            for _ in range(c.num_experts)
        )
        self.slow_updaters = nn.ModuleList(
            nn.GRUCell(c.fast_dim + c.observation_dim + c.action_dim, config.slow_dim)
            for _ in range(c.num_experts)
        )
        for readout in self.slow_readouts:
            _small_last_layer(readout, config.readout_init_std)
        for mapping in self.consolidators:
            _small_last_layer(mapping, config.consolidation_init_std)
        ref = next(fast_model.parameters())
        if ref.dtype != torch.float32:
            raise TypeError("this compatibility stage requires a float32 B model")
        self.to(device=ref.device, dtype=ref.dtype)
        self.fast_model.eval()

    def train(self, mode: bool = True) -> SlowCore:
        super().train(mode)
        self.fast_model.eval()
        return self

    def new_parameters(self):
        for module in (self.slow_readouts, self.consolidators, self.slow_updaters):
            yield from module.parameters()

    def initialize_belief(self, state: Tensor) -> SlowBelief:
        self._state_action(state)
        fast = self.fast_model.initialize_belief(state)
        c = self.fast_model.config
        return SlowBelief(
            fast,
            state.new_zeros((state.shape[0], c.num_experts, c.fast_dim)),
            state.new_zeros((state.shape[0], c.num_experts, self.config.slow_dim)),
        )

    def _state_action(self, state: Tensor, action: Tensor | None = None) -> None:
        c = self.fast_model.config
        if not isinstance(state, Tensor) or state.ndim != 2 or state.shape[0] <= 0:
            raise ValueError("state must be a nonempty [batch,state_dim] tensor")
        _tensor(
            state,
            (state.shape[0], c.observation_dim),
            next(self.fast_model.parameters()),
            "state",
        )
        if action is not None:
            _tensor(action, (state.shape[0], c.action_dim), state, "action")

    def validate_belief(self, belief: SlowBelief, state: Tensor) -> None:
        if not isinstance(belief, SlowBelief):
            raise TypeError("belief must be SlowBelief")
        c = self.fast_model.config
        validate_stage1b1_fast_belief(
            belief.fast,
            leading_shape=state.shape[:-1],
            config=self.fast_model.accepted_model.config,
            device=state.device,
        )
        if (
            belief.fast.F.dtype != torch.float32
            or belief.fast.q_previous.dtype != torch.float32
        ):
            raise TypeError("fast belief must be float32")
        _tensor(belief.H, (state.shape[0], c.num_experts, c.fast_dim), state, "H")
        _tensor(
            belief.S, (state.shape[0], c.num_experts, self.config.slow_dim), state, "S"
        )
        if bool((belief.H.abs() > 1 + 1e-6).any()):
            raise ValueError("H coordinate bound exceeded")
        if bool((belief.S.norm(dim=-1) > self.config.slow_max_norm + 1e-6).any()):
            raise ValueError("S norm bound exceeded")

    def slow_correction(
        self, state: Tensor, action: Tensor, slow: Tensor, k: int
    ) -> Tensor:
        expert = self.slow_readouts[k]
        with_slow = torch.cat((state, action, slow), dim=-1)
        without_slow = torch.cat((state, action, torch.zeros_like(slow)), dim=-1)
        return expert(with_slow) - expert(without_slow)

    def predict_experts(
        self,
        state: Tensor,
        action: Tensor,
        belief: SlowBelief,
        *,
        mode: Mode = "full",
    ) -> SlowPrediction:
        _mode(mode)
        self._state_action(state, action)
        self.validate_belief(belief, state)
        fast = self.fast_model.predict_experts(state, action, belief.fast)
        if mode == "full":
            d_slow = torch.stack(
                [
                    self.slow_correction(state, action, belief.S[:, k], k)
                    for k in range(self.fast_model.config.num_experts)
                ],
                dim=-2,
            )
            d_hat, z_hat = fast.d_hat + d_slow, fast.z_hat + d_slow
        else:
            d_slow = torch.zeros_like(fast.d_hat)
            d_hat, z_hat = fast.d_hat, fast.z_hat
        return SlowPrediction(fast.d_base, fast.d_fast, d_slow, d_hat, z_hat)

    def update_slow(
        self,
        state: Tensor,
        action: Tensor,
        belief: SlowBelief,
        new_fast: Tensor,
    ) -> SlowWrite:
        self._state_action(state, action)
        self.validate_belief(belief, state)
        _tensor(new_fast, tuple(belief.fast.F.shape), state, "new_fast")
        c = self.config
        F_prior = self.fast_model.fast_retention * belief.fast.F
        actual_delta_F = new_fast - F_prior
        H_prior = c.rho * belief.H
        next_S, codes = [], []
        for k, (updater, mapping) in enumerate(
            zip(self.slow_updaters, self.consolidators)
        ):
            slow = belief.S[:, k]
            with_history = torch.cat(
                (c.fixed_slow_input_gain * H_prior[:, k], state, action), -1
            )
            without_history = torch.cat(
                (torch.zeros_like(H_prior[:, k]), state, action), -1
            )
            proposal = c.slow_step_limit * torch.tanh(
                updater(with_history, slow) - updater(without_history, slow)
            )
            next_S.append(
                clip_vectors_by_norm(
                    slow + proposal,
                    max_norm=c.slow_max_norm,
                    epsilon=c.epsilon,
                )
            )
            with_write = torch.cat((new_fast[:, k], actual_delta_F[:, k]), -1)
            no_write = torch.cat((F_prior[:, k], torch.zeros_like(F_prior[:, k])), -1)
            codes.append(
                (1 - c.fixed_ignore)
                * torch.tanh(mapping(with_write) - mapping(no_write))
            )
        S = torch.stack(next_S, dim=-2)
        code = torch.stack(codes, dim=-2)
        H = H_prior + (1 - c.rho) * code
        return SlowWrite(
            H, S, F_prior, actual_delta_F, H_prior, H - H_prior, S - belief.S, code
        )

    def post_action_update(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        route: PriorRoutingOutput,
        prediction: SlowPrediction,
        belief: SlowBelief,
        *,
        mode: Mode = "full",
    ) -> TransitionResult:

        _mode(mode)
        self._state_action(state, action)
        self.validate_belief(belief, state)
        _tensor(next_state, tuple(state.shape), state, "next_state")
        fast = self.fast_model.post_action_update(
            state,
            action,
            next_state,
            route,
            prediction.as_base_prediction(),
            belief.fast,
        )
        write = (
            None
            if mode == "off"
            else self.update_slow(state, action, belief, fast.belief.F)
        )
        result = SlowBelief(
            fast.belief,
            belief.H if write is None else write.H,
            belief.S if write is None else write.S,
        )
        self.validate_belief(result, state)
        return TransitionResult(result, route, prediction, fast, write)

    def observe_transition(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        belief: SlowBelief,
        *,
        mode: Mode = "full",
    ) -> TransitionResult:
        prediction = self.predict_experts(state, action, belief, mode=mode)
        route = self.fast_model.prior_routing(state, action, belief.fast)
        return self.post_action_update(
            state, action, next_state, route, prediction, belief, mode=mode
        )

    @torch.no_grad()
    def natural_no_current_error(
        self,
        state: Tensor,
        action: Tensor,
        belief: SlowBelief,
    ) -> SlowBelief:

        self._state_action(state, action)
        self.validate_belief(belief, state)
        route = self.fast_model.prior_routing(state, action, belief.fast)
        F = self.fast_model.fast_retention * belief.fast.F
        write = self.update_slow(state, action, belief, F)
        return SlowBelief(
            Stage1B1FastBelief(route.p.detach(), F), write.H, write.S
        ).detach()

    def burn_in(
        self, observations: Tensor, actions: Tensor, *, mode: Mode = "full"
    ) -> SlowBelief:
        if actions.ndim != 3 or observations.ndim != 3:
            raise ValueError("burn-in expects [batch,time,dim] tensors")
        if observations.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
            raise ValueError("burn-in requires exactly one more state than action")
        with torch.no_grad():
            belief = self.initialize_belief(observations[:, 0])
            for t in range(actions.shape[1]):
                belief = self.observe_transition(
                    observations[:, t],
                    actions[:, t],
                    observations[:, t + 1],
                    belief,
                    mode=mode,
                ).belief
        return belief.detach()

    def predict_future(
        self,
        start_state: Tensor,
        actions: Tensor,
        belief: SlowBelief,
        *,
        mode: Mode = "full",
    ) -> QueryPrediction:

        _mode(mode)
        self._state_action(start_state)
        self.validate_belief(belief, start_state)
        if (
            not isinstance(actions, Tensor)
            or actions.ndim != 3
            or actions.shape[1] <= 0
        ):
            raise ValueError("actions must be [batch,positive_horizon,action_dim]")
        _tensor(
            actions,
            (start_state.shape[0], actions.shape[1], self.fast_model.config.action_dim),
            start_state,
            "query actions",
        )
        route = self.fast_model.prior_routing(start_state, actions[:, 0], belief.fast)
        paths, fast_paths, slow_paths, corrections = [], [], [], []
        for k in range(self.fast_model.config.num_experts):
            state, F, S = start_state, belief.fast.F[:, k], belief.S[:, k]
            path, fs, ss, ds = [], [], [], []
            for t in range(actions.shape[1]):
                prediction = self.fast_model.predict_expert(state, actions[:, t], F, k)
                delta = (
                    self.slow_correction(state, actions[:, t], S, k)
                    if mode == "full"
                    else torch.zeros_like(prediction.z_hat)
                )
                state = prediction.z_hat + delta if mode == "full" else prediction.z_hat
                path.append(state)
                fs.append(F)
                ss.append(S)
                ds.append(delta)
                F = self.fast_model.fast_retention * F
            paths.append(torch.stack(path, 1))
            fast_paths.append(torch.stack(fs, 1))
            slow_paths.append(torch.stack(ss, 1))
            corrections.append(torch.stack(ds, 1))
        return QueryPrediction(
            torch.stack(paths, -2),
            torch.stack(fast_paths, -2),
            torch.stack(slow_paths, -2),
            torch.stack(corrections, -2),
            route.log_p,
        )


def score_future(
    query: QueryPrediction, targets: Tensor, *, sigma: float = 0.5
) -> Tensor:

    if isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    states = query.predicted_states
    B, H, K, D = states.shape
    _tensor(targets, (B, H, D), states, "targets")
    _tensor(query.route_log_p, (B, K), states, "route_log_p")
    error = states - targets.detach().unsqueeze(-2)
    costs = torch.zeros((B, K), dtype=torch.float64, device=states.device)
    for t in range(H):
        costs = costs + error[:, t].to(torch.float64).square().sum(-1) / (
            2 * sigma * sigma
        )
    logp = torch.log_softmax(query.route_log_p.to(torch.float64), dim=-1)
    return (-torch.logsumexp(logp - costs, -1) / (H * D)).to(query.route_log_p.dtype)


def relative_slow_penalty(
    adapted: SlowBelief, natural: SlowBelief, config: SlowConfig
) -> Tensor:

    if adapted.S.shape != natural.S.shape:
        raise ValueError("slow state shape mismatch")
    return (
        ((adapted.S - natural.S.detach()) / config.slow_step_limit).abs().mean((-2, -1))
    )
