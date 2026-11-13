from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from afterwm.lib.experts.models.belief import Stage1B0Belief
from afterwm.lib.experts.models.stage1a import (
    MLP,
    BaseExpertPrediction,
    PriorRoutingOutput,
    Stage1AForwardOutput,
)
from afterwm.lib.experts.models.stage1b0 import (
    Stage1B0WorldModel,
    clip_vectors_by_norm,
)


@dataclass(frozen=True, slots=True)
class Stage1B1FastModelConfig:

    observation_dim: int
    action_dim: int
    num_experts: int
    fast_dim: int = 32
    fast_hidden_dims: tuple[int, ...] = (128, 128)
    fast_retention: float = 0.8
    fast_innovation_limit: float = 1.0
    residual_clip_norm: float = 1.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("observation_dim", "action_dim", "num_experts", "fast_dim"):
            _positive_integer(getattr(self, name), name=name)
        hidden = tuple(self.fast_hidden_dims)
        if not hidden:
            raise ValueError("fast_hidden_dims must not be empty")
        for index, value in enumerate(hidden):
            _positive_integer(value, name=f"fast_hidden_dims[{index}]")
        _finite_open_interval(
            self.fast_retention,
            lower=0.0,
            upper=1.0,
            name="fast_retention",
        )
        _finite_open_closed_interval(
            self.fast_innovation_limit,
            lower=0.0,
            upper=1.0,
            name="fast_innovation_limit",
        )
        _positive_finite(self.residual_clip_norm, name="residual_clip_norm")
        _positive_finite(self.epsilon, name="epsilon")
        object.__setattr__(self, "fast_hidden_dims", hidden)
        object.__setattr__(self, "fast_retention", float(self.fast_retention))
        object.__setattr__(
            self,
            "fast_innovation_limit",
            float(self.fast_innovation_limit),
        )
        object.__setattr__(
            self,
            "residual_clip_norm",
            float(self.residual_clip_norm),
        )
        object.__setattr__(self, "epsilon", float(self.epsilon))

    @classmethod
    def from_q_only(
        cls,
        model: Stage1B0WorldModel,
        **overrides: object,
    ) -> Stage1B1FastModelConfig:
        if not isinstance(model, Stage1B0WorldModel):
            raise TypeError("model must be a Stage1B0WorldModel")
        if "residual_clip_norm" in overrides:
            raise ValueError(
                "residual_clip_norm is inherited from the accepted q-only model"
            )
        return cls(
            observation_dim=model.config.observation_dim,
            action_dim=model.config.action_dim,
            num_experts=model.config.num_experts,
            fast_dim=model.config.fast_dim,
            residual_clip_norm=model.config.residual_clip_norm,
            **overrides,
        )

    @property
    def per_expert_input_dim(self) -> int:
        return (2 * self.observation_dim) + self.action_dim + 2 + self.fast_dim


@dataclass(frozen=True, slots=True)
class Stage1B1FastBelief:

    q_previous: Tensor
    F: Tensor

    def detach(self) -> Stage1B1FastBelief:
        return Stage1B1FastBelief(
            q_previous=self.q_previous.detach(),
            F=self.F.detach(),
        )


@dataclass(frozen=True, slots=True)
class FastBeliefUpdate:
    F: Tensor
    innovation: Tensor
    raw_with_error: Tensor
    raw_without_error: Tensor
    evidence_error_raw: Tensor
    evidence_error_safe: Tensor


@dataclass(frozen=True, slots=True)
class Stage1B1FastUpdateDiagnostics:
    prior_p: Tensor
    posterior_q: Tensor
    log_p: Tensor
    log_q: Tensor
    posterior_logits: Tensor
    q_delta_logits: Tensor
    e_raw: Tensor
    e_safe: Tensor
    fast_evidence_error_raw: Tensor
    fast_evidence_error_safe: Tensor
    F_previous: Tensor
    F_next: Tensor
    F_innovation: Tensor


@dataclass(frozen=True, slots=True)
class Stage1B1FastPostActionOutput:
    belief: Stage1B1FastBelief
    diag: Stage1B1FastUpdateDiagnostics


class FastBeliefUpdater(nn.Module):


    def __init__(self, config: Stage1B1FastModelConfig) -> None:
        super().__init__()
        self.config = config
        self.network = MLP(
            config.per_expert_input_dim,
            config.fast_hidden_dims,
            config.fast_dim,
            layer_norm=True,
        )
        output = self.network.network[-1]
        if not isinstance(output, nn.Linear):
            raise RuntimeError("fast belief updater must end in a linear layer")
        nn.init.zeros_(output.weight)
        if output.bias is not None:
            nn.init.zeros_(output.bias)

    def build_input(
        self,
        state: Tensor,
        action: Tensor,
        prior_probability: Tensor,
        posterior_probability: Tensor,
        F_previous: Tensor,
        evidence_error: Tensor,
    ) -> Tensor:
        experts = self.config.num_experts
        expanded_state = state.unsqueeze(-2).expand(*state.shape[:-1], experts, -1)
        expanded_action = action.unsqueeze(-2).expand(*action.shape[:-1], experts, -1)
        return torch.cat(
            (
                expanded_state.to(dtype=torch.float32),
                expanded_action.to(dtype=torch.float32),
                prior_probability.to(dtype=torch.float32).unsqueeze(-1),
                posterior_probability.to(dtype=torch.float32).unsqueeze(-1),
                F_previous.to(dtype=torch.float32),
                evidence_error.to(dtype=torch.float32),
            ),
            dim=-1,
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        prior_probability: Tensor,
        posterior_probability: Tensor,
        F_previous: Tensor,
        evidence_error_raw: Tensor,
    ) -> FastBeliefUpdate:
        evidence_safe = clip_vectors_by_norm(
            evidence_error_raw,
            max_norm=self.config.residual_clip_norm,
            epsilon=self.config.epsilon,
        )
        with_error = self.build_input(
            state,
            action,
            prior_probability,
            posterior_probability,
            F_previous,
            evidence_safe,
        )
        without_error = self.build_input(
            state,
            action,
            prior_probability,
            posterior_probability,
            F_previous,
            torch.zeros_like(evidence_safe),
        )
        raw_with = self.network(with_error)
        raw_without = self.network(without_error)
        innovation = torch.tanh(raw_with - raw_without)
        retention = self.config.fast_retention
        write = (1.0 - retention) * self.config.fast_innovation_limit
        F_next = (retention * F_previous.to(dtype=torch.float32)) + (write * innovation)
        return FastBeliefUpdate(
            F=F_next,
            innovation=innovation,
            raw_with_error=raw_with,
            raw_without_error=raw_without,
            evidence_error_raw=evidence_error_raw,
            evidence_error_safe=evidence_safe,
        )


class Stage1B1FastWorldModel(nn.Module):

    def __init__(
        self,
        q_only_model: Stage1B0WorldModel,
        config: Stage1B1FastModelConfig,
    ) -> None:
        super().__init__()
        if not isinstance(q_only_model, Stage1B0WorldModel):
            raise TypeError("q_only_model must be a Stage1B0WorldModel")
        if not isinstance(config, Stage1B1FastModelConfig):
            raise TypeError("config must be Stage1B1FastModelConfig")
        _validate_q_only_config(q_only_model, config)
        self.q_only_model = q_only_model
        self.config = config
        self.fast_updater = FastBeliefUpdater(config)
        self._freeze_accepted_models()

    @property
    def base_model(self):
        return self.q_only_model.base_model

    def _freeze_accepted_models(self) -> None:
        self.q_only_model.requires_grad_(False)
        self.q_only_model.eval()

    def train(self, mode: bool = True) -> Stage1B1FastWorldModel:
        if not isinstance(mode, bool):
            raise TypeError("mode must be a boolean")
        super().train(mode)
        self.q_only_model.eval()
        return self

    def initialize_belief(self, reference_state: Tensor) -> Stage1B1FastBelief:
        _validate_reference_state(reference_state, self.config)
        leading = reference_state.shape[:-1]
        return Stage1B1FastBelief(
            q_previous=torch.full(
                (*leading, self.config.num_experts),
                1.0 / self.config.num_experts,
                dtype=torch.float32,
                device=reference_state.device,
            ),
            F=torch.zeros(
                (*leading, self.config.num_experts, self.config.fast_dim),
                dtype=torch.float32,
                device=reference_state.device,
            ),
        )

    def prior_routing(
        self,
        state: Tensor,
        action: Tensor,
        belief: Stage1B1FastBelief,
    ) -> PriorRoutingOutput:
        self._validate_inputs(state, action, belief)
        return self.base_model.prior_routing(state, action, belief.q_previous)

    def predict_base_experts(
        self,
        state: Tensor,
        action: Tensor,
    ) -> BaseExpertPrediction:
        return self.base_model.predict_base_experts(state, action)

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        belief: Stage1B1FastBelief,
    ) -> Stage1AForwardOutput:
        return Stage1AForwardOutput(
            route=self.prior_routing(state, action, belief),
            prediction=self.predict_base_experts(state, action),
        )

    def post_action_update(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        route: PriorRoutingOutput,
        prediction: BaseExpertPrediction,
        belief: Stage1B1FastBelief,
        *,
        fast_evidence_error_override: Tensor | None = None,
    ) -> Stage1B1FastPostActionOutput:


        self._validate_inputs(state, action, belief)
        _validate_next_and_predictions(next_state, state, prediction, self.config)
        e_raw = prediction.z_hat - next_state.unsqueeze(-2)
        e_safe = clip_vectors_by_norm(
            e_raw,
            max_norm=self.q_only_model.config.residual_clip_norm,
            epsilon=self.q_only_model.config.epsilon,
        )
        placeholder = _q_only_view(belief, self.q_only_model.config.slow_dim)
        raw_with, raw_without = self.q_only_model.belief_updater(
            state,
            action,
            route.p,
            placeholder,
            e_safe,
        )
        q_delta = self.q_only_model.config.logit_step_limit * torch.tanh(
            raw_with - raw_without
        )
        posterior_logits = route.log_p.to(dtype=torch.float32) + q_delta
        log_q = torch.log_softmax(posterior_logits, dim=-1)
        q = log_q.exp()

        fast_error = (
            e_raw
            if fast_evidence_error_override is None
            else fast_evidence_error_override
        )
        _validate_fast_override(fast_error, e_raw)
        fast = self.fast_updater(
            state,
            action,
            route.p,
            q,
            belief.F,
            fast_error,
        )
        next_belief = Stage1B1FastBelief(q_previous=q, F=fast.F)
        validate_stage1b1_fast_belief(
            next_belief,
            leading_shape=state.shape[:-1],
            config=self.config,
            device=state.device,
        )
        return Stage1B1FastPostActionOutput(
            belief=next_belief,
            diag=Stage1B1FastUpdateDiagnostics(
                prior_p=route.p,
                posterior_q=q,
                log_p=route.log_p,
                log_q=log_q,
                posterior_logits=posterior_logits,
                q_delta_logits=q_delta,
                e_raw=e_raw,
                e_safe=e_safe,
                fast_evidence_error_raw=fast.evidence_error_raw,
                fast_evidence_error_safe=fast.evidence_error_safe,
                F_previous=belief.F,
                F_next=fast.F,
                F_innovation=fast.innovation,
            ),
        )

    def _validate_inputs(
        self,
        state: Tensor,
        action: Tensor,
        belief: Stage1B1FastBelief,
    ) -> None:
        _validate_state_action(state, action, self.config)
        validate_stage1b1_fast_belief(
            belief,
            leading_shape=state.shape[:-1],
            config=self.config,
            device=state.device,
        )


def validate_stage1b1_fast_belief(
    belief: Stage1B1FastBelief,
    *,
    leading_shape: tuple[int, ...] | torch.Size,
    config: Stage1B1FastModelConfig,
    device: torch.device,
    probability_tolerance: float = 1e-5,
) -> None:
    if not isinstance(belief, Stage1B1FastBelief):
        raise TypeError("belief must be Stage1B1FastBelief")
    expected_q = (*tuple(leading_shape), config.num_experts)
    expected_F = (*tuple(leading_shape), config.num_experts, config.fast_dim)
    for name, value, shape in (
        ("q_previous", belief.q_previous, expected_q),
        ("F", belief.F, expected_F),
    ):
        if not isinstance(value, Tensor) or value.shape != shape:
            raise ValueError(f"belief.{name} shape must equal {shape}")
        if not value.is_floating_point() or value.device != device:
            raise TypeError(f"belief.{name} must be floating point on {device}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"belief.{name} must be finite")
    q = belief.q_previous.to(dtype=torch.float32)
    if bool((q <= 0.0).any()):
        raise ValueError("belief.q_previous must be strictly positive")
    if not torch.allclose(
        q.sum(dim=-1),
        torch.ones_like(q[..., 0]),
        atol=probability_tolerance,
        rtol=probability_tolerance,
    ):
        raise ValueError("belief.q_previous must sum to one")
    if bool((belief.F.abs() > (1.0 + 1e-6)).any()):
        raise ValueError("belief.F left its bounded [-1,1] range")


def _q_only_view(belief: Stage1B1FastBelief, slow_dim: int) -> Stage1B0Belief:
    leading = belief.q_previous.shape[:-1]
    experts = belief.q_previous.shape[-1]
    common = {"dtype": belief.F.dtype, "device": belief.F.device}
    return Stage1B0Belief(
        q_previous=belief.q_previous,
        F=belief.F,
        S=torch.zeros((*leading, experts, slow_dim), **common),
        H=torch.zeros_like(belief.F),
    )


def _validate_q_only_config(
    model: Stage1B0WorldModel,
    config: Stage1B1FastModelConfig,
) -> None:
    expected = (
        ("observation_dim", model.config.observation_dim, config.observation_dim),
        ("action_dim", model.config.action_dim, config.action_dim),
        ("num_experts", model.config.num_experts, config.num_experts),
        ("fast_dim", model.config.fast_dim, config.fast_dim),
        (
            "residual_clip_norm",
            model.config.residual_clip_norm,
            config.residual_clip_norm,
        ),
    )
    for name, old, new in expected:
        if old != new:
            raise ValueError(f"q-only {name}={old} does not match q+F {name}={new}")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("accepted Stage 1A and U_q must already be frozen")


def _validate_reference_state(
    state: Tensor,
    config: Stage1B1FastModelConfig,
) -> None:
    if not isinstance(state, Tensor) or state.shape[-1] != config.observation_dim:
        raise ValueError("reference_state has the wrong observation dimension")
    if not state.is_floating_point() or not bool(torch.isfinite(state).all()):
        raise TypeError("reference_state must be a finite floating-point tensor")


def _validate_state_action(
    state: Tensor,
    action: Tensor,
    config: Stage1B1FastModelConfig,
) -> None:
    if not isinstance(state, Tensor) or not isinstance(action, Tensor):
        raise TypeError("state and action must be tensors")
    if state.shape[-1] != config.observation_dim:
        raise ValueError("state has the wrong observation dimension")
    if action.shape[-1] != config.action_dim:
        raise ValueError("action has the wrong action dimension")
    if state.shape[:-1] != action.shape[:-1]:
        raise ValueError("state and action leading dimensions must match")
    if state.device != action.device or state.dtype != action.dtype:
        raise TypeError("state and action must share device and dtype")
    if not state.is_floating_point():
        raise TypeError("state and action must be floating point")
    if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(action).all()):
        raise ValueError("state and action must be finite")


def _validate_next_and_predictions(
    next_state: Tensor,
    state: Tensor,
    prediction: BaseExpertPrediction,
    config: Stage1B1FastModelConfig,
) -> None:
    if not isinstance(next_state, Tensor) or next_state.shape != state.shape:
        raise ValueError("next_state must match state")
    expected = (*state.shape[:-1], config.num_experts, config.observation_dim)
    if not isinstance(prediction, BaseExpertPrediction):
        raise TypeError("prediction must be BaseExpertPrediction")
    for name in ("d_base", "d_hat", "z_hat"):
        value = getattr(prediction, name)
        if value.shape != expected or value.device != state.device:
            raise ValueError(f"prediction.{name} has the wrong shape or device")


def _validate_fast_override(selected: Tensor, reference: Tensor) -> None:
    if not isinstance(selected, Tensor) or selected.shape != reference.shape:
        raise ValueError("fast evidence override must match Expert error shape")
    if selected.device != reference.device or selected.dtype != reference.dtype:
        raise TypeError("fast evidence override must share device and dtype")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("fast evidence override must be finite")


def _positive_integer(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_finite(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive finite")


def _finite_open_interval(
    value: float,
    *,
    lower: float,
    upper: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not lower < float(value) < upper:
        raise ValueError(f"{name} must be in ({lower},{upper})")


def _finite_open_closed_interval(
    value: float,
    *,
    lower: float,
    upper: float,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or not lower < float(value) <= upper:
        raise ValueError(f"{name} must be in ({lower},{upper}]")


__all__ = [
    "FastBeliefUpdate",
    "FastBeliefUpdater",
    "Stage1B1FastBelief",
    "Stage1B1FastModelConfig",
    "Stage1B1FastPostActionOutput",
    "Stage1B1FastUpdateDiagnostics",
    "Stage1B1FastWorldModel",
    "validate_stage1b1_fast_belief",
]
