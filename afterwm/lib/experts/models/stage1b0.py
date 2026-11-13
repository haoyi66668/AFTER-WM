from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from afterwm.lib.experts.models.belief import (
    Stage1B0Belief,
    initialize_stage1b0_belief,
    validate_stage1b0_belief,
)
from afterwm.lib.experts.models.stage1a import (
    MLP,
    BaseExpertPrediction,
    PriorRoutingOutput,
    Stage1ABaseWorldModel,
    Stage1AForwardOutput,
    Stage1AModelConfig,
)


@dataclass(frozen=True, slots=True)
class Stage1B0ModelConfig:

    observation_dim: int
    action_dim: int
    num_experts: int
    fast_dim: int = 32
    slow_dim: int = 32
    uq_hidden_dims: tuple[int, ...] = (128, 128)
    residual_clip_norm: float = 1.0
    logit_step_limit: float = 1.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        _require_positive_integer(self.observation_dim, name="observation_dim")
        _require_positive_integer(self.action_dim, name="action_dim")
        _require_positive_integer(self.num_experts, name="num_experts")
        _require_positive_integer(self.fast_dim, name="fast_dim")
        _require_positive_integer(self.slow_dim, name="slow_dim")

        hidden_dims = tuple(self.uq_hidden_dims)

        if not hidden_dims:
            raise ValueError("uq_hidden_dims must contain at least one hidden layer")

        for index, hidden_dim in enumerate(hidden_dims):
            _require_positive_integer(
                hidden_dim,
                name=f"uq_hidden_dims[{index}]",
            )

        _require_positive_finite(
            self.residual_clip_norm,
            name="residual_clip_norm",
        )
        _require_positive_finite(
            self.logit_step_limit,
            name="logit_step_limit",
        )
        _require_positive_finite(self.epsilon, name="epsilon")

        object.__setattr__(self, "uq_hidden_dims", hidden_dims)
        object.__setattr__(
            self,
            "residual_clip_norm",
            float(self.residual_clip_norm),
        )
        object.__setattr__(
            self,
            "logit_step_limit",
            float(self.logit_step_limit),
        )
        object.__setattr__(self, "epsilon", float(self.epsilon))

    @classmethod
    def from_stage1a(
        cls,
        config: Stage1AModelConfig,
        **overrides: object,
    ) -> Stage1B0ModelConfig:

        return cls(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            num_experts=config.num_experts,
            **overrides,
        )

    @property
    def updater_input_dim(self) -> int:

        return (
            self.observation_dim
            + self.action_dim
            + self.num_experts
            + self.num_experts
            * (self.observation_dim + (2 * self.fast_dim) + self.slow_dim)
        )

    @property
    def c_e(self) -> float:

        return self.residual_clip_norm

    @property
    def c_q(self) -> float:

        return self.logit_step_limit


@dataclass(frozen=True, slots=True)
class RoutingBeliefUpdate:

    q: Tensor
    log_q: Tensor
    logits: Tensor
    e_raw: Tensor
    evidence_error_raw: Tensor
    e_safe: Tensor
    delta_logits: Tensor


@dataclass(frozen=True, slots=True)
class Stage1B0UpdateDiagnostics:

    e_raw: Tensor
    evidence_error_raw: Tensor
    e_safe: Tensor
    prior_p: Tensor
    posterior_q: Tensor
    posterior_logits: Tensor
    log_p: Tensor
    log_q: Tensor
    delta_logits: Tensor

    @property
    def q(self) -> Tensor:

        return self.posterior_q


@dataclass(frozen=True, slots=True)
class Stage1B0PostActionOutput:

    belief: Stage1B0Belief
    diag: Stage1B0UpdateDiagnostics

    @property
    def routing(self) -> Stage1B0UpdateDiagnostics:

        return self.diag


class RoutingBeliefUpdater(nn.Module):

    def __init__(self, config: Stage1B0ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.network = MLP(
            config.updater_input_dim,
            config.uq_hidden_dims,
            config.num_experts,
            layer_norm=True,
        )

        output_layer = self.network.network[-1]

        if not isinstance(output_layer, nn.Linear):
            raise RuntimeError("belief updater must end with a linear output layer")

        nn.init.zeros_(output_layer.weight)

        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)

    @property
    def input_dim(self) -> int:
        return self.config.updater_input_dim

    def build_input(
        self,
        state: Tensor,
        action: Tensor,
        prior_probability: Tensor,
        belief: Stage1B0Belief,
        evidence_error: Tensor,
    ) -> Tensor:

        pieces = (
            state,
            action,
            prior_probability,
            belief.F.flatten(start_dim=-2),
            belief.S.flatten(start_dim=-2),
            belief.H.flatten(start_dim=-2),
            evidence_error.flatten(start_dim=-2),
        )

        return torch.cat(
            tuple(piece.to(dtype=torch.float32) for piece in pieces),
            dim=-1,
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        prior_probability: Tensor,
        belief: Stage1B0Belief,
        evidence_error: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with_error = self.build_input(
            state,
            action,
            prior_probability,
            belief,
            evidence_error,
        )
        without_error = self.build_input(
            state,
            action,
            prior_probability,
            belief,
            torch.zeros_like(evidence_error),
        )

        return self.network(with_error), self.network(without_error)


class Stage1B0WorldModel(nn.Module):

    def __init__(
        self,
        base_model: Stage1ABaseWorldModel,
        config: Stage1B0ModelConfig,
    ) -> None:
        super().__init__()

        if not isinstance(base_model, Stage1ABaseWorldModel):
            raise TypeError("base_model must be a Stage1ABaseWorldModel")

        _validate_base_config(base_model.config, config)

        self.base_model = base_model
        self.config = config
        self.belief_updater = RoutingBeliefUpdater(config)
        self._freeze_base_model()

    def _freeze_base_model(self) -> None:
        self.base_model.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True) -> Stage1B0WorldModel:

        if not isinstance(mode, bool):
            raise TypeError("mode must be a boolean")

        super().train(mode)
        self.base_model.eval()
        return self

    def initialize_belief(self, reference_state: Tensor) -> Stage1B0Belief:
        return initialize_stage1b0_belief(
            reference_state,
            num_experts=self.config.num_experts,
            fast_dim=self.config.fast_dim,
            slow_dim=self.config.slow_dim,
        )

    def prior_routing(
        self,
        state: Tensor,
        action: Tensor,
        belief: Stage1B0Belief,
    ) -> PriorRoutingOutput:
        self._validate_inputs(state, action, belief)

        return self.base_model.prior_routing(
            state,
            action,
            belief.q_previous,
        )

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
        belief: Stage1B0Belief,
    ) -> Stage1AForwardOutput:
        return Stage1AForwardOutput(
            route=self.prior_routing(state, action, belief),
            prediction=self.predict_base_experts(state, action),
        )

    def update_routing(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        route: PriorRoutingOutput,
        prediction: BaseExpertPrediction,
        belief: Stage1B0Belief,
        *,
        evidence_error_override: Tensor | None = None,
    ) -> RoutingBeliefUpdate:


        self._validate_inputs(state, action, belief)
        _validate_next_state(next_state, state)
        expected_error_shape = (
            *state.shape[:-1],
            self.config.num_experts,
            self.config.observation_dim,
        )
        _validate_route(route, state, self.config.num_experts)
        _validate_prediction(prediction, state, expected_error_shape)

        e_raw = prediction.z_hat - next_state.unsqueeze(-2)

        if evidence_error_override is None:
            evidence_error_raw = e_raw
        else:
            _validate_evidence_override(
                evidence_error_override,
                expected_shape=expected_error_shape,
                reference=e_raw,
            )
            evidence_error_raw = evidence_error_override

        e_safe = clip_vectors_by_norm(
            evidence_error_raw,
            max_norm=self.config.residual_clip_norm,
            epsilon=self.config.epsilon,
        )
        raw_with_error, raw_without_error = self.belief_updater(
            state,
            action,
            route.p,
            belief,
            e_safe,
        )
        delta_logits = self.config.logit_step_limit * torch.tanh(
            raw_with_error - raw_without_error
        )
        posterior_logits = route.log_p.to(dtype=torch.float32) + delta_logits.to(
            dtype=torch.float32
        )
        log_q = torch.log_softmax(posterior_logits, dim=-1)
        q = log_q.exp()

        return RoutingBeliefUpdate(
            q=q,
            log_q=log_q,
            logits=posterior_logits,
            e_raw=e_raw,
            evidence_error_raw=evidence_error_raw,
            e_safe=e_safe,
            delta_logits=delta_logits,
        )

    def post_action_update(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        route: PriorRoutingOutput,
        prediction: BaseExpertPrediction,
        belief: Stage1B0Belief,
        *,
        evidence_error_override: Tensor | None = None,
    ) -> Stage1B0PostActionOutput:
        route_update = self.update_routing(
            state,
            action,
            next_state,
            route,
            prediction,
            belief,
            evidence_error_override=evidence_error_override,
        )
        next_belief = Stage1B0Belief(
            q_previous=route_update.q,
            F=belief.F,
            S=belief.S,
            H=belief.H,
        )
        diag = Stage1B0UpdateDiagnostics(
            e_raw=route_update.e_raw,
            evidence_error_raw=route_update.evidence_error_raw,
            e_safe=route_update.e_safe,
            prior_p=route.p,
            posterior_q=route_update.q,
            posterior_logits=route_update.logits,
            log_p=route.log_p,
            log_q=route_update.log_q,
            delta_logits=route_update.delta_logits,
        )

        return Stage1B0PostActionOutput(
            belief=next_belief,
            diag=diag,
        )

    def _validate_inputs(
        self,
        state: Tensor,
        action: Tensor,
        belief: Stage1B0Belief,
    ) -> None:
        _validate_state_action(state, action, self.config)
        validate_stage1b0_belief(
            belief,
            leading_shape=state.shape[:-1],
            num_experts=self.config.num_experts,
            fast_dim=self.config.fast_dim,
            slow_dim=self.config.slow_dim,
            device=state.device,
            require_zero_placeholders=True,
        )


def clip_vectors_by_norm(
    values: Tensor,
    *,
    max_norm: float,
    epsilon: float = 1e-8,
) -> Tensor:

    if not isinstance(values, Tensor):
        raise TypeError("values must be a torch.Tensor")

    if not values.is_floating_point():
        raise TypeError("values must use a floating-point dtype")

    _require_positive_finite(max_norm, name="max_norm")
    _require_positive_finite(epsilon, name="epsilon")

    norms = torch.linalg.vector_norm(values, ord=2, dim=-1, keepdim=True)
    scale = torch.clamp(
        float(max_norm) / (norms + float(epsilon)),
        max=1.0,
    )
    return values * scale


def _validate_base_config(
    base: Stage1AModelConfig,
    stage1b0: Stage1B0ModelConfig,
) -> None:
    values = (
        ("observation_dim", base.observation_dim, stage1b0.observation_dim),
        ("action_dim", base.action_dim, stage1b0.action_dim),
        ("num_experts", base.num_experts, stage1b0.num_experts),
    )

    for name, base_value, stage1b0_value in values:
        if base_value != stage1b0_value:
            raise ValueError(
                f"Stage1B-0 {name}={stage1b0_value} does not match "
                f"Stage1A {name}={base_value}"
            )


def _validate_state_action(
    state: Tensor,
    action: Tensor,
    config: Stage1B0ModelConfig,
) -> None:
    if not isinstance(state, Tensor):
        raise TypeError("state must be a torch.Tensor")

    if not isinstance(action, Tensor):
        raise TypeError("action must be a torch.Tensor")

    if state.ndim < 1 or state.shape[-1] != config.observation_dim:
        raise ValueError(f"state last dimension must equal {config.observation_dim}")

    if action.ndim < 1 or action.shape[-1] != config.action_dim:
        raise ValueError(f"action last dimension must equal {config.action_dim}")

    if state.shape[:-1] != action.shape[:-1]:
        raise ValueError("state and action leading dimensions must match")

    if not state.is_floating_point() or not action.is_floating_point():
        raise TypeError("state and action must use floating-point dtypes")

    if state.dtype != action.dtype:
        raise TypeError("state and action dtypes must match")

    if state.device != action.device:
        raise ValueError("state and action must be on the same device")


def _validate_next_state(next_state: Tensor, state: Tensor) -> None:
    if not isinstance(next_state, Tensor):
        raise TypeError("next_state must be a torch.Tensor")

    if next_state.shape != state.shape:
        raise ValueError(
            f"next_state shape must equal state shape {state.shape}, "
            f"got {next_state.shape}"
        )

    if next_state.dtype != state.dtype:
        raise TypeError("next_state and state dtypes must match")

    if next_state.device != state.device:
        raise ValueError("next_state and state must be on the same device")


def _validate_route(
    route: PriorRoutingOutput,
    state: Tensor,
    num_experts: int,
) -> None:
    if not isinstance(route, PriorRoutingOutput):
        raise TypeError("route must be a PriorRoutingOutput")

    expected_shape = (*state.shape[:-1], num_experts)

    for name in ("p", "log_p"):
        value = getattr(route, name)

        if value.shape != expected_shape:
            raise ValueError(
                f"route.{name} shape must equal {expected_shape}, got {value.shape}"
            )

        if value.device != state.device:
            raise ValueError(f"route.{name} must be on the same device as state")


def _validate_prediction(
    prediction: BaseExpertPrediction,
    state: Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    if not isinstance(prediction, BaseExpertPrediction):
        raise TypeError("prediction must be a BaseExpertPrediction")

    if prediction.z_hat.shape != expected_shape:
        raise ValueError(
            f"prediction.z_hat shape must equal {expected_shape}, "
            f"got {prediction.z_hat.shape}"
        )

    if prediction.z_hat.device != state.device:
        raise ValueError("prediction.z_hat must be on the same device as state")


def _validate_evidence_override(
    evidence_error_override: Tensor,
    *,
    expected_shape: tuple[int, ...],
    reference: Tensor,
) -> None:
    if not isinstance(evidence_error_override, Tensor):
        raise TypeError("evidence_error_override must be a torch.Tensor")

    if evidence_error_override.shape != expected_shape:
        raise ValueError(
            "evidence_error_override shape must equal "
            f"{expected_shape}, got {evidence_error_override.shape}"
        )

    if not evidence_error_override.is_floating_point():
        raise TypeError("evidence_error_override must use a floating-point dtype")

    if evidence_error_override.dtype != reference.dtype:
        raise TypeError("evidence_error_override and e_raw dtypes must match")

    if evidence_error_override.device != reference.device:
        raise ValueError("evidence_error_override and e_raw must share a device")


def _require_positive_integer(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_positive_finite(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")

    if not math.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
