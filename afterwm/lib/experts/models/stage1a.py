from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class Stage1AModelConfig:

    observation_dim: int
    action_dim: int
    num_experts: int
    router_hidden_dims: tuple[int, ...] = (128, 128)
    expert_hidden_dims: tuple[int, ...] = (256, 256, 256)
    alpha_q: float = 0.5
    route_probability_floor: float = 1e-3
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        _require_positive_integer(
            self.observation_dim,
            name="observation_dim",
        )
        _require_positive_integer(
            self.action_dim,
            name="action_dim",
        )
        _require_positive_integer(
            self.num_experts,
            name="num_experts",
        )

        router_hidden_dims = _normalize_hidden_dims(
            self.router_hidden_dims,
            name="router_hidden_dims",
        )
        expert_hidden_dims = _normalize_hidden_dims(
            self.expert_hidden_dims,
            name="expert_hidden_dims",
        )

        _require_closed_unit_interval(
            self.alpha_q,
            name="alpha_q",
        )
        _require_open_unit_interval(
            self.route_probability_floor,
            name="route_probability_floor",
        )
        _require_positive_finite(
            self.epsilon,
            name="epsilon",
        )

        object.__setattr__(
            self,
            "router_hidden_dims",
            router_hidden_dims,
        )
        object.__setattr__(
            self,
            "expert_hidden_dims",
            expert_hidden_dims,
        )
        object.__setattr__(
            self,
            "alpha_q",
            float(self.alpha_q),
        )
        object.__setattr__(
            self,
            "route_probability_floor",
            float(self.route_probability_floor),
        )
        object.__setattr__(
            self,
            "epsilon",
            float(self.epsilon),
        )


@dataclass(frozen=True, slots=True)
class PriorRoutingOutput:

    logits: Tensor
    log_p: Tensor
    p: Tensor
    raw_logits: Tensor
    router_logits: Tensor


@dataclass(frozen=True, slots=True)
class BaseExpertPrediction:

    d_base: Tensor
    d_hat: Tensor
    z_hat: Tensor


@dataclass(frozen=True, slots=True)
class Stage1AForwardOutput:

    route: PriorRoutingOutput
    prediction: BaseExpertPrediction


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        *,
        layer_norm: bool,
    ) -> None:
        super().__init__()

        _require_positive_integer(
            input_dim,
            name="input_dim",
        )
        _require_positive_integer(
            output_dim,
            name="output_dim",
        )

        normalized_hidden_dims = _normalize_hidden_dims(
            hidden_dims,
            name="hidden_dims",
        )

        layers: list[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in normalized_hidden_dims:
            layers.append(
                nn.Linear(
                    current_dim,
                    hidden_dim,
                )
            )

            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.SiLU())
            current_dim = hidden_dim

        layers.append(
            nn.Linear(
                current_dim,
                output_dim,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        return self.network(inputs)


class Stage1ABaseWorldModel(nn.Module):

    def __init__(
        self,
        config: Stage1AModelConfig,
    ) -> None:
        super().__init__()

        self.config = config

        joint_input_dim = config.observation_dim + config.action_dim

        self.router = MLP(
            joint_input_dim,
            config.router_hidden_dims,
            config.num_experts,
            layer_norm=True,
        )

        router_output_layer = self.router.network[-1]

        if not isinstance(router_output_layer, nn.Linear):
            raise RuntimeError("router must end with a linear output layer")

        nn.init.zeros_(router_output_layer.weight)

        if router_output_layer.bias is not None:
            nn.init.zeros_(router_output_layer.bias)

        self.g0_experts = nn.ModuleList(
            MLP(
                joint_input_dim,
                config.expert_hidden_dims,
                config.observation_dim,
                layer_norm=False,
            )
            for _ in range(config.num_experts)
        )

    def uniform_q(
        self,
        state: Tensor,
    ) -> Tensor:

        _validate_state_only(
            state,
            expected_dim=(self.config.observation_dim),
        )

        leading_shape = state.shape[:-1]

        return torch.full(
            (
                *leading_shape,
                self.config.num_experts,
            ),
            fill_value=(1.0 / self.config.num_experts),
            dtype=torch.float32,
            device=state.device,
        )

    def prior_routing(
        self,
        state: Tensor,
        action: Tensor,
        q_previous: Tensor | None = None,
    ) -> PriorRoutingOutput:

        joint_input = self._joint_input(
            state,
            action,
        )

        if q_previous is None:
            q_previous = self.uniform_q(state)

        _validate_q_previous(
            q_previous,
            leading_shape=state.shape[:-1],
            num_experts=(self.config.num_experts),
            device=state.device,
        )

        router_logits = self.router(joint_input)

        q_memory = (1.0 - self.config.route_probability_floor) * q_previous.to(
            dtype=torch.float32
        ) + (self.config.route_probability_floor / self.config.num_experts)

        memory_logits = self.config.alpha_q * torch.log(
            q_memory.clamp_min(self.config.epsilon)
        )

        raw_logits = router_logits.to(dtype=torch.float32) + memory_logits

        raw_log_p = torch.log_softmax(
            raw_logits,
            dim=-1,
        )

        raw_probability = raw_log_p.exp()

        probability = (1.0 - self.config.route_probability_floor) * raw_probability + (
            self.config.route_probability_floor / self.config.num_experts
        )

        log_probability = torch.log(probability)

        return PriorRoutingOutput(
            logits=log_probability,
            log_p=log_probability,
            p=probability,
            raw_logits=raw_logits,
            router_logits=router_logits,
        )

    def predict_base_experts(
        self,
        state: Tensor,
        action: Tensor,
    ) -> BaseExpertPrediction:

        joint_input = self._joint_input(
            state,
            action,
        )

        base_deltas = torch.stack(
            [expert(joint_input) for expert in self.g0_experts],
            dim=-2,
        )

        next_states = state.unsqueeze(-2) + base_deltas

        return BaseExpertPrediction(
            d_base=base_deltas,
            d_hat=base_deltas,
            z_hat=next_states,
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        q_previous: Tensor | None = None,
    ) -> Stage1AForwardOutput:
        route = self.prior_routing(
            state,
            action,
            q_previous,
        )

        prediction = self.predict_base_experts(
            state,
            action,
        )

        return Stage1AForwardOutput(
            route=route,
            prediction=prediction,
        )

    def _joint_input(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        _validate_state_action(
            state,
            action,
            observation_dim=(self.config.observation_dim),
            action_dim=(self.config.action_dim),
        )

        return torch.cat(
            (state, action),
            dim=-1,
        )


def _validate_state_action(
    state: Tensor,
    action: Tensor,
    *,
    observation_dim: int,
    action_dim: int,
) -> None:
    _validate_state_only(
        state,
        expected_dim=observation_dim,
    )

    if not isinstance(action, Tensor):
        raise TypeError("action must be a torch.Tensor")

    if action.ndim < 1:
        raise ValueError("action must have at least one dimension")

    if action.shape[-1] != action_dim:
        raise ValueError(
            f"action last dimension must equal {action_dim}, got {action.shape[-1]}"
        )

    if state.shape[:-1] != action.shape[:-1]:
        raise ValueError("state and action leading dimensions must match")

    if not action.is_floating_point():
        raise TypeError("action must use a floating-point dtype")

    if state.dtype != action.dtype:
        raise TypeError("state and action dtypes must match")

    if state.device != action.device:
        raise ValueError("state and action must be on the same device")


def _validate_state_only(
    state: Tensor,
    *,
    expected_dim: int,
) -> None:
    if not isinstance(state, Tensor):
        raise TypeError("state must be a torch.Tensor")

    if state.ndim < 1:
        raise ValueError("state must have at least one dimension")

    if state.shape[-1] != expected_dim:
        raise ValueError(
            f"state last dimension must equal {expected_dim}, got {state.shape[-1]}"
        )

    if not state.is_floating_point():
        raise TypeError("state must use a floating-point dtype")


def _validate_q_previous(
    q_previous: Tensor,
    *,
    leading_shape: torch.Size,
    num_experts: int,
    device: torch.device,
) -> None:
    if not isinstance(
        q_previous,
        Tensor,
    ):
        raise TypeError("q_previous must be a torch.Tensor")

    expected_shape = (
        *leading_shape,
        num_experts,
    )

    if q_previous.shape != expected_shape:
        raise ValueError(
            f"q_previous shape must equal {expected_shape}, got {q_previous.shape}"
        )

    if not q_previous.is_floating_point():
        raise TypeError("q_previous must use a floating-point dtype")

    if q_previous.device != device:
        raise ValueError("q_previous and state must be on the same device")


def _normalize_hidden_dims(
    values: tuple[int, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    normalized = tuple(values)

    if not normalized:
        raise ValueError(f"{name} must contain at least one hidden layer")

    for index, value in enumerate(normalized):
        _require_positive_integer(
            value,
            name=f"{name}[{index}]",
        )

    return normalized


def _require_positive_integer(
    value: int,
    *,
    name: str,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_closed_unit_interval(
    value: float,
    *,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        int | float,
    ):
        raise TypeError(f"{name} must be numeric")

    if not math.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")


def _require_open_unit_interval(
    value: float,
    *,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        int | float,
    ):
        raise TypeError(f"{name} must be numeric")

    if not math.isfinite(value) or not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")


def _require_positive_finite(
    value: float,
    *,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        int | float,
    ):
        raise TypeError(f"{name} must be numeric")

    if not math.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
