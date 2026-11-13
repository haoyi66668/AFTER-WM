from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from afterwm.lib.experts.models.stage1a import (
    MLP,
    BaseExpertPrediction,
    PriorRoutingOutput,
)
from afterwm.lib.experts.models.stage1b1 import (
    Stage1B1FastBelief,
    Stage1B1FastModelConfig,
    Stage1B1FastPostActionOutput,
    Stage1B1FastWorldModel,
    validate_stage1b1_fast_belief,
)


@dataclass(frozen=True, slots=True)
class Stage1B2FastReadoutModelConfig:

    observation_dim: int
    action_dim: int
    num_experts: int
    fast_dim: int
    fast_readout_hidden_dims: tuple[int, ...] = (128, 128)

    def __post_init__(self) -> None:
        for name in ("observation_dim", "action_dim", "num_experts", "fast_dim"):
            _positive_integer(getattr(self, name), name=name)
        hidden = tuple(self.fast_readout_hidden_dims)
        if not hidden:
            raise ValueError("fast_readout_hidden_dims must not be empty")
        for index, value in enumerate(hidden):
            _positive_integer(value, name=f"fast_readout_hidden_dims[{index}]")
        object.__setattr__(self, "fast_readout_hidden_dims", hidden)

    @classmethod
    def from_q_plus_f(
        cls,
        model: Stage1B1FastWorldModel,
        **overrides: object,
    ) -> Stage1B2FastReadoutModelConfig:
        if not isinstance(model, Stage1B1FastWorldModel):
            raise TypeError("model must be a Stage1B1FastWorldModel")
        forbidden = {
            "observation_dim",
            "action_dim",
            "num_experts",
            "fast_dim",
        }.intersection(overrides)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise ValueError(f"accepted q+F dimensions cannot be overridden: {names}")
        return cls(
            observation_dim=model.config.observation_dim,
            action_dim=model.config.action_dim,
            num_experts=model.config.num_experts,
            fast_dim=model.config.fast_dim,
            **overrides,
        )

    @property
    def per_expert_input_dim(self) -> int:
        return self.observation_dim + self.action_dim + self.fast_dim


@dataclass(frozen=True, slots=True)
class FastReadoutExpertPrediction:

    d_base: Tensor
    d_fast: Tensor
    d_hat: Tensor
    z_hat: Tensor

    def as_base_prediction(self) -> BaseExpertPrediction:

        return BaseExpertPrediction(
            d_base=self.d_base,
            d_hat=self.d_hat,
            z_hat=self.z_hat,
        )


@dataclass(frozen=True, slots=True)
class Stage1B2FastReadoutForwardOutput:
    route: PriorRoutingOutput
    prediction: FastReadoutExpertPrediction


class Stage1B2FastReadoutWorldModel(nn.Module):


    def __init__(
        self,
        accepted_model: Stage1B1FastWorldModel,
        config: Stage1B2FastReadoutModelConfig,
    ) -> None:
        super().__init__()
        if not isinstance(accepted_model, Stage1B1FastWorldModel):
            raise TypeError("accepted_model must be a Stage1B1FastWorldModel")
        if not isinstance(config, Stage1B2FastReadoutModelConfig):
            raise TypeError("config must be Stage1B2FastReadoutModelConfig")
        _validate_accepted_config(accepted_model.config, config)
        if any(parameter.requires_grad for parameter in accepted_model.parameters()):
            raise ValueError("accepted Stage 1A, U_q, and U_F must already be frozen")
        if any(parameter.grad is not None for parameter in accepted_model.parameters()):
            raise ValueError("accepted q+F model has stale gradients")

        self.accepted_model = accepted_model
        self.config = config
        self.fast_readout_experts = nn.ModuleList(
            MLP(
                config.per_expert_input_dim,
                config.fast_readout_hidden_dims,
                config.observation_dim,
                layer_norm=False,
            )
            for _ in range(config.num_experts)
        )
        for expert in self.fast_readout_experts:
            output = expert.network[-1]
            if not isinstance(output, nn.Linear):
                raise RuntimeError("fast readout must end in a linear layer")
            bias_free_output = nn.Linear(
                output.in_features,
                output.out_features,
                bias=False,
                device=output.weight.device,
                dtype=output.weight.dtype,
            )
            nn.init.zeros_(bias_free_output.weight)
            expert.network[-1] = bias_free_output
        self._freeze_accepted_model()

    @property
    def base_model(self):
        return self.accepted_model.base_model

    @property
    def q_only_model(self):
        return self.accepted_model.q_only_model

    @property
    def fast_updater(self):
        return self.accepted_model.fast_updater

    @property
    def fast_retention(self) -> float:
        return self.accepted_model.config.fast_retention

    def _freeze_accepted_model(self) -> None:
        self.accepted_model.requires_grad_(False)
        self.accepted_model.eval()

    def train(self, mode: bool = True) -> Stage1B2FastReadoutWorldModel:
        if not isinstance(mode, bool):
            raise TypeError("mode must be a boolean")
        super().train(mode)
        self.accepted_model.eval()
        return self

    def initialize_belief(self, reference_state: Tensor) -> Stage1B1FastBelief:
        return self.accepted_model.initialize_belief(reference_state)

    def prior_routing(
        self,
        state: Tensor,
        action: Tensor,
        belief: Stage1B1FastBelief,
    ) -> PriorRoutingOutput:
        return self.accepted_model.prior_routing(state, action, belief)

    def predict_base_experts(
        self,
        state: Tensor,
        action: Tensor,
    ) -> BaseExpertPrediction:
        return self.accepted_model.predict_base_experts(state, action)

    def predict_expert(
        self,
        state: Tensor,
        action: Tensor,
        fast_state: Tensor,
        expert_index: int,
        *,
        enable_fast_readout: bool = True,
    ) -> FastReadoutExpertPrediction:

        _validate_state_action(state, action, self.config)
        if isinstance(expert_index, bool) or not isinstance(expert_index, int):
            raise TypeError("expert_index must be an integer")
        if not 0 <= expert_index < self.config.num_experts:
            raise ValueError("expert_index is out of range")
        expected_fast = (*state.shape[:-1], self.config.fast_dim)
        _validate_float_tensor(
            fast_state,
            shape=expected_fast,
            device=state.device,
            name="fast_state",
        )
        if not isinstance(enable_fast_readout, bool):
            raise TypeError("enable_fast_readout must be a boolean")

        joint = torch.cat((state, action), dim=-1)
        d_base = self.base_model.g0_experts[expert_index](joint)
        if enable_fast_readout:
            expert = self.fast_readout_experts[expert_index]
            with_fast = torch.cat((state, action, fast_state), dim=-1)
            without_fast = torch.cat(
                (state, action, torch.zeros_like(fast_state)), dim=-1
            )
            d_fast = expert(with_fast) - expert(without_fast)
        else:
            d_fast = torch.zeros_like(d_base)
        d_hat = d_base + d_fast
        return FastReadoutExpertPrediction(
            d_base=d_base,
            d_fast=d_fast,
            d_hat=d_hat,
            z_hat=state + d_hat,
        )

    def predict_experts(
        self,
        state: Tensor,
        action: Tensor,
        belief: Stage1B1FastBelief,
        *,
        enable_fast_readout: bool = True,
        fast_state_override: Tensor | None = None,
    ) -> FastReadoutExpertPrediction:


        _validate_state_action(state, action, self.config)
        validate_stage1b1_fast_belief(
            belief,
            leading_shape=state.shape[:-1],
            config=self.accepted_model.config,
            device=state.device,
        )
        if not isinstance(enable_fast_readout, bool):
            raise TypeError("enable_fast_readout must be a boolean")
        fast_state = belief.F if fast_state_override is None else fast_state_override
        expected_fast = (
            *state.shape[:-1],
            self.config.num_experts,
            self.config.fast_dim,
        )
        _validate_float_tensor(
            fast_state,
            shape=expected_fast,
            device=state.device,
            name="fast_state_override"
            if fast_state_override is not None
            else "belief.F",
        )
        predictions = [
            self.predict_expert(
                state,
                action,
                fast_state[..., expert_index, :],
                expert_index,
                enable_fast_readout=enable_fast_readout,
            )
            for expert_index in range(self.config.num_experts)
        ]
        return FastReadoutExpertPrediction(
            d_base=torch.stack([value.d_base for value in predictions], dim=-2),
            d_fast=torch.stack([value.d_fast for value in predictions], dim=-2),
            d_hat=torch.stack([value.d_hat for value in predictions], dim=-2),
            z_hat=torch.stack([value.z_hat for value in predictions], dim=-2),
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        belief: Stage1B1FastBelief,
        *,
        enable_fast_readout: bool = True,
    ) -> Stage1B2FastReadoutForwardOutput:
        return Stage1B2FastReadoutForwardOutput(
            route=self.prior_routing(state, action, belief),
            prediction=self.predict_experts(
                state,
                action,
                belief,
                enable_fast_readout=enable_fast_readout,
            ),
        )

    def post_action_update(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        route: PriorRoutingOutput,
        prediction: BaseExpertPrediction | FastReadoutExpertPrediction,
        belief: Stage1B1FastBelief,
        *,
        fast_evidence_error_override: Tensor | None = None,
    ) -> Stage1B1FastPostActionOutput:


        accepted_prediction = (
            prediction.as_base_prediction()
            if isinstance(prediction, FastReadoutExpertPrediction)
            else prediction
        )
        if not isinstance(accepted_prediction, BaseExpertPrediction):
            raise TypeError("prediction must be a fast-readout or base prediction")

        return self.accepted_model.post_action_update(
            state,
            action,
            next_state,
            route,
            accepted_prediction,
            belief,
            fast_evidence_error_override=fast_evidence_error_override,
        )

    def post_action_update_readout_only(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        route: PriorRoutingOutput,
        belief: Stage1B1FastBelief,
        *,
        fast_evidence_error_override: Tensor | None = None,
    ) -> Stage1B1FastPostActionOutput:

        base_prediction = self.predict_base_experts(state, action)
        return self.accepted_model.post_action_update(
            state,
            action,
            next_state,
            route,
            base_prediction,
            belief,
            fast_evidence_error_override=fast_evidence_error_override,
        )


def _validate_accepted_config(
    accepted: Stage1B1FastModelConfig,
    config: Stage1B2FastReadoutModelConfig,
) -> None:
    for name in ("observation_dim", "action_dim", "num_experts", "fast_dim"):
        if getattr(accepted, name) != getattr(config, name):
            raise ValueError(f"accepted q+F {name} does not match readout config")


def _validate_state_action(
    state: Tensor,
    action: Tensor,
    config: Stage1B2FastReadoutModelConfig,
) -> None:
    if not isinstance(state, Tensor) or not isinstance(action, Tensor):
        raise TypeError("state and action must be tensors")
    if state.shape[:-1] != action.shape[:-1]:
        raise ValueError("state and action leading dimensions must match")
    if state.shape[-1:] != (config.observation_dim,):
        raise ValueError("state has the wrong observation dimension")
    if action.shape[-1:] != (config.action_dim,):
        raise ValueError("action has the wrong action dimension")
    if state.device != action.device or state.dtype != action.dtype:
        raise TypeError("state and action must share device and dtype")
    if not state.is_floating_point() or not action.is_floating_point():
        raise TypeError("state and action must be floating point")
    if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(action).all()):
        raise ValueError("state and action must be finite")


def _validate_float_tensor(
    value: Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    name: str,
) -> None:
    if not isinstance(value, Tensor) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_floating_point() or value.device != device:
        raise TypeError(f"{name} must be floating point on {device}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _positive_finite(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive finite")


__all__ = [
    "FastReadoutExpertPrediction",
    "Stage1B2FastReadoutForwardOutput",
    "Stage1B2FastReadoutModelConfig",
    "Stage1B2FastReadoutWorldModel",
]
