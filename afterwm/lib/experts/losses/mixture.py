from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor
from torch.nn import functional

Reduction: TypeAlias = Literal["none", "mean", "sum"]


@dataclass(frozen=True, slots=True)
class MixtureNLLResult:

    loss: Tensor
    per_item_nll: Tensor
    component_nll: Tensor
    training_log_responsibility: Tensor
    training_responsibility: Tensor
    prediction_error: Tensor


@dataclass(frozen=True, slots=True)
class TrajectoryMixtureNLLResult:

    loss: Tensor
    per_item_nll: Tensor
    path_component_nll: Tensor
    per_step_component_nll: Tensor
    training_log_responsibility: Tensor
    training_responsibility: Tensor
    prediction_error: Tensor


def mixture_nll(
    predicted_next_states: Tensor,
    target_next_state: Tensor,
    log_mixing_probabilities: Tensor,
    *,
    sigma: float = 1.0,
    reduction: Reduction = "mean",
) -> MixtureNLLResult:


    _validate_transition_inputs(
        predicted_next_states,
        target_next_state,
        log_mixing_probabilities,
    )
    sigma = _validate_sigma(sigma)
    _validate_reduction(reduction)

    prediction_error = predicted_next_states - target_next_state.unsqueeze(-2)

    working_error, working_log_probabilities = _working_tensors(
        prediction_error,
        log_mixing_probabilities,
    )

    state_dim = working_error.shape[-1]

    component_nll = working_error.square().sum(dim=-1) / (2.0 * sigma * sigma)

    normalized_log_probabilities = _normalize_log_probabilities(
        working_log_probabilities
    )

    joint_log_score = normalized_log_probabilities - component_nll

    per_item_nll = -torch.logsumexp(joint_log_score, dim=-1) / state_dim

    training_log_responsibility = torch.log_softmax(
        joint_log_score,
        dim=-1,
    )
    training_responsibility = training_log_responsibility.exp()

    return MixtureNLLResult(
        loss=_reduce(per_item_nll, reduction),
        per_item_nll=per_item_nll,
        component_nll=component_nll,
        training_log_responsibility=training_log_responsibility,
        training_responsibility=training_responsibility,
        prediction_error=prediction_error,
    )


def trajectory_mixture_nll(
    predicted_future_states: Tensor,
    target_future_states: Tensor,
    log_initial_mixing_probabilities: Tensor,
    *,
    sigma: float = 1.0,
    reduction: Reduction = "mean",
) -> TrajectoryMixtureNLLResult:


    _validate_trajectory_inputs(
        predicted_future_states,
        target_future_states,
        log_initial_mixing_probabilities,
    )
    sigma = _validate_sigma(sigma)
    _validate_reduction(reduction)

    prediction_error = predicted_future_states - target_future_states.unsqueeze(-2)

    working_error, working_log_probabilities = _working_tensors(
        prediction_error,
        log_initial_mixing_probabilities,
    )

    horizon = working_error.shape[-3]
    state_dim = working_error.shape[-1]

    per_step_component_nll = working_error.square().sum(dim=-1) / (2.0 * sigma * sigma)

    path_component_nll = per_step_component_nll.sum(dim=-2)

    normalized_log_probabilities = _normalize_log_probabilities(
        working_log_probabilities
    )

    joint_log_score = normalized_log_probabilities - path_component_nll

    per_item_nll = -torch.logsumexp(joint_log_score, dim=-1) / (horizon * state_dim)

    training_log_responsibility = torch.log_softmax(
        joint_log_score,
        dim=-1,
    )
    training_responsibility = training_log_responsibility.exp()

    return TrajectoryMixtureNLLResult(
        loss=_reduce(per_item_nll, reduction),
        per_item_nll=per_item_nll,
        path_component_nll=path_component_nll,
        per_step_component_nll=per_step_component_nll,
        training_log_responsibility=training_log_responsibility,
        training_responsibility=training_responsibility,
        prediction_error=prediction_error,
    )


def masked_mode_cross_entropy(
    router_logits: Tensor,
    mode_id: Tensor,
    has_mode_label: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:


    _validate_masked_classification_inputs(
        router_logits,
        mode_id,
        has_mode_label,
    )
    _validate_reduction(reduction)

    selected_logits = router_logits[has_mode_label]
    selected_mode_id = mode_id[has_mode_label]

    if selected_mode_id.numel() == 0:
        if reduction == "none":
            return router_logits.new_empty((0,))

        return router_logits.sum() * 0.0

    _validate_selected_mode_ids(
        selected_mode_id,
        num_experts=router_logits.shape[-1],
    )

    return functional.cross_entropy(
        selected_logits,
        selected_mode_id,
        reduction=reduction,
    )


def masked_selected_expert_nll(
    component_nll: Tensor,
    mode_id: Tensor,
    has_mode_label: Tensor,
    *,
    reduction: Reduction = "mean",
) -> Tensor:

    _validate_masked_classification_inputs(
        component_nll,
        mode_id,
        has_mode_label,
        value_name="component_nll",
    )
    _validate_reduction(reduction)

    selected_component_nll = component_nll[has_mode_label]
    selected_mode_id = mode_id[has_mode_label]

    if selected_mode_id.numel() == 0:
        if reduction == "none":
            return component_nll.new_empty((0,))

        return component_nll.sum() * 0.0

    _validate_selected_mode_ids(
        selected_mode_id,
        num_experts=component_nll.shape[-1],
    )

    selected = selected_component_nll.gather(
        dim=-1,
        index=selected_mode_id.unsqueeze(-1),
    ).squeeze(-1)

    return _reduce(
        selected,
        reduction,
    )


def routing_balance_loss(
    probabilities: Tensor,
    target_usage: Tensor | None = None,
) -> Tensor:

    if not isinstance(probabilities, Tensor):
        raise TypeError("probabilities must be a torch.Tensor")

    if probabilities.ndim < 1 or probabilities.shape[-1] <= 0:
        raise ValueError("probabilities must have shape [..., K]")

    if not probabilities.is_floating_point():
        raise TypeError("probabilities must use a floating-point dtype")

    if torch.any(probabilities < 0):
        raise ValueError("probabilities must be non-negative")

    probability_sums = probabilities.sum(dim=-1)

    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("probabilities must sum to one along the Expert dimension")

    num_experts = probabilities.shape[-1]

    leading_dimensions = tuple(range(probabilities.ndim - 1))

    mean_usage = (
        probabilities.mean(dim=leading_dimensions)
        if leading_dimensions
        else probabilities
    )

    if target_usage is None:
        target = torch.full_like(
            mean_usage,
            1.0 / num_experts,
        )
    else:
        if not isinstance(target_usage, Tensor):
            raise TypeError("target_usage must be a torch.Tensor or None")

        if target_usage.shape != (num_experts,):
            raise ValueError(f"target_usage shape must equal ({num_experts},)")

        if not target_usage.is_floating_point():
            raise TypeError("target_usage must use a floating-point dtype")

        if target_usage.device != probabilities.device:
            raise ValueError("target_usage and probabilities must share a device")

        target = target_usage.to(dtype=probabilities.dtype)

        target_sum_is_one = torch.allclose(
            target.sum(),
            target.new_tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
        )

        if torch.any(target < 0) or not target_sum_is_one:
            raise ValueError("target_usage must be a probability distribution")

    return (mean_usage - target).square().mean()


def minimum_marginal_usage_loss(
    log_probabilities: Tensor,
    minimum_usage: float,
) -> Tensor:


    _validate_float_tensor(
        log_probabilities,
        name="log_probabilities",
        minimum_dimensions=1,
    )

    if log_probabilities.shape[-1] <= 0:
        raise ValueError("log_probabilities must have shape [..., K] with K > 0")

    if log_probabilities.numel() == 0:
        raise ValueError("log_probabilities must contain at least one item")

    if not torch.isfinite(log_probabilities).all():
        raise ValueError("log_probabilities must contain only finite values")

    num_experts = log_probabilities.shape[-1]
    minimum_usage = _validate_minimum_usage(
        minimum_usage,
        num_experts=num_experts,
    )

    log_probability_sums = torch.logsumexp(
        log_probabilities,
        dim=-1,
    )

    if not torch.allclose(
        log_probability_sums,
        torch.zeros_like(log_probability_sums),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError(
            "log_probabilities must be normalized along the Expert dimension"
        )

    working_log_probabilities = log_probabilities

    if working_log_probabilities.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        working_log_probabilities = working_log_probabilities.to(dtype=torch.float32)

    flattened = working_log_probabilities.reshape(
        -1,
        num_experts,
    )
    num_items = flattened.shape[0]

    log_mean_usage = torch.logsumexp(
        flattened,
        dim=0,
    ) - math.log(num_items)

    return torch.relu(math.log(minimum_usage) - log_mean_usage).mean()


def _validate_transition_inputs(
    predicted: Tensor,
    target: Tensor,
    log_probabilities: Tensor,
) -> None:
    _validate_float_tensor(
        predicted,
        name="predicted_next_states",
        minimum_dimensions=2,
    )
    _validate_float_tensor(
        target,
        name="target_next_state",
        minimum_dimensions=1,
    )
    _validate_float_tensor(
        log_probabilities,
        name="log_mixing_probabilities",
        minimum_dimensions=1,
    )

    if predicted.shape[-1] != target.shape[-1]:
        raise ValueError("predicted and target state dimensions must match")

    if predicted.shape[:-2] != target.shape[:-1]:
        raise ValueError("predicted and target leading dimensions must match")

    expected_log_shape = (
        *predicted.shape[:-2],
        predicted.shape[-2],
    )

    if log_probabilities.shape != expected_log_shape:
        raise ValueError(
            "log_mixing_probabilities shape must equal "
            f"{expected_log_shape}, "
            f"got {tuple(log_probabilities.shape)}"
        )

    _validate_common_device_and_prediction_dtype(
        predicted,
        target,
        log_probabilities,
    )


def _validate_trajectory_inputs(
    predicted: Tensor,
    target: Tensor,
    log_probabilities: Tensor,
) -> None:
    _validate_float_tensor(
        predicted,
        name="predicted_future_states",
        minimum_dimensions=3,
    )
    _validate_float_tensor(
        target,
        name="target_future_states",
        minimum_dimensions=2,
    )
    _validate_float_tensor(
        log_probabilities,
        name="log_initial_mixing_probabilities",
        minimum_dimensions=1,
    )

    if predicted.shape[-1] != target.shape[-1]:
        raise ValueError("predicted and target state dimensions must match")

    if predicted.shape[-3] != target.shape[-2]:
        raise ValueError("predicted and target horizons must match")

    if predicted.shape[:-3] != target.shape[:-2]:
        raise ValueError("predicted and target leading dimensions must match")

    expected_log_shape = (
        *predicted.shape[:-3],
        predicted.shape[-2],
    )

    if log_probabilities.shape != expected_log_shape:
        raise ValueError(
            "log_initial_mixing_probabilities "
            "shape must equal "
            f"{expected_log_shape}, "
            f"got {tuple(log_probabilities.shape)}"
        )

    _validate_common_device_and_prediction_dtype(
        predicted,
        target,
        log_probabilities,
    )


def _validate_masked_classification_inputs(
    values: Tensor,
    mode_id: Tensor,
    has_mode_label: Tensor,
    *,
    value_name: str = "router_logits",
) -> None:
    _validate_float_tensor(
        values,
        name=value_name,
        minimum_dimensions=1,
    )

    if not isinstance(mode_id, Tensor):
        raise TypeError("mode_id must be a torch.Tensor")

    if mode_id.dtype != torch.long:
        raise TypeError("mode_id must use torch.long dtype")

    if not isinstance(has_mode_label, Tensor):
        raise TypeError("has_mode_label must be a torch.Tensor")

    if has_mode_label.dtype != torch.bool:
        raise TypeError("has_mode_label must use torch.bool dtype")

    expected_shape = values.shape[:-1]

    if mode_id.shape != expected_shape:
        raise ValueError(f"mode_id shape must equal {expected_shape}")

    if has_mode_label.shape != expected_shape:
        raise ValueError(f"has_mode_label shape must equal {expected_shape}")

    same_device = values.device == mode_id.device == has_mode_label.device

    if not same_device:
        raise ValueError("values, mode_id and has_mode_label must share a device")


def _validate_selected_mode_ids(
    selected_mode_id: Tensor,
    *,
    num_experts: int,
) -> None:
    contains_negative_id = torch.any(selected_mode_id < 0)
    contains_large_id = torch.any(selected_mode_id >= num_experts)

    if contains_negative_id or contains_large_id:
        raise ValueError("labelled mode_id values must index an existing Expert")


def _validate_common_device_and_prediction_dtype(
    predicted: Tensor,
    target: Tensor,
    log_probabilities: Tensor,
) -> None:
    same_device = predicted.device == target.device == log_probabilities.device

    if not same_device:
        raise ValueError("prediction, target and log probabilities must share a device")

    if predicted.dtype != target.dtype:
        raise TypeError("prediction and target dtypes must match")


def _validate_float_tensor(
    value: Tensor,
    *,
    name: str,
    minimum_dimensions: int,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    if value.ndim < minimum_dimensions:
        raise ValueError(f"{name} must have at least {minimum_dimensions} dimensions")

    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _working_tensors(
    prediction_error: Tensor,
    log_probabilities: Tensor,
) -> tuple[Tensor, Tensor]:
    working_dtype = torch.promote_types(
        prediction_error.dtype,
        log_probabilities.dtype,
    )

    if working_dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        working_dtype = torch.float32

    return (
        prediction_error.to(dtype=working_dtype),
        log_probabilities.to(dtype=working_dtype),
    )


def _normalize_log_probabilities(
    log_probabilities: Tensor,
) -> Tensor:
    return log_probabilities - torch.logsumexp(
        log_probabilities,
        dim=-1,
        keepdim=True,
    )


def _reduce(
    values: Tensor,
    reduction: Reduction,
) -> Tensor:
    if reduction == "none":
        return values

    if reduction == "mean":
        return values.mean()

    return values.sum()


def _validate_sigma(
    value: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("sigma must be numeric")

    sigma = float(value)

    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")

    return sigma


def _validate_minimum_usage(
    value: float,
    *,
    num_experts: int,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("minimum_usage must be numeric")

    minimum_usage = float(value)
    maximum_usage = 1.0 / num_experts

    if not math.isfinite(minimum_usage) or not 0.0 < minimum_usage < maximum_usage:
        raise ValueError(
            "minimum_usage must be finite and lie strictly between "
            f"0 and 1 / K ({maximum_usage})"
        )

    return minimum_usage


def _validate_reduction(
    value: str,
) -> None:
    if value not in {
        "none",
        "mean",
        "sum",
    }:
        raise ValueError("reduction must be one of: none, mean, sum")
