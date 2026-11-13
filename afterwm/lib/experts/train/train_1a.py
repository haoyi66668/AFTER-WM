from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from afterwm.lib.experts.losses import (
    masked_mode_cross_entropy,
    masked_selected_expert_nll,
    minimum_marginal_usage_loss,
    mixture_nll,
    routing_balance_loss,
    trajectory_mixture_nll,
)
from afterwm.lib.experts.models import Stage1ABaseWorldModel, rollout_base_experts


@dataclass(frozen=True, slots=True)
class Stage1ATrainConfig:

    sigma: float = 1.0
    rollout_weight: float = 1.0
    mode_weight: float = 0.0
    balance_weight: float = 0.0
    responsibility_alive_weight: float = 0.0
    minimum_responsibility_usage: float | None = None
    gradient_clip_norm: float = 10.0
    supervise_router_mode: bool = False
    mode_to_expert: tuple[tuple[int, int], ...] = ()
    target_usage: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _validate_scalar(self.sigma, "sigma", lower=0.0, strict=True)
        _validate_scalar(self.rollout_weight, "rollout_weight", lower=0.0)
        _validate_scalar(self.mode_weight, "mode_weight", lower=0.0)
        _validate_scalar(self.balance_weight, "balance_weight", lower=0.0)
        _validate_scalar(
            self.responsibility_alive_weight,
            "responsibility_alive_weight",
            lower=0.0,
        )
        _validate_scalar(
            self.gradient_clip_norm,
            "gradient_clip_norm",
            lower=0.0,
            strict=True,
        )

        if not isinstance(self.supervise_router_mode, bool):
            raise TypeError("supervise_router_mode must be a bool")

        normalized_mapping = _normalize_mode_mapping(self.mode_to_expert)
        normalized_usage = _normalize_target_usage(self.target_usage)
        minimum_responsibility_usage = _normalize_optional_usage_floor(
            self.minimum_responsibility_usage,
        )

        if self.mode_weight > 0.0 and not normalized_mapping:
            raise ValueError(
                "mode_to_expert must be provided when mode_weight is positive"
            )

        if self.supervise_router_mode and self.mode_weight <= 0.0:
            raise ValueError("supervise_router_mode requires a positive mode_weight")

        if self.balance_weight > 0.0 and normalized_usage is None:
            raise ValueError(
                "target_usage must be explicit when balance_weight is positive"
            )

        if (
            self.responsibility_alive_weight > 0.0
            and minimum_responsibility_usage is None
        ):
            raise ValueError(
                "minimum_responsibility_usage must be provided when "
                "responsibility_alive_weight is positive"
            )

        object.__setattr__(self, "mode_to_expert", normalized_mapping)
        object.__setattr__(self, "target_usage", normalized_usage)
        object.__setattr__(
            self,
            "minimum_responsibility_usage",
            minimum_responsibility_usage,
        )


@dataclass(frozen=True, slots=True)
class Stage1ALossOutput:

    total: Tensor
    one_step_nll: Tensor
    rollout_nll: Tensor
    expert_mode_nll: Tensor
    router_mode_ce: Tensor
    balance_loss: Tensor
    responsibility_alive_loss: Tensor
    window_count: int
    transition_count: int
    labelled_transition_count: int


@dataclass(frozen=True, slots=True)
class Stage1ATrainStepResult:

    total_loss: float
    one_step_nll: float
    rollout_nll: float
    expert_mode_nll: float
    router_mode_ce: float
    balance_loss: float
    responsibility_alive_loss: float
    gradient_norm: float
    router_gradient_norm: float
    expert_gradient_norms: tuple[float, ...]
    window_count: int
    transition_count: int
    labelled_transition_count: int


@dataclass(frozen=True, slots=True)
class Stage1AEpochResult:

    total_loss: float
    one_step_nll: float
    rollout_nll: float
    expert_mode_nll: float
    router_mode_ce: float
    balance_loss: float
    responsibility_alive_loss: float
    mean_gradient_norm: float
    max_gradient_norm: float
    mean_router_gradient_norm: float
    mean_expert_gradient_norms: tuple[float, ...]
    batch_count: int
    window_count: int
    transition_count: int
    labelled_transition_count: int


def compute_stage1a_loss(
    model: Stage1ABaseWorldModel,
    batch: Mapping[str, Any],
    config: Stage1ATrainConfig,
) -> Stage1ALossOutput:

    _validate_model_and_config(model, config)

    parameter = next(model.parameters())
    observations, actions, mode_id, has_mode_label = _batch_tensors(
        batch,
        model,
        device=parameter.device,
        dtype=parameter.dtype,
    )

    batch_size = observations.shape[0]
    horizon = actions.shape[1]

    current_states = observations[:, :-1]
    target_next_states = observations[:, 1:]

    forward = model(current_states, actions)

    one_step = mixture_nll(
        forward.prediction.z_hat,
        target_next_states,
        forward.route.log_p,
        sigma=config.sigma,
    )

    zero = one_step.loss.new_zeros(())
    rollout_nll = zero

    if config.rollout_weight > 0.0:
        initial_route = model.prior_routing(
            observations[:, 0],
            actions[:, 0],
        )

        predicted_paths = rollout_base_experts(
            model,
            observations[:, 0],
            actions,
        )

        rollout = trajectory_mixture_nll(
            predicted_paths,
            target_next_states,
            initial_route.log_p,
            sigma=config.sigma,
        )
        rollout_nll = rollout.loss

    expert_mode_nll = zero
    router_mode_ce = zero

    labelled_transition_count = int(has_mode_label.sum().item()) * horizon

    if config.mode_weight > 0.0:
        expert_target, target_mask = _mode_targets(
            mode_id,
            has_mode_label,
            horizon=horizon,
            num_experts=model.config.num_experts,
            mapping=config.mode_to_expert,
        )

        if int(target_mask.sum().item()) != labelled_transition_count:
            raise RuntimeError("mapped mode-label count changed unexpectedly")

        expert_mode_nll = (
            masked_selected_expert_nll(
                one_step.component_nll,
                expert_target,
                target_mask,
            )
            / model.config.observation_dim
        )

        if config.supervise_router_mode:
            router_mode_ce = masked_mode_cross_entropy(
                forward.route.router_logits,
                expert_target,
                target_mask,
            )

    balance = zero
    responsibility_alive = zero

    if config.balance_weight > 0.0:
        target_usage = _target_usage_tensor(
            config,
            num_experts=model.config.num_experts,
            device=forward.route.p.device,
            dtype=forward.route.p.dtype,
        )

        balance = routing_balance_loss(
            forward.route.p,
            target_usage,
        )

    if config.responsibility_alive_weight > 0.0:
        minimum_usage = config.minimum_responsibility_usage

        if minimum_usage is None:
            raise RuntimeError("minimum_responsibility_usage was not configured")

        if minimum_usage >= 1.0 / model.config.num_experts:
            raise ValueError(
                "minimum_responsibility_usage must be smaller than 1 / num_experts"
            )

        responsibility_alive = minimum_marginal_usage_loss(
            one_step.training_log_responsibility,
            minimum_usage=minimum_usage,
        )

    total = (
        one_step.loss
        + config.rollout_weight * rollout_nll
        + config.mode_weight * (expert_mode_nll + router_mode_ce)
        + config.balance_weight * balance
        + config.responsibility_alive_weight * responsibility_alive
    )

    for name, value in (
        ("total Stage 1A loss", total),
        ("one-step NLL", one_step.loss),
        ("rollout NLL", rollout_nll),
        ("expert mode NLL", expert_mode_nll),
        ("router mode cross-entropy", router_mode_ce),
        ("routing balance loss", balance),
        ("responsibility alive loss", responsibility_alive),
    ):
        _require_finite(value, name)

    return Stage1ALossOutput(
        total=total,
        one_step_nll=one_step.loss,
        rollout_nll=rollout_nll,
        expert_mode_nll=expert_mode_nll,
        router_mode_ce=router_mode_ce,
        balance_loss=balance,
        responsibility_alive_loss=responsibility_alive,
        window_count=batch_size,
        transition_count=batch_size * horizon,
        labelled_transition_count=labelled_transition_count,
    )


def train_stage1a_step(
    model: Stage1ABaseWorldModel,
    optimizer: Optimizer,
    batch: Mapping[str, Any],
    config: Stage1ATrainConfig,
) -> Stage1ATrainStepResult:

    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch Optimizer")

    _validate_model_and_config(model, config)

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    _validate_optimizer_parameters(
        optimizer,
        trainable,
    )

    model.train()

    optimizer.zero_grad(set_to_none=True)

    losses = compute_stage1a_loss(
        model,
        batch,
        config,
    )

    losses.total.backward()

    router_gradient_norm = _module_gradient_norm(model.router)
    expert_gradient_norms = tuple(
        _module_gradient_norm(expert) for expert in model.g0_experts
    )

    try:
        gradient_norm_tensor = clip_grad_norm_(
            trainable,
            max_norm=config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as error:
        optimizer.zero_grad(set_to_none=True)

        raise FloatingPointError("Stage 1A gradients contain NaN or Inf") from error

    gradient_norm = float(torch.as_tensor(gradient_norm_tensor).detach().cpu().item())

    optimizer.step()

    return Stage1ATrainStepResult(
        total_loss=_as_float(losses.total),
        one_step_nll=_as_float(losses.one_step_nll),
        rollout_nll=_as_float(losses.rollout_nll),
        expert_mode_nll=_as_float(losses.expert_mode_nll),
        router_mode_ce=_as_float(losses.router_mode_ce),
        balance_loss=_as_float(losses.balance_loss),
        responsibility_alive_loss=_as_float(
            losses.responsibility_alive_loss,
        ),
        gradient_norm=gradient_norm,
        router_gradient_norm=router_gradient_norm,
        expert_gradient_norms=expert_gradient_norms,
        window_count=losses.window_count,
        transition_count=losses.transition_count,
        labelled_transition_count=(losses.labelled_transition_count),
    )


def train_stage1a_epoch(
    model: Stage1ABaseWorldModel,
    optimizer: Optimizer,
    batches: Iterable[Mapping[str, Any]],
    config: Stage1ATrainConfig,
) -> Stage1AEpochResult:

    batch_count = 0
    window_count = 0
    transition_count = 0
    labelled_count = 0

    one_step_total = 0.0
    rollout_total = 0.0
    expert_mode_total = 0.0
    router_mode_total = 0.0
    balance_total = 0.0
    responsibility_alive_total = 0.0

    gradient_total = 0.0
    max_gradient = 0.0
    router_gradient_total = 0.0
    expert_gradient_totals = [0.0] * model.config.num_experts

    for batch in batches:
        step = train_stage1a_step(
            model,
            optimizer,
            batch,
            config,
        )

        batch_count += 1
        window_count += step.window_count
        transition_count += step.transition_count
        labelled_count += step.labelled_transition_count

        one_step_total += step.one_step_nll * step.transition_count

        rollout_total += step.rollout_nll * step.window_count

        expert_mode_total += step.expert_mode_nll * step.labelled_transition_count

        router_mode_total += step.router_mode_ce * step.labelled_transition_count

        balance_total += step.balance_loss
        responsibility_alive_total += step.responsibility_alive_loss
        gradient_total += step.gradient_norm
        router_gradient_total += step.router_gradient_norm

        if len(step.expert_gradient_norms) != len(expert_gradient_totals):
            raise RuntimeError("Expert gradient count changed within an epoch")

        for index, value in enumerate(step.expert_gradient_norms):
            expert_gradient_totals[index] += value
        max_gradient = max(
            max_gradient,
            step.gradient_norm,
        )

    if batch_count == 0:
        raise ValueError("training batches must contain at least one batch")

    if window_count == 0 or transition_count == 0:
        raise RuntimeError("training epoch did not observe valid windows")

    one_step_nll = one_step_total / transition_count

    rollout_nll = rollout_total / window_count

    expert_mode_nll = expert_mode_total / labelled_count if labelled_count > 0 else 0.0

    router_mode_ce = router_mode_total / labelled_count if labelled_count > 0 else 0.0

    balance_loss = balance_total / batch_count
    responsibility_alive_loss = responsibility_alive_total / batch_count

    total_loss = (
        one_step_nll
        + config.rollout_weight * rollout_nll
        + config.mode_weight * (expert_mode_nll + router_mode_ce)
        + config.balance_weight * balance_loss
        + config.responsibility_alive_weight * responsibility_alive_loss
    )

    return Stage1AEpochResult(
        total_loss=total_loss,
        one_step_nll=one_step_nll,
        rollout_nll=rollout_nll,
        expert_mode_nll=expert_mode_nll,
        router_mode_ce=router_mode_ce,
        balance_loss=balance_loss,
        responsibility_alive_loss=responsibility_alive_loss,
        mean_gradient_norm=(gradient_total / batch_count),
        max_gradient_norm=max_gradient,
        mean_router_gradient_norm=(router_gradient_total / batch_count),
        mean_expert_gradient_norms=tuple(
            value / batch_count for value in expert_gradient_totals
        ),
        batch_count=batch_count,
        window_count=window_count,
        transition_count=transition_count,
        labelled_transition_count=labelled_count,
    )


def _batch_tensors(
    batch: Mapping[str, Any],
    model: Stage1ABaseWorldModel,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")

    observations = _tensor(
        batch,
        "observations",
    )

    actions = _tensor(
        batch,
        "actions",
    )

    mode_id = _tensor(
        batch,
        "mode_id",
    )

    has_mode_label = _tensor(
        batch,
        "has_mode_label",
    )

    if observations.ndim != 3:
        raise ValueError("observations must have shape [B, H+1, D_z]")

    if actions.ndim != 3:
        raise ValueError("actions must have shape [B, H, D_a]")

    if observations.shape[0] <= 0:
        raise ValueError("training batch must not be empty")

    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions batch sizes must match")

    if observations.shape[1] != actions.shape[1] + 1:
        raise ValueError("observations must contain exactly H+1 states")

    if actions.shape[1] <= 0:
        raise ValueError("training horizon must be positive")

    if observations.shape[-1] != model.config.observation_dim:
        raise ValueError("observation dimension does not match model config")

    if actions.shape[-1] != model.config.action_dim:
        raise ValueError("action dimension does not match model config")

    if mode_id.shape != (observations.shape[0],):
        raise ValueError("mode_id must have shape [B]")

    if has_mode_label.shape != (observations.shape[0],):
        raise ValueError("has_mode_label must have shape [B]")

    if not observations.is_floating_point() or not actions.is_floating_point():
        raise TypeError("observations and actions must be floating-point tensors")

    if mode_id.dtype != torch.long:
        raise TypeError("mode_id must use torch.long dtype")

    if has_mode_label.dtype != torch.bool:
        raise TypeError("has_mode_label must use torch.bool dtype")

    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError("observations and actions must contain only finite values")

    if torch.any(mode_id[has_mode_label] < 0):
        raise ValueError("labelled mode_id values must be non-negative")

    if torch.any(mode_id[~has_mode_label] != -1):
        raise ValueError("unlabelled mode_id values must equal -1")

    return (
        observations.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        ),
        actions.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        ),
        mode_id.to(
            device=device,
            non_blocking=True,
        ),
        has_mode_label.to(
            device=device,
            non_blocking=True,
        ),
    )


def _mode_targets(
    physical_mode_id: Tensor,
    has_mode_label: Tensor,
    *,
    horizon: int,
    num_experts: int,
    mapping: tuple[tuple[int, int], ...],
) -> tuple[Tensor, Tensor]:

    for _, expert_index in mapping:
        if expert_index >= num_experts:
            raise ValueError("mode_to_expert references a missing Expert")

    target_window = torch.full_like(
        physical_mode_id,
        -1,
    )

    mapped_window = torch.zeros_like(
        has_mode_label,
    )

    for physical_mode, expert_index in mapping:
        selected = has_mode_label & physical_mode_id.eq(physical_mode)

        target_window[selected] = expert_index
        mapped_window |= selected

    unmapped = has_mode_label & ~mapped_window

    if torch.any(unmapped):
        missing_values = physical_mode_id[unmapped].detach().cpu().tolist()

        missing = sorted(int(value) for value in missing_values)

        raise ValueError(f"labelled physical modes have no Expert mapping: {missing}")

    target = target_window[:, None].expand(
        -1,
        horizon,
    )

    mask = mapped_window[:, None].expand(
        -1,
        horizon,
    )

    return target, mask


def _target_usage_tensor(
    config: Stage1ATrainConfig,
    *,
    num_experts: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:

    if config.target_usage is None:
        raise RuntimeError("target_usage was not configured")

    if len(config.target_usage) != num_experts:
        raise ValueError("target_usage length must equal the number of Experts")

    return torch.tensor(
        config.target_usage,
        device=device,
        dtype=dtype,
    )


def _validate_optimizer_parameters(
    optimizer: Optimizer,
    trainable: list[Tensor],
) -> None:

    if not trainable:
        raise ValueError("model has no trainable Stage 1A parameters")

    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    missing = [
        parameter
        for parameter in trainable
        if id(parameter) not in optimizer_parameters
    ]

    if missing:
        raise ValueError("optimizer does not contain every trainable model parameter")


def _validate_model_and_config(
    model: Stage1ABaseWorldModel,
    config: Stage1ATrainConfig,
) -> None:
    if not isinstance(
        model,
        Stage1ABaseWorldModel,
    ):
        raise TypeError("model must be a Stage1ABaseWorldModel")

    if not isinstance(
        config,
        Stage1ATrainConfig,
    ):
        raise TypeError("config must be a Stage1ATrainConfig")


def _normalize_mode_mapping(
    raw: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:

    try:
        pairs = tuple(tuple(pair) for pair in raw)
    except TypeError as error:
        raise TypeError("mode_to_expert must contain integer pairs") from error

    normalized: list[tuple[int, int]] = []
    seen_modes: set[int] = set()

    for index, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"mode_to_expert[{index}] must contain two integers")

        physical_mode, expert_index = pair

        if not isinstance(physical_mode, int) or isinstance(physical_mode, bool):
            raise TypeError("physical mode ids must be integers")

        if not isinstance(expert_index, int) or isinstance(expert_index, bool):
            raise TypeError("Expert indices must be integers")

        if physical_mode < 0 or expert_index < 0:
            raise ValueError("mode ids and Expert indices must be non-negative")

        if physical_mode in seen_modes:
            raise ValueError("physical mode ids in mode_to_expert must be unique")

        seen_modes.add(physical_mode)

        normalized.append(
            (
                physical_mode,
                expert_index,
            )
        )

    return tuple(normalized)


def _normalize_target_usage(
    raw: tuple[float, ...] | None,
) -> tuple[float, ...] | None:

    if raw is None:
        return None

    values = tuple(raw)

    if not values:
        raise ValueError("target_usage must not be empty")

    for value in values:
        _validate_scalar(
            value,
            "target_usage item",
            lower=0.0,
        )

    if not math.isclose(
        sum(values),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("target_usage must sum to one")

    return tuple(float(value) for value in values)


def _normalize_optional_usage_floor(
    raw: float | None,
) -> float | None:

    if raw is None:
        return None

    _validate_scalar(
        raw,
        "minimum_responsibility_usage",
        lower=0.0,
        strict=True,
    )

    numeric = float(raw)

    if numeric >= 1.0:
        raise ValueError("minimum_responsibility_usage must be smaller than one")

    return numeric


def _module_gradient_norm(
    module: torch.nn.Module,
) -> float:

    squared_norm = 0.0

    for parameter in module.parameters():
        gradient = parameter.grad

        if gradient is None:
            continue

        value = gradient.detach().to(dtype=torch.float32)
        squared_norm += float(value.square().sum().cpu().item())

    return math.sqrt(squared_norm)


def _tensor(
    batch: Mapping[str, Any],
    key: str,
) -> Tensor:
    value = batch.get(key)

    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a torch.Tensor")

    return value


def _require_finite(
    value: Tensor,
    name: str,
) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _as_float(
    value: Tensor,
) -> float:
    return float(value.detach().cpu().item())


def _validate_scalar(
    value: float,
    name: str,
    *,
    lower: float,
    strict: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")

    numeric = float(value)

    invalid_bound = numeric <= lower if strict else numeric < lower

    if not math.isfinite(numeric) or invalid_bound:
        relation = "greater than" if strict else "at least"

        raise ValueError(f"{name} must be finite and {relation} {lower}")
