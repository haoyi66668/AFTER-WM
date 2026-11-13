from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from afterwm.lib.experts.losses import mixture_nll
from afterwm.lib.experts.models import Stage1ABaseWorldModel


@dataclass(frozen=True, slots=True)
class Stage1ASpecializationResult:

    mixture_nll: float
    prior_usage: tuple[float, ...]
    responsibility_usage: tuple[float, ...]
    effective_prior_experts: float
    effective_responsibility_experts: float
    mean_prior_entropy: float
    routing_mutual_information: float
    mean_max_prior_probability: float
    mean_pairwise_expert_distance: float
    minimum_pairwise_expert_distance: float
    merge_contribution_by_expert: tuple[float, ...]
    mode_component_nll: dict[int, tuple[float, ...]]
    best_expert_by_mode: dict[int, int]
    best_expert_margin_by_mode: dict[int, float]
    distinct_best_expert_count: int


def evaluate_stage1a_specialization(
    model: Stage1ABaseWorldModel,
    batch: Mapping[str, Any],
    diagnostic_mode_id: Tensor,
    *,
    sigma: float = 1.0,
) -> Stage1ASpecializationResult:


    if not isinstance(model, Stage1ABaseWorldModel):
        raise TypeError("model must be a Stage1ABaseWorldModel")

    if not isinstance(batch, Mapping):
        raise TypeError("batch must be a mapping")

    observations = _required_tensor(
        batch,
        "observations",
    )
    actions = _required_tensor(
        batch,
        "actions",
    )

    _validate_sequences(
        observations,
        actions,
        model,
    )
    _validate_diagnostic_mode_id(
        diagnostic_mode_id,
        batch_size=observations.shape[0],
    )

    parameter = next(model.parameters())
    observations = observations.to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    actions = actions.to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            current_states = observations[:, :-1]
            target_next_states = observations[:, 1:]

            forward = model(
                current_states,
                actions,
            )

            mixture = mixture_nll(
                forward.prediction.z_hat,
                target_next_states,
                forward.route.log_p,
                sigma=sigma,
            )

            merge_totals = _merged_one_step_nll_totals(
                forward.prediction.z_hat,
                target_next_states,
                forward.route.log_p,
                sigma=sigma,
            )
            merge_contribution = _finalize_merge_contribution(
                merge_totals,
                baseline_total=_scalar_float(mixture.per_item_nll.double().sum()),
                item_count=mixture.per_item_nll.numel(),
            )

            prior = forward.route.p.to(dtype=torch.float64)
            prior = prior / prior.sum(
                dim=-1,
                keepdim=True,
            )
            prior_usage = prior.mean(dim=tuple(range(prior.ndim - 1)))

            log_responsibility = _training_log_responsibility(mixture).double()
            flat_log_responsibility = log_responsibility.reshape(
                -1,
                model.config.num_experts,
            )
            log_responsibility_usage = torch.logsumexp(
                flat_log_responsibility,
                dim=0,
            ) - math.log(flat_log_responsibility.shape[0])
            responsibility_usage = torch.exp(log_responsibility_usage)
            responsibility_usage = responsibility_usage / responsibility_usage.sum()

            mean_prior_entropy = _mean_entropy(prior)
            marginal_prior_entropy = _entropy(prior_usage)
            routing_mutual_information = min(
                math.log(model.config.num_experts),
                max(
                    0.0,
                    marginal_prior_entropy - mean_prior_entropy,
                ),
            )
            mean_max_prior_probability = _scalar_float(prior.max(dim=-1).values.mean())

            component_nll = mixture.component_nll / model.config.observation_dim

            diagnostic_mode_id = diagnostic_mode_id.to(
                device=parameter.device,
            )

            mode_component_nll = _mode_component_nll(
                component_nll,
                diagnostic_mode_id,
            )

            best_expert_by_mode = {
                mode: min(
                    range(len(values)),
                    key=values.__getitem__,
                )
                for mode, values in mode_component_nll.items()
            }

            best_expert_margin_by_mode = {
                mode: _best_expert_margin(values)
                for mode, values in mode_component_nll.items()
            }

            (
                mean_pairwise_distance,
                minimum_pairwise_distance,
            ) = _pairwise_distance_statistics(forward.prediction.z_hat)
    finally:
        model.train(was_training)

    result = Stage1ASpecializationResult(
        mixture_nll=_scalar_float(mixture.loss),
        prior_usage=_float_tuple(prior_usage),
        responsibility_usage=_float_tuple(responsibility_usage),
        effective_prior_experts=math.exp(marginal_prior_entropy),
        effective_responsibility_experts=math.exp(_entropy(responsibility_usage)),
        mean_prior_entropy=mean_prior_entropy,
        routing_mutual_information=(routing_mutual_information),
        mean_max_prior_probability=(mean_max_prior_probability),
        mean_pairwise_expert_distance=(mean_pairwise_distance),
        minimum_pairwise_expert_distance=(minimum_pairwise_distance),
        merge_contribution_by_expert=(merge_contribution),
        mode_component_nll=mode_component_nll,
        best_expert_by_mode=best_expert_by_mode,
        best_expert_margin_by_mode=(best_expert_margin_by_mode),
        distinct_best_expert_count=len(set(best_expert_by_mode.values())),
    )

    _require_finite_result(result)
    return result


def _required_tensor(
    batch: Mapping[str, Any],
    key: str,
) -> Tensor:
    value = batch.get(key)

    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a torch.Tensor")

    return value


def _validate_sequences(
    observations: Tensor,
    actions: Tensor,
    model: Stage1ABaseWorldModel,
) -> None:
    if observations.ndim != 3:
        raise ValueError("observations must have shape [B, H+1, D_z]")

    if actions.ndim != 3:
        raise ValueError("actions must have shape [B, H, D_a]")

    if observations.shape[0] <= 0:
        raise ValueError("diagnostic batch must not be empty")

    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions batch sizes must match")

    if observations.shape[1] != actions.shape[1] + 1:
        raise ValueError("observations must contain exactly H+1 states")

    if actions.shape[1] <= 0:
        raise ValueError("diagnostic horizon must be positive")

    if observations.shape[-1] != model.config.observation_dim:
        raise ValueError("observation dimension does not match model config")

    if actions.shape[-1] != model.config.action_dim:
        raise ValueError("action dimension does not match model config")

    if not observations.is_floating_point() or not actions.is_floating_point():
        raise TypeError("observations and actions must be floating-point tensors")

    if not torch.isfinite(observations).all() or not torch.isfinite(actions).all():
        raise ValueError("observations and actions must contain only finite values")


def _validate_diagnostic_mode_id(
    mode_id: Tensor,
    *,
    batch_size: int,
) -> None:
    if not isinstance(mode_id, Tensor):
        raise TypeError("diagnostic_mode_id must be a torch.Tensor")

    if mode_id.shape != (batch_size,):
        raise ValueError("diagnostic_mode_id must have shape [B]")

    if mode_id.dtype != torch.long:
        raise TypeError("diagnostic_mode_id must use torch.long dtype")

    if torch.any(mode_id < 0):
        raise ValueError("diagnostic_mode_id values must be non-negative")


def _mode_component_nll(
    component_nll: Tensor,
    mode_id: Tensor,
) -> dict[int, tuple[float, ...]]:
    horizon = component_nll.shape[1]
    num_experts = component_nll.shape[-1]

    flat_component = component_nll.reshape(
        -1,
        num_experts,
    )
    flat_mode = (
        mode_id[:, None]
        .expand(
            -1,
            horizon,
        )
        .reshape(-1)
    )

    output: dict[int, tuple[float, ...]] = {}

    for mode in sorted(int(value) for value in torch.unique(flat_mode).tolist()):
        selected = flat_component[flat_mode.eq(mode)]

        if selected.numel() == 0:
            raise RuntimeError("diagnostic mode unexpectedly has no transitions")

        output[mode] = _float_tuple(selected.mean(dim=0))

    return output


def _pairwise_distance_statistics(
    predictions: Tensor,
) -> tuple[float, float]:

    num_experts = predictions.shape[-2]

    if num_experts < 2:
        return 0.0, 0.0

    distances: list[Tensor] = []

    for first in range(num_experts):
        for second in range(first + 1, num_experts):
            difference = predictions[..., first, :] - predictions[..., second, :]
            per_item_rms = difference.double().square().mean(dim=-1).sqrt()
            distances.append(per_item_rms.mean())

    stacked = torch.stack(distances)
    return (
        _scalar_float(stacked.mean()),
        _scalar_float(stacked.min()),
    )


def _merged_one_step_nll_totals(
    predicted_next_states: Tensor,
    target_next_states: Tensor,
    log_mixing_probabilities: Tensor,
    *,
    sigma: float,
) -> Tensor:


    num_experts = predicted_next_states.shape[-2]
    totals = torch.full(
        (num_experts, num_experts),
        torch.nan,
        dtype=torch.float64,
        device=predicted_next_states.device,
    )

    if num_experts < 2:
        return totals

    normalized_log_probabilities = log_mixing_probabilities - torch.logsumexp(
        log_mixing_probabilities,
        dim=-1,
        keepdim=True,
    )

    for removed in range(num_experts):
        for recipient in range(num_experts):
            if removed == recipient:
                continue

            merged_log_probabilities = normalized_log_probabilities.clone()
            merged_log_probabilities[..., recipient] = torch.logaddexp(
                normalized_log_probabilities[..., recipient],
                normalized_log_probabilities[..., removed],
            )
            merged_log_probabilities[..., removed] = -torch.inf

            merged = mixture_nll(
                predicted_next_states,
                target_next_states,
                merged_log_probabilities,
                sigma=sigma,
                reduction="none",
            )
            totals[removed, recipient] = merged.per_item_nll.double().sum()

    return totals


def _finalize_merge_contribution(
    merged_nll_totals: Tensor,
    *,
    baseline_total: float,
    item_count: int,
) -> tuple[float, ...]:

    if item_count <= 0:
        raise ValueError("item_count must be positive")

    num_experts = merged_nll_totals.shape[0]

    if merged_nll_totals.shape != (num_experts, num_experts):
        raise ValueError("merged_nll_totals must have shape [K, K]")

    if num_experts == 1:
        return (0.0,)

    baseline_mean = baseline_total / item_count
    contributions: list[float] = []

    for removed in range(num_experts):
        candidates = torch.cat(
            (
                merged_nll_totals[removed, :removed],
                merged_nll_totals[removed, removed + 1 :],
            )
        )

        if not torch.isfinite(candidates).all():
            raise FloatingPointError(
                "merged one-step validation NLL contains NaN or Inf"
            )

        merged_mean = _scalar_float(candidates.min()) / item_count
        contributions.append(merged_mean - baseline_mean)

    return tuple(contributions)


def _best_expert_margin(
    values: tuple[float, ...],
) -> float:
    if len(values) < 2:
        return 0.0

    ordered = sorted(values)
    return ordered[1] - ordered[0]


def _training_log_responsibility(mixture: Any) -> Tensor:

    value = getattr(
        mixture,
        "training_log_responsibility",
        None,
    )

    if isinstance(value, Tensor):
        return value

    responsibility = mixture.training_responsibility
    tiny = torch.finfo(responsibility.dtype).tiny
    return responsibility.clamp_min(tiny).log()


def _mean_entropy(probabilities: Tensor) -> float:
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)

    return _scalar_float(entropy.mean())


def _entropy(probabilities: Tensor) -> float:
    return _scalar_float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())


def _float_tuple(values: Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in values.detach().cpu().tolist())


def _scalar_float(value: Tensor) -> float:
    return float(value.detach().cpu().item())


def _require_finite_result(
    result: Stage1ASpecializationResult,
) -> None:
    scalar_values = (
        result.mixture_nll,
        result.effective_prior_experts,
        result.effective_responsibility_experts,
        result.mean_prior_entropy,
        result.routing_mutual_information,
        result.mean_max_prior_probability,
        result.mean_pairwise_expert_distance,
        result.minimum_pairwise_expert_distance,
    )

    grouped_values = (
        *result.prior_usage,
        *result.responsibility_usage,
        *result.merge_contribution_by_expert,
        *(value for values in result.mode_component_nll.values() for value in values),
        *result.best_expert_margin_by_mode.values(),
    )

    if not all(math.isfinite(value) for value in (*scalar_values, *grouped_values)):
        raise FloatingPointError("Stage 1A specialization metrics contain NaN or Inf")

    num_experts = len(result.prior_usage)
    tolerance = 1e-6

    if not (
        -tolerance <= result.mean_prior_entropy <= math.log(num_experts) + tolerance
    ):
        raise RuntimeError("mean_prior_entropy is outside its valid range")

    if not (
        -tolerance
        <= result.routing_mutual_information
        <= math.log(num_experts) + tolerance
    ):
        raise RuntimeError("routing_mutual_information is outside its valid range")

    if not (
        1.0 / num_experts - tolerance
        <= result.mean_max_prior_probability
        <= 1.0 + tolerance
    ):
        raise RuntimeError("mean_max_prior_probability is outside its valid range")
