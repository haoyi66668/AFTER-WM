from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from afterwm.lib.experts.eval.specialization import (
    _finalize_merge_contribution,
    _merged_one_step_nll_totals,
)
from afterwm.lib.experts.losses import (
    mixture_nll,
    trajectory_mixture_nll,
)
from afterwm.lib.experts.models import (
    Stage1ABaseWorldModel,
    rollout_base_experts,
)


@dataclass(frozen=True, slots=True)
class Stage1AValidationConfig:

    sigma: float = 1.0
    rollout_weight: float = 1.0
    minimum_effective_experts: float = 1.25
    minimum_effective_responsibility_experts: float | None = None
    minimum_prior_usage: float = 0.0
    minimum_responsibility_usage: float = 0.0
    minimum_pairwise_output_distance: float = 1e-4
    minimum_merge_contribution: float | None = None

    def __post_init__(self) -> None:
        _validate_scalar(
            self.sigma,
            "sigma",
            lower=0.0,
            strict=True,
        )
        _validate_scalar(
            self.rollout_weight,
            "rollout_weight",
            lower=0.0,
        )
        _validate_scalar(
            self.minimum_effective_experts,
            "minimum_effective_experts",
            lower=1.0,
        )
        if self.minimum_effective_responsibility_experts is None:
            object.__setattr__(
                self,
                "minimum_effective_responsibility_experts",
                float(self.minimum_effective_experts),
            )
        else:
            _validate_scalar(
                self.minimum_effective_responsibility_experts,
                "minimum_effective_responsibility_experts",
                lower=1.0,
            )
        _validate_scalar(
            self.minimum_prior_usage,
            "minimum_prior_usage",
            lower=0.0,
        )
        _validate_scalar(
            self.minimum_responsibility_usage,
            "minimum_responsibility_usage",
            lower=0.0,
        )
        _validate_scalar(
            self.minimum_pairwise_output_distance,
            "minimum_pairwise_output_distance",
            lower=0.0,
        )
        if self.minimum_merge_contribution is not None:
            _validate_finite_scalar(
                self.minimum_merge_contribution,
                "minimum_merge_contribution",
            )


@dataclass(frozen=True, slots=True)
class Stage1ASourceMetrics:

    window_count: int
    transition_count: int
    one_step_nll: float
    rollout_nll: float
    score: float


@dataclass(frozen=True, slots=True)
class Stage1AValidationResult:

    score: float
    one_step_nll: float
    rollout_nll: float
    window_count: int
    transition_count: int
    prior_usage: tuple[float, ...]
    responsibility_usage: tuple[float, ...]
    minimum_prior_usage: float
    minimum_responsibility_usage: float
    effective_prior_experts: float
    effective_responsibility_experts: float
    effective_expert_count: float
    mean_prior_entropy: float
    routing_mutual_information: float
    mean_max_prior_probability: float
    mean_pairwise_expert_distance: float
    minimum_pairwise_expert_distance: float
    merge_contribution_by_expert: tuple[float, ...]
    expert_alive: bool
    expert_starvation: bool
    route_imbalance: bool
    strictly_balanced: bool
    route_collapse: bool
    responsibility_collapse: bool
    output_collapse: bool
    duplicate_expert_collapse: bool
    merge_contribution_collapse: bool
    expert_collapse: bool
    checkpoint_invalid: bool
    source_metrics: dict[str, Stage1ASourceMetrics]
    mode_component_nll: dict[int, tuple[float, ...]]


@dataclass(slots=True)
class _SourceAccumulator:

    window_count: int = 0
    transition_count: int = 0
    one_step_total: float = 0.0
    rollout_total: float = 0.0


def validate_stage1a(
    model: Stage1ABaseWorldModel,
    batches: Iterable[Mapping[str, Any]],
    config: Stage1AValidationConfig | None = None,
) -> Stage1AValidationResult:

    if not isinstance(
        model,
        Stage1ABaseWorldModel,
    ):
        raise TypeError("model must be a Stage1ABaseWorldModel")

    if config is None:
        config = Stage1AValidationConfig()
    elif not isinstance(
        config,
        Stage1AValidationConfig,
    ):
        raise TypeError("config must be a Stage1AValidationConfig")

    num_experts = model.config.num_experts

    if num_experts > 1 and config.minimum_effective_experts > num_experts:
        raise ValueError(
            "minimum_effective_experts must not exceed the number of Experts"
        )

    responsibility_threshold = config.minimum_effective_responsibility_experts
    assert responsibility_threshold is not None

    if num_experts > 1 and responsibility_threshold > num_experts:
        raise ValueError(
            "minimum_effective_responsibility_experts must not exceed "
            "the number of Experts"
        )

    maximum_feasible_minimum_usage = 1.0 / num_experts

    if config.minimum_prior_usage > maximum_feasible_minimum_usage:
        raise ValueError("minimum_prior_usage must not exceed 1 / num_experts")

    if config.minimum_responsibility_usage > maximum_feasible_minimum_usage:
        raise ValueError("minimum_responsibility_usage must not exceed 1 / num_experts")

    parameter = next(model.parameters())

    device = parameter.device
    dtype = parameter.dtype
    was_training = model.training

    one_step_total = 0.0
    rollout_total = 0.0
    transition_count = 0
    window_count = 0

    prior_usage_sum = torch.zeros(
        num_experts,
        dtype=torch.float64,
    )
    prior_entropy_total = 0.0
    maximum_prior_probability_total = 0.0
    usage_count = 0

    log_responsibility_sum = torch.full(
        (num_experts,),
        -torch.inf,
        dtype=torch.float64,
    )

    pair_distance_sums = torch.zeros(
        (num_experts, num_experts),
        dtype=torch.float64,
    )
    pair_distance_item_count = 0

    merge_nll_totals = torch.zeros(
        (num_experts, num_experts),
        dtype=torch.float64,
    )

    source_stats: dict[
        str,
        _SourceAccumulator,
    ] = {}

    mode_totals: dict[
        int,
        Tensor,
    ] = {}

    mode_counts: dict[
        int,
        int,
    ] = {}

    model.eval()

    try:
        with torch.no_grad():
            for batch in batches:
                observations, actions = _sequence_tensors(
                    batch,
                    model,
                    device=device,
                    dtype=dtype,
                )

                batch_size = observations.shape[0]
                horizon = actions.shape[1]

                source_ids = _source_ids(
                    batch,
                    batch_size,
                )

                current_states = observations[:, :-1]

                target_next_states = observations[:, 1:]

                forward = model(
                    current_states,
                    actions,
                )

                one_step = mixture_nll(
                    forward.prediction.z_hat,
                    target_next_states,
                    forward.route.log_p,
                    sigma=config.sigma,
                    reduction="none",
                )

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
                    reduction="none",
                )

                _require_finite(
                    one_step.per_item_nll,
                    "one-step validation NLL",
                )

                _require_finite(
                    one_step.component_nll,
                    "one-step component validation NLL",
                )

                _require_finite(
                    rollout.per_item_nll,
                    "rollout validation NLL",
                )

                one_step_total += one_step.per_item_nll.double().sum().item()

                transition_count += one_step.per_item_nll.numel()

                rollout_total += rollout.per_item_nll.double().sum().item()

                window_count += batch_size

                prior = forward.route.p.detach().double()
                prior = prior / prior.sum(
                    dim=-1,
                    keepdim=True,
                )
                prior_usage_sum += prior.sum(dim=(0, 1)).cpu()
                prior_entropy_total += (
                    (
                        -(
                            prior * prior.clamp_min(torch.finfo(prior.dtype).tiny).log()
                        ).sum(dim=-1)
                    )
                    .sum()
                    .item()
                )
                maximum_prior_probability_total += prior.max(dim=-1).values.sum().item()

                batch_log_responsibility_sum = torch.logsumexp(
                    _training_log_responsibility(one_step)
                    .detach()
                    .double()
                    .reshape(-1, num_experts),
                    dim=0,
                ).cpu()
                log_responsibility_sum = torch.logaddexp(
                    log_responsibility_sum,
                    batch_log_responsibility_sum,
                )

                usage_count += batch_size * horizon

                pair_sums, pair_item_count = _pairwise_output_statistics(
                    forward.prediction.d_base
                )

                pair_distance_sums += pair_sums
                pair_distance_item_count += pair_item_count

                batch_merge_totals = _merged_one_step_nll_totals(
                    forward.prediction.z_hat,
                    target_next_states,
                    forward.route.log_p,
                    sigma=config.sigma,
                ).cpu()
                off_diagonal = ~torch.eye(
                    num_experts,
                    dtype=torch.bool,
                )
                merge_nll_totals[off_diagonal] += batch_merge_totals[off_diagonal]

                _accumulate_sources(
                    source_stats,
                    source_ids,
                    one_step.per_item_nll,
                    rollout.per_item_nll,
                )

                mode_id, has_mode_label = _mode_labels(
                    batch,
                    batch_size,
                )

                _accumulate_modes(
                    mode_totals,
                    mode_counts,
                    one_step.component_nll,
                    mode_id,
                    has_mode_label,
                    observation_dim=(model.config.observation_dim),
                )

    finally:
        model.train(was_training)

    if window_count == 0:
        raise ValueError("validation batches must contain at least one window")

    if transition_count == 0 or usage_count == 0:
        raise RuntimeError("validation did not observe any valid transition")

    one_step_nll = one_step_total / transition_count

    rollout_nll = rollout_total / window_count

    score = one_step_nll + config.rollout_weight * rollout_nll

    prior_usage = prior_usage_sum / usage_count

    prior_usage = prior_usage / prior_usage.sum()

    responsibility_usage = torch.exp(log_responsibility_sum - math.log(usage_count))
    responsibility_usage = responsibility_usage / responsibility_usage.sum()

    effective_prior_experts = _effective_expert_count(prior_usage)
    effective_responsibility_experts = _effective_expert_count(responsibility_usage)

    marginal_prior_entropy = math.log(effective_prior_experts)
    mean_prior_entropy = prior_entropy_total / usage_count
    routing_mutual_information = min(
        math.log(num_experts),
        max(0.0, marginal_prior_entropy - mean_prior_entropy),
    )
    mean_max_prior_probability = maximum_prior_probability_total / usage_count

    if not all(
        math.isfinite(value)
        for value in (
            mean_prior_entropy,
            routing_mutual_information,
            mean_max_prior_probability,
        )
    ):
        raise FloatingPointError("Router information diagnostics contain NaN or Inf")

    information_tolerance = 1e-6

    if not (
        -information_tolerance
        <= mean_prior_entropy
        <= math.log(num_experts) + information_tolerance
    ):
        raise RuntimeError("mean_prior_entropy is outside its valid range")

    if not (
        -information_tolerance
        <= routing_mutual_information
        <= math.log(num_experts) + information_tolerance
    ):
        raise RuntimeError("routing_mutual_information is outside its valid range")

    if not (
        1.0 / num_experts - information_tolerance
        <= mean_max_prior_probability
        <= 1.0 + information_tolerance
    ):
        raise RuntimeError("mean_max_prior_probability is outside its valid range")

    (
        mean_pairwise_distance,
        minimum_pairwise_distance,
    ) = _finalize_pairwise_output_statistics(
        pair_distance_sums,
        item_count=pair_distance_item_count,
    )

    merge_contribution = _finalize_merge_contribution(
        merge_nll_totals,
        baseline_total=one_step_total,
        item_count=transition_count,
    )

    prior_starvation = num_experts > 1 and (
        float(prior_usage.min()) < config.minimum_prior_usage
    )
    responsibility_starvation = num_experts > 1 and (
        float(responsibility_usage.min()) < config.minimum_responsibility_usage
    )
    prior_imbalance = num_experts > 1 and (
        effective_prior_experts < config.minimum_effective_experts
    )
    responsibility_imbalance = num_experts > 1 and (
        effective_responsibility_experts < responsibility_threshold
    )

    route_collapse = prior_starvation or prior_imbalance
    responsibility_collapse = responsibility_starvation or responsibility_imbalance

    output_collapse = (
        num_experts > 1
        and mean_pairwise_distance < config.minimum_pairwise_output_distance
    )

    duplicate_expert_collapse = (
        num_experts > 1
        and minimum_pairwise_distance < config.minimum_pairwise_output_distance
    )

    merge_contribution_collapse = (
        num_experts > 1
        and config.minimum_merge_contribution is not None
        and min(merge_contribution) < config.minimum_merge_contribution
    )

    expert_starvation = (
        prior_starvation
        or responsibility_starvation
        or output_collapse
        or duplicate_expert_collapse
        or merge_contribution_collapse
    )
    route_imbalance = prior_imbalance or responsibility_imbalance
    expert_alive = not expert_starvation
    strictly_balanced = expert_alive and not route_imbalance

    expert_collapse = (
        route_collapse
        or responsibility_collapse
        or output_collapse
        or duplicate_expert_collapse
        or merge_contribution_collapse
    )

    return Stage1AValidationResult(
        score=score,
        one_step_nll=one_step_nll,
        rollout_nll=rollout_nll,
        window_count=window_count,
        transition_count=transition_count,
        prior_usage=tuple(float(value) for value in prior_usage),
        responsibility_usage=tuple(float(value) for value in responsibility_usage),
        minimum_prior_usage=float(prior_usage.min()),
        minimum_responsibility_usage=float(responsibility_usage.min()),
        effective_prior_experts=effective_prior_experts,
        effective_responsibility_experts=(effective_responsibility_experts),
        effective_expert_count=effective_prior_experts,
        mean_prior_entropy=mean_prior_entropy,
        routing_mutual_information=(routing_mutual_information),
        mean_max_prior_probability=(mean_max_prior_probability),
        mean_pairwise_expert_distance=(mean_pairwise_distance),
        minimum_pairwise_expert_distance=(minimum_pairwise_distance),
        merge_contribution_by_expert=merge_contribution,
        expert_alive=expert_alive,
        expert_starvation=expert_starvation,
        route_imbalance=route_imbalance,
        strictly_balanced=strictly_balanced,
        route_collapse=route_collapse,
        responsibility_collapse=responsibility_collapse,
        output_collapse=output_collapse,
        duplicate_expert_collapse=duplicate_expert_collapse,
        merge_contribution_collapse=(merge_contribution_collapse),
        expert_collapse=expert_collapse,
        checkpoint_invalid=expert_collapse,
        source_metrics=_finalize_sources(
            source_stats,
            config.rollout_weight,
        ),
        mode_component_nll=_finalize_modes(
            mode_totals,
            mode_counts,
        ),
    )


def _sequence_tensors(
    batch: Mapping[str, Any],
    model: Stage1ABaseWorldModel,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:

    if not isinstance(batch, Mapping):
        raise TypeError("each validation batch must be a mapping")

    observations = _tensor(
        batch,
        "observations",
    )

    actions = _tensor(
        batch,
        "actions",
    )

    if observations.ndim != 3:
        raise ValueError("observations must have shape [B, H+1, D_z]")

    if actions.ndim != 3:
        raise ValueError("actions must have shape [B, H, D_a]")

    if observations.shape[0] <= 0:
        raise ValueError("validation batch must not be empty")

    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions batch sizes must match")

    if observations.shape[1] != actions.shape[1] + 1:
        raise ValueError("observations must contain exactly H+1 states")

    if actions.shape[1] <= 0:
        raise ValueError("validation horizon must be positive")

    if observations.shape[-1] != model.config.observation_dim:
        raise ValueError("observation dimension does not match model config")

    if actions.shape[-1] != model.config.action_dim:
        raise ValueError("action dimension does not match model config")

    if not observations.is_floating_point():
        raise TypeError("observations must use a floating-point dtype")

    if not actions.is_floating_point():
        raise TypeError("actions must use a floating-point dtype")

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
    )


def _source_ids(
    batch: Mapping[str, Any],
    batch_size: int,
) -> tuple[str, ...]:

    raw = batch.get("source_id")

    if isinstance(raw, str):
        values = (raw,)

    elif isinstance(raw, Sequence):
        values = tuple(raw)

    else:
        raise TypeError("source_id must be a string or a sequence of strings")

    if len(values) != batch_size:
        raise ValueError("source_id count must equal batch size")

    if not all(isinstance(value, str) for value in values):
        raise TypeError("every source_id must be a string")

    if not all(values):
        raise ValueError("every source_id must be non-empty")

    return values


def _mode_labels(
    batch: Mapping[str, Any],
    batch_size: int,
) -> tuple[Tensor, Tensor]:

    mode_id = _tensor(
        batch,
        "mode_id",
    )

    has_label = _tensor(
        batch,
        "has_mode_label",
    )

    if mode_id.shape != (batch_size,):
        raise ValueError("mode_id must have shape [B]")

    if has_label.shape != (batch_size,):
        raise ValueError("has_mode_label must have shape [B]")

    if mode_id.dtype != torch.long:
        raise TypeError("mode_id must use torch.long dtype")

    if has_label.dtype != torch.bool:
        raise TypeError("has_mode_label must use torch.bool dtype")

    if torch.any(mode_id[has_label] < 0):
        raise ValueError("labelled mode_id values must be non-negative")

    if torch.any(mode_id[~has_label] != -1):
        raise ValueError("unlabelled mode_id values must equal -1")

    return (
        mode_id.detach().cpu(),
        has_label.detach().cpu(),
    )


def _pairwise_output_statistics(
    expert_deltas: Tensor,
) -> tuple[Tensor, int]:

    num_experts = expert_deltas.shape[-2]

    sums = torch.zeros(
        (num_experts, num_experts),
        dtype=torch.float64,
    )
    item_count = expert_deltas[..., 0, :].numel() // expert_deltas.shape[-1]

    for first in range(num_experts):
        for second in range(
            first + 1,
            num_experts,
        ):
            difference = expert_deltas[..., first, :] - expert_deltas[..., second, :]

            distance = torch.sqrt(difference.double().square().mean(dim=-1))

            sums[first, second] = distance.sum().item()

    return sums, item_count


def _finalize_pairwise_output_statistics(
    distance_sums: Tensor,
    *,
    item_count: int,
) -> tuple[float, float]:

    num_experts = distance_sums.shape[0]

    if distance_sums.shape != (num_experts, num_experts):
        raise ValueError("distance_sums must have shape [K, K]")

    if num_experts < 2:
        return 0.0, 0.0

    if item_count <= 0:
        raise ValueError("item_count must be positive")

    pair_mask = torch.triu(
        torch.ones_like(distance_sums, dtype=torch.bool),
        diagonal=1,
    )
    pair_means = distance_sums[pair_mask] / item_count

    if not torch.isfinite(pair_means).all():
        raise FloatingPointError("pairwise Expert output distance contains NaN or Inf")

    return (
        float(pair_means.mean()),
        float(pair_means.min()),
    )


def _training_log_responsibility(one_step: Any) -> Tensor:

    value = getattr(
        one_step,
        "training_log_responsibility",
        None,
    )

    if isinstance(value, Tensor):
        return value

    responsibility = one_step.training_responsibility
    tiny = torch.finfo(responsibility.dtype).tiny
    return responsibility.clamp_min(tiny).log()


def _effective_expert_count(usage: Tensor) -> float:

    if not torch.isfinite(usage).all() or torch.any(usage < 0):
        raise FloatingPointError("Expert usage contains NaN, Inf, or negatives")

    entropy = -torch.sum(
        usage * torch.log(usage.clamp_min(torch.finfo(usage.dtype).tiny))
    ).item()
    return math.exp(entropy)


def _accumulate_sources(
    stats: dict[str, _SourceAccumulator],
    source_ids: tuple[str, ...],
    one_step: Tensor,
    rollout: Tensor,
) -> None:

    one_step_cpu = one_step.detach().double().cpu()

    rollout_cpu = rollout.detach().double().cpu()

    for index, source_id in enumerate(source_ids):
        current = stats.setdefault(
            source_id,
            _SourceAccumulator(),
        )

        current.window_count += 1

        current.transition_count += one_step_cpu.shape[1]

        current.one_step_total += one_step_cpu[index].sum().item()

        current.rollout_total += rollout_cpu[index].item()


def _accumulate_modes(
    totals: dict[int, Tensor],
    counts: dict[int, int],
    component_nll: Tensor,
    mode_id: Tensor,
    has_label: Tensor,
    *,
    observation_dim: int,
) -> None:

    per_window = (component_nll.double().mean(dim=1) / observation_dim).cpu()

    for index in range(mode_id.shape[0]):
        if not bool(has_label[index]):
            continue

        current_mode = int(mode_id[index].item())

        if current_mode not in totals:
            totals[current_mode] = torch.zeros_like(per_window[index])

            counts[current_mode] = 0

        totals[current_mode] += per_window[index]

        counts[current_mode] += 1


def _finalize_sources(
    stats: dict[str, _SourceAccumulator],
    rollout_weight: float,
) -> dict[str, Stage1ASourceMetrics]:

    result: dict[
        str,
        Stage1ASourceMetrics,
    ] = {}

    for source_id, current in sorted(stats.items()):
        one_step_nll = current.one_step_total / current.transition_count

        rollout_nll = current.rollout_total / current.window_count

        result[source_id] = Stage1ASourceMetrics(
            window_count=(current.window_count),
            transition_count=(current.transition_count),
            one_step_nll=one_step_nll,
            rollout_nll=rollout_nll,
            score=(one_step_nll + rollout_weight * rollout_nll),
        )

    return result


def _finalize_modes(
    totals: dict[int, Tensor],
    counts: dict[int, int],
) -> dict[int, tuple[float, ...]]:

    return {
        mode: tuple(float(value) for value in (totals[mode] / counts[mode]))
        for mode in sorted(totals)
    }


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


def _validate_scalar(
    value: float,
    name: str,
    *,
    lower: float,
    strict: bool = False,
) -> None:

    if isinstance(value, bool) or not isinstance(
        value,
        int | float,
    ):
        raise TypeError(f"{name} must be numeric")

    numeric = float(value)

    invalid_bound = numeric <= lower if strict else numeric < lower

    if not math.isfinite(numeric) or invalid_bound:
        relation = "greater than" if strict else "at least"

        raise ValueError(f"{name} must be finite and {relation} {lower}")


def _validate_finite_scalar(
    value: float,
    name: str,
) -> None:

    if isinstance(value, bool) or not isinstance(
        value,
        int | float,
    ):
        raise TypeError(f"{name} must be numeric")

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
