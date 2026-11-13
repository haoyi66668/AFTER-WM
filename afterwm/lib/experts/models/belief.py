from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Stage1B0Belief:


    q_previous: Tensor
    F: Tensor
    S: Tensor
    H: Tensor

    def detach(self) -> Stage1B0Belief:

        return Stage1B0Belief(
            q_previous=self.q_previous.detach(),
            F=self.F.detach(),
            S=self.S.detach(),
            H=self.H.detach(),
        )


def initialize_stage1b0_belief(
    reference_state: Tensor,
    *,
    num_experts: int,
    fast_dim: int,
    slow_dim: int,
) -> Stage1B0Belief:

    if not isinstance(reference_state, Tensor):
        raise TypeError("reference_state must be a torch.Tensor")

    if reference_state.ndim < 1:
        raise ValueError("reference_state must have at least one dimension")

    if not reference_state.is_floating_point():
        raise TypeError("reference_state must use a floating-point dtype")

    _require_positive_integer(num_experts, name="num_experts")
    _require_positive_integer(fast_dim, name="fast_dim")
    _require_positive_integer(slow_dim, name="slow_dim")

    leading_shape = reference_state.shape[:-1]
    common = {
        "dtype": torch.float32,
        "device": reference_state.device,
    }

    return Stage1B0Belief(
        q_previous=torch.full(
            (*leading_shape, num_experts),
            fill_value=(1.0 / num_experts),
            **common,
        ),
        F=torch.zeros(
            (*leading_shape, num_experts, fast_dim),
            **common,
        ),
        S=torch.zeros(
            (*leading_shape, num_experts, slow_dim),
            **common,
        ),
        H=torch.zeros(
            (*leading_shape, num_experts, fast_dim),
            **common,
        ),
    )


def validate_stage1b0_belief(
    belief: Stage1B0Belief,
    *,
    leading_shape: torch.Size | tuple[int, ...],
    num_experts: int,
    fast_dim: int,
    slow_dim: int,
    device: torch.device,
    require_zero_placeholders: bool,
    probability_tolerance: float = 1e-5,
) -> None:

    if not isinstance(belief, Stage1B0Belief):
        raise TypeError("belief must be a Stage1B0Belief")

    normalized_leading_shape = tuple(leading_shape)
    expected_shapes = {
        "q_previous": (*normalized_leading_shape, num_experts),
        "F": (*normalized_leading_shape, num_experts, fast_dim),
        "S": (*normalized_leading_shape, num_experts, slow_dim),
        "H": (*normalized_leading_shape, num_experts, fast_dim),
    }

    for name, expected_shape in expected_shapes.items():
        value = getattr(belief, name)

        if not isinstance(value, Tensor):
            raise TypeError(f"belief.{name} must be a torch.Tensor")

        if value.shape != expected_shape:
            raise ValueError(
                f"belief.{name} shape must equal {expected_shape}, got {value.shape}"
            )

        if not value.is_floating_point():
            raise TypeError(f"belief.{name} must use a floating-point dtype")

        if value.device != device:
            raise ValueError(f"belief.{name} must be on device {device}")

        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"belief.{name} must contain only finite values")

    q_previous = belief.q_previous.to(dtype=torch.float32)

    if bool((q_previous <= 0.0).any()):
        raise ValueError("belief.q_previous must be strictly positive")

    probability_sums = q_previous.sum(dim=-1)

    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=probability_tolerance,
        rtol=probability_tolerance,
    ):
        raise ValueError("belief.q_previous must sum to one along the expert axis")

    if require_zero_placeholders:
        for name in ("F", "S", "H"):
            value = getattr(belief, name)

            if bool(torch.count_nonzero(value)):
                raise ValueError(
                    f"belief.{name} must remain exactly zero in Stage1B-0 q-only"
                )


def _require_positive_integer(
    value: int,
    *,
    name: str,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")

    if value <= 0:
        raise ValueError(f"{name} must be positive")
