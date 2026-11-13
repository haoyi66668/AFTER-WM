from __future__ import annotations

import torch
from torch import Tensor

from afterwm.lib.experts.models.stage1a import Stage1ABaseWorldModel


def rollout_base_experts(
    model: Stage1ABaseWorldModel,
    initial_state: Tensor,
    action_sequence: Tensor,
) -> Tensor:


    if not isinstance(model, Stage1ABaseWorldModel):
        raise TypeError("model must be a Stage1ABaseWorldModel")

    _validate_rollout_inputs(
        model,
        initial_state,
        action_sequence,
    )

    expert_paths: list[Tensor] = []

    for expert in model.g0_experts:
        imagined_state = initial_state
        path: list[Tensor] = []

        for action in action_sequence.unbind(dim=-2):
            joint_input = torch.cat(
                (imagined_state, action),
                dim=-1,
            )

            predicted_delta = expert(joint_input)

            imagined_state = imagined_state + predicted_delta

            path.append(imagined_state)

        expert_paths.append(
            torch.stack(
                path,
                dim=-2,
            )
        )

    return torch.stack(
        expert_paths,
        dim=-2,
    )


def _validate_rollout_inputs(
    model: Stage1ABaseWorldModel,
    initial_state: Tensor,
    action_sequence: Tensor,
) -> None:

    if not isinstance(initial_state, Tensor):
        raise TypeError("initial_state must be a torch.Tensor")

    if not isinstance(action_sequence, Tensor):
        raise TypeError("action_sequence must be a torch.Tensor")

    if initial_state.ndim < 1:
        raise ValueError("initial_state must have at least one dimension")

    if action_sequence.ndim != initial_state.ndim + 1:
        raise ValueError(
            "action_sequence must add exactly one horizon "
            "dimension before the action dimension"
        )

    if initial_state.shape[-1] != model.config.observation_dim:
        raise ValueError(
            f"initial_state last dimension must equal {model.config.observation_dim}"
        )

    if action_sequence.shape[-1] != model.config.action_dim:
        raise ValueError(
            f"action_sequence last dimension must equal {model.config.action_dim}"
        )

    if action_sequence.shape[-2] <= 0:
        raise ValueError("action_sequence horizon must be positive")

    if initial_state.shape[:-1] != action_sequence.shape[:-2]:
        raise ValueError(
            "initial_state and action_sequence leading dimensions must match"
        )

    if not initial_state.is_floating_point():
        raise TypeError("initial_state must use a floating-point dtype")

    if not action_sequence.is_floating_point():
        raise TypeError("action_sequence must use a floating-point dtype")

    if initial_state.dtype != action_sequence.dtype:
        raise TypeError("initial_state and action_sequence dtypes must match")

    if initial_state.device != action_sequence.device:
        raise ValueError("initial_state and action_sequence must share a device")
