from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _classify_contact_regime(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> str:
    contact_count = int(current.get("contact_count", 0))

    if contact_count <= 0:
        return "airborne"

    if previous is not None:
        previous_signature = previous.get("signature_sha256")
        current_signature = current.get("signature_sha256")

        if (
            isinstance(previous_signature, str)
            and isinstance(current_signature, str)
            and previous_signature != current_signature
        ):
            return "transition"

    return "stable"


def _build_parent_rollin_actions(
    *,
    action_dim: int,
    steps: int,
    seed: int,
    action_limit: float,
) -> np.ndarray:
    if steps <= 0:
        return np.empty((0, action_dim), dtype=np.float32)

    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(steps, action_dim))
    actions = np.empty_like(innovations)
    state = np.zeros(action_dim, dtype=np.float64)

    for index, innovation in enumerate(innovations):
        state = 0.82 * state + 0.38 * innovation
        actions[index] = state

    scale = np.maximum(np.max(np.abs(actions), axis=1, keepdims=True), 1.0)
    actions = actions / scale
    actions *= action_limit

    return np.asarray(np.clip(actions, -1.0, 1.0), dtype=np.float32)
