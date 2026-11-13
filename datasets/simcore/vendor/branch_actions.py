from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ACTION_PLAN_KINDS = ("smooth_ar1", "piecewise", "periodic", "iid")


@dataclass(frozen=True, slots=True)
class BranchActionPlan:

    kind: str
    seed: int
    actions: np.ndarray
    sha256: str = ""

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        _validate_seed(self.seed)
        actions = _validated_actions(self.actions)
        expected_hash = compute_action_plan_sha256(actions)
        if self.sha256 and self.sha256 != expected_hash:
            raise ValueError("action plan sha256 does not match the action array")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "sha256", expected_hash)

    @property
    def plan_id(self) -> str:
        return f"{self.kind}-{self.sha256[:16]}"

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dimension(self) -> int:
        return int(self.actions.shape[1])


def generate_action_plan(
    kind: str,
    horizon: int,
    action_dimension: int,
    seed: int,
    action_limit: float = 1.0,
) -> BranchActionPlan:


    _validate_kind(kind)
    _validate_positive_integer(horizon, name="horizon")
    _validate_positive_integer(action_dimension, name="action_dimension")
    _validate_seed(seed)
    limit = _validate_action_limit(action_limit)
    rng = np.random.default_rng(
        derive_deterministic_seed(
            seed,
            "branch-action-plan-v1",
            kind,
            horizon,
            action_dimension,
            limit,
        )
    )

    shape = (horizon, action_dimension)
    if kind == "smooth_ar1":
        actions = _smooth_ar1(rng, shape=shape, limit=limit)
    elif kind == "piecewise":
        actions = _piecewise(rng, shape=shape, limit=limit)
    elif kind == "periodic":
        actions = _periodic(rng, shape=shape, limit=limit)
    else:
        actions = rng.uniform(-limit, limit, size=shape)
    actions = np.clip(actions, -limit, limit).astype(np.float32, copy=False)
    return BranchActionPlan(kind=kind, seed=seed, actions=actions)


def build_action_plan_schedule(
    count: int,
    kinds: Sequence[str] | Mapping[str, float] = ACTION_PLAN_KINDS,
    *,
    seed: int = 0,
) -> tuple[str, ...]:

    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    _validate_seed(seed)
    if isinstance(kinds, Mapping):
        keys = tuple(kinds)
        raw_weights = tuple(float(kinds[key]) for key in keys)
    else:
        keys = tuple(kinds)
        if not keys:
            raise ValueError("kinds must not be empty")
        raw_weights = tuple(1.0 for _ in keys)
    if not keys:
        raise ValueError("kinds must not be empty")
    if len(set(keys)) != len(keys):
        raise ValueError("kinds must be unique")
    for kind in keys:
        _validate_kind(kind)
    if any(not math.isfinite(weight) or weight < 0.0 for weight in raw_weights):
        raise ValueError("schedule weights must be finite and non-negative")
    total_weight = sum(raw_weights)
    if total_weight <= 0.0:
        raise ValueError("at least one schedule weight must be positive")
    if count == 0:
        return ()

    weights = tuple(weight / total_weight for weight in raw_weights)
    expected = tuple(count * weight for weight in weights)
    allocations = [int(math.floor(value)) for value in expected]
    remaining = count - sum(allocations)
    tie_rng = np.random.default_rng(
        derive_deterministic_seed(seed, "branch-action-schedule-ties-v1", *keys)
    )
    tie_breakers = tie_rng.random(len(keys))
    ranked = sorted(
        range(len(keys)),
        key=lambda index: (
            -(expected[index] - allocations[index]),
            tie_breakers[index],
            index,
        ),
    )
    for index in ranked[:remaining]:
        allocations[index] += 1

    schedule = [
        kind
        for kind, allocation in zip(keys, allocations, strict=True)
        for _ in range(allocation)
    ]
    permutation_rng = np.random.default_rng(
        derive_deterministic_seed(seed, "branch-action-schedule-order-v1", *keys)
    )
    order = permutation_rng.permutation(len(schedule))
    return tuple(schedule[int(index)] for index in order)


def compute_action_plan_sha256(actions: np.ndarray) -> str:

    array = _validated_actions(actions)
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    header = json.dumps(
        {"dtype": "float32-le", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def derive_deterministic_seed(base_seed: int, *components: Any) -> int:

    _validate_seed(base_seed)
    payload = json.dumps(
        [base_seed, *components],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _smooth_ar1(
    rng: np.random.Generator,
    *,
    shape: tuple[int, int],
    limit: float,
) -> np.ndarray:
    horizon, action_dimension = shape
    rho = 0.82
    innovations = rng.uniform(-limit, limit, size=shape)
    actions = np.empty(shape, dtype=np.float64)
    actions[0] = innovations[0]
    innovation_scale = math.sqrt(1.0 - rho * rho)
    for step in range(1, horizon):
        actions[step] = rho * actions[step - 1] + innovation_scale * innovations[step]
    assert actions.shape[1] == action_dimension
    return actions


def _piecewise(
    rng: np.random.Generator,
    *,
    shape: tuple[int, int],
    limit: float,
) -> np.ndarray:
    horizon, action_dimension = shape
    segment_count = min(3, max(1, math.ceil(horizon / 2)))
    if segment_count == 1:
        boundaries = np.array([0, horizon], dtype=np.int64)
    else:
        possible = np.arange(1, horizon, dtype=np.int64)
        selected = np.sort(rng.choice(possible, size=segment_count - 1, replace=False))
        boundaries = np.concatenate((np.array([0]), selected, np.array([horizon])))
    values = rng.uniform(
        -limit,
        limit,
        size=(segment_count, action_dimension),
    )
    actions = np.empty(shape, dtype=np.float64)
    for segment in range(segment_count):
        actions[boundaries[segment] : boundaries[segment + 1]] = values[segment]
    return actions


def _periodic(
    rng: np.random.Generator,
    *,
    shape: tuple[int, int],
    limit: float,
) -> np.ndarray:
    horizon, action_dimension = shape
    time = np.arange(horizon, dtype=np.float64)[:, None]
    cycles = rng.integers(1, 3, size=(1, action_dimension))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=(1, action_dimension))
    amplitudes = rng.uniform(0.55, 1.0, size=(1, action_dimension)) * limit
    biases = rng.uniform(-0.12, 0.12, size=(1, action_dimension)) * limit
    return (
        amplitudes * np.sin(2.0 * math.pi * cycles * time / max(horizon, 1) + phases)
        + biases
    )


def _validated_actions(actions: np.ndarray) -> np.ndarray:
    array = np.asarray(actions)
    if array.ndim != 2:
        raise ValueError("actions must have shape [horizon, action_dimension]")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("actions dimensions must be positive")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("actions must have a floating dtype")
    array = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError("actions must be finite")
    array.setflags(write=False)
    return array


def _validate_kind(kind: str) -> None:
    if not isinstance(kind, str):
        raise TypeError("action plan kind must be a string")
    if kind not in ACTION_PLAN_KINDS:
        raise ValueError(
            f"unknown action plan kind {kind!r}; expected one of {ACTION_PLAN_KINDS}"
        )


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _validate_positive_integer(value: int, *, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_action_limit(action_limit: float) -> float:
    if not isinstance(action_limit, (int, float)) or isinstance(action_limit, bool):
        raise TypeError("action_limit must be numeric")
    result = float(action_limit)
    if not math.isfinite(result) or result <= 0.0 or result > 1.0:
        raise ValueError("action_limit must be finite and in (0, 1]")
    return result


__all__ = [
    "BranchActionPlan",
    "build_action_plan_schedule",
    "compute_action_plan_sha256",
    "derive_deterministic_seed",
    "generate_action_plan",
]
