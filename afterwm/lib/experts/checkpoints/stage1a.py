from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from afterwm import legacy as LEGACY
from afterwm.lib.experts.models import Stage1ABaseWorldModel, Stage1AModelConfig

STAGE1B0_BASE_SHA256 = (
    "31b2395cb46b5a3d2dfe934b84712d8bb81bf7ff7e9d985fa42609daec09ef87"
)

_CHECKPOINT_FORMAT = LEGACY.STAGE1_CHECKPOINT_FORMAT
_SUPPORTED_CHECKPOINT_VERSIONS = frozenset((1, 2))
_MODEL_CONFIG_KEYS = frozenset(
    (
        "observation_dim",
        "action_dim",
        "num_experts",
        "router_hidden_dims",
        "expert_hidden_dims",
        "alpha_q",
        "route_probability_floor",
        "epsilon",
    )
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class FrozenStage1ANormalizerIdentity:

    dataset_manifest_sha256: str
    observation_dim: int
    action_dim: int
    observation_count: int
    action_count: int
    epsilon: float
    observation_mean: tuple[float, ...]
    observation_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FrozenStage1ABase:

    model: Stage1ABaseWorldModel
    config: Stage1AModelConfig
    checkpoint_path: Path
    checkpoint_sha256: str
    state_sha256: str
    epoch: int
    validation_score: float | None
    normalizer_identity: FrozenStage1ANormalizerIdentity


def file_sha256(path: str | Path) -> str:

    selected_path = Path(path)

    if not selected_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {selected_path}")

    digest = hashlib.sha256()

    with selected_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def stage1a_state_sha256(
    model_or_state: Stage1ABaseWorldModel | Mapping[str, Tensor],
) -> str:


    if isinstance(model_or_state, Stage1ABaseWorldModel):
        state = model_or_state.state_dict()
    elif isinstance(model_or_state, Mapping):
        state = model_or_state
    else:
        raise TypeError("model_or_state must be a Stage1ABaseWorldModel or mapping")

    if not state:
        raise ValueError("Stage 1A state_dict must not be empty")

    digest = hashlib.sha256()

    for name in sorted(state):
        value = state[name]

        if not isinstance(name, str):
            raise TypeError("Stage 1A state_dict keys must be strings")
        if not isinstance(value, Tensor):
            raise TypeError("Stage 1A state_dict values must be tensors")

        tensor = value.detach().to(device="cpu").contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")

        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(8, "big"))
        digest.update(dtype_bytes)
        digest.update(len(tensor.shape).to_bytes(8, "big"))

        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=True))

        byte_view = tensor.reshape(-1).view(torch.uint8)
        digest.update(byte_view.numpy().tobytes(order="C"))

    return digest.hexdigest()


def load_frozen_stage1a_base(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str = STAGE1B0_BASE_SHA256,
    device: str | torch.device = "cpu",
    expected_config: Stage1AModelConfig | None = None,
) -> FrozenStage1ABase:


    selected_path = Path(checkpoint_path).resolve()
    expected_digest = _normalize_sha256(expected_sha256, name="expected_sha256")
    actual_digest = file_sha256(selected_path)

    if actual_digest != expected_digest:
        raise ValueError(
            "Stage 1A checkpoint SHA-256 mismatch: "
            f"expected {expected_digest}, got {actual_digest}"
        )

    checkpoint = torch.load(
        selected_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint root must be a mapping")

    _validate_checkpoint_header(checkpoint)
    _validate_strict_unlabelled_policy(checkpoint)

    model_config = _load_model_config(checkpoint.get("model_config"))
    normalizer_identity = _load_normalizer_identity(checkpoint.get("normalizer"))

    if expected_config is None:
        expected_config = Stage1AModelConfig(
            observation_dim=17,
            action_dim=6,
            num_experts=3,
            router_hidden_dims=(128, 128),
            expert_hidden_dims=(256, 256, 256),
            alpha_q=0.5,
            route_probability_floor=1e-3,
            epsilon=1e-8,
        )

    if model_config != expected_config:
        raise ValueError(
            "Stage 1A checkpoint model_config does not match the frozen "
            "Stage 1B base configuration"
        )

    state = checkpoint.get("model_state_dict")

    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint.model_state_dict must be a non-empty mapping")
    if not all(isinstance(name, str) for name in state):
        raise TypeError("checkpoint.model_state_dict keys must be strings")
    if not all(isinstance(value, Tensor) for value in state.values()):
        raise TypeError("checkpoint.model_state_dict values must be tensors")

    model = Stage1ABaseWorldModel(model_config)

    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:
        raise ValueError(
            "checkpoint model_state_dict is not strictly loadable under model_config"
        ) from error

    model.to(torch.device(device))
    model.requires_grad_(False)
    model.eval()

    state_digest = stage1a_state_sha256(model)
    score = _optional_finite_float(
        checkpoint.get(
            "best_validation_score",
            checkpoint.get("validation_score"),
        ),
        name="checkpoint validation score",
    )

    frozen = FrozenStage1ABase(
        model=model,
        config=model_config,
        checkpoint_path=selected_path,
        checkpoint_sha256=actual_digest,
        state_sha256=state_digest,
        epoch=_positive_integer(checkpoint.get("epoch"), name="checkpoint.epoch"),
        validation_score=score,
        normalizer_identity=normalizer_identity,
    )
    assert_frozen_stage1a_unchanged(frozen)
    return frozen


def assert_frozen_stage1a_unchanged(
    frozen_or_model: FrozenStage1ABase | Stage1ABaseWorldModel,
    *,
    expected_state_sha256: str | None = None,
) -> None:

    if isinstance(frozen_or_model, FrozenStage1ABase):
        model = frozen_or_model.model
        expected_digest = frozen_or_model.state_sha256
    elif isinstance(frozen_or_model, Stage1ABaseWorldModel):
        model = frozen_or_model

        if expected_state_sha256 is None:
            raise ValueError(
                "expected_state_sha256 is required when passing a bare model"
            )

        expected_digest = _normalize_sha256(
            expected_state_sha256,
            name="expected_state_sha256",
        )
    else:
        raise TypeError(
            "frozen_or_model must be FrozenStage1ABase or Stage1ABaseWorldModel"
        )

    if model.training:
        raise RuntimeError("frozen Stage 1A base must remain in eval mode")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            raise RuntimeError(f"frozen Stage 1A parameter requires grad: {name}")
        if parameter.grad is not None:
            raise RuntimeError(f"frozen Stage 1A parameter has a gradient: {name}")

    actual_digest = stage1a_state_sha256(model)

    if actual_digest != expected_digest:
        raise RuntimeError(
            "frozen Stage 1A state changed: "
            f"expected {expected_digest}, got {actual_digest}"
        )


def _validate_checkpoint_header(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("format_name") != _CHECKPOINT_FORMAT:
        raise ValueError("checkpoint has an unexpected format_name")

    version = checkpoint.get("format_version")

    if not isinstance(version, int) or isinstance(version, bool):
        raise TypeError("checkpoint.format_version must be an integer")
    if version not in _SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError("checkpoint has an unsupported format_version")

    _positive_integer(checkpoint.get("epoch"), name="checkpoint.epoch")


def _load_model_config(value: Any) -> Stage1AModelConfig:
    if not isinstance(value, Mapping):
        raise TypeError("checkpoint.model_config must be a mapping")

    keys = frozenset(value)

    if keys != _MODEL_CONFIG_KEYS:
        missing = sorted(_MODEL_CONFIG_KEYS - keys)
        extra = sorted(keys - _MODEL_CONFIG_KEYS)
        raise ValueError(
            "checkpoint.model_config keys do not match the Stage 1A schema: "
            f"missing={missing}, extra={extra}"
        )

    try:
        return Stage1AModelConfig(
            observation_dim=value["observation_dim"],
            action_dim=value["action_dim"],
            num_experts=value["num_experts"],
            router_hidden_dims=tuple(value["router_hidden_dims"]),
            expert_hidden_dims=tuple(value["expert_hidden_dims"]),
            alpha_q=value["alpha_q"],
            route_probability_floor=value["route_probability_floor"],
            epsilon=value["epsilon"],
        )
    except Exception as error:
        raise ValueError("checkpoint.model_config is invalid") from error


def _validate_strict_unlabelled_policy(checkpoint: Mapping[str, Any]) -> None:
    train = _mapping(checkpoint.get("train_config"), name="checkpoint.train_config")
    label = _mapping(checkpoint.get("label_policy"), name="checkpoint.label_policy")
    metrics = _mapping(checkpoint.get("metrics"), name="checkpoint.metrics")
    train_metrics = _mapping(metrics.get("train"), name="checkpoint.metrics.train")
    optimization = _mapping(
        metrics.get("optimization_policy"),
        name="checkpoint.metrics.optimization_policy",
    )

    _require_zero(train.get("mode_weight"), name="train_config.mode_weight")
    _require_false(
        train.get("supervise_router_mode"),
        name="train_config.supervise_router_mode",
    )
    _require_empty_mode_mapping(
        train.get("mode_to_expert"),
        name="train_config.mode_to_expert",
    )

    _require_false(
        label.get("labels_used_during_warmup"),
        name="label_policy.labels_used_during_warmup",
    )
    _require_zero(
        label.get("expert_mode_warmup_epochs"),
        name="label_policy.expert_mode_warmup_epochs",
    )
    _require_zero(
        label.get("expert_mode_warmup_initial_weight"),
        name="label_policy.expert_mode_warmup_initial_weight",
    )
    _require_empty_mode_mapping(
        label.get("mode_to_expert"),
        name="label_policy.mode_to_expert",
    )
    _require_false(
        label.get("router_mode_supervision"),
        name="label_policy.router_mode_supervision",
    )

    if label.get("saved_after_mode_weight_reached_zero") is not True:
        raise ValueError(
            "label_policy.saved_after_mode_weight_reached_zero must be true"
        )

    _require_zero(
        train_metrics.get("labelled_transition_count"),
        name="metrics.train.labelled_transition_count",
    )
    _require_false(
        optimization.get("mode_labels_used_for_expert_loss"),
        name="metrics.optimization_policy.mode_labels_used_for_expert_loss",
    )
    _require_false(
        optimization.get("router_mode_supervision"),
        name="metrics.optimization_policy.router_mode_supervision",
    )
    _require_zero(
        optimization.get("expert_mode_weight"),
        name="metrics.optimization_policy.expert_mode_weight",
    )


def _load_normalizer_identity(value: Any) -> FrozenStage1ANormalizerIdentity:
    metadata = _mapping(value, name="checkpoint.normalizer")
    manifest_sha256 = metadata.get("dataset_manifest_sha256")

    if not isinstance(manifest_sha256, str):
        raise TypeError(
            "checkpoint.normalizer.dataset_manifest_sha256 must be a string"
        )
    manifest_sha256 = _normalize_sha256(
        manifest_sha256,
        name="checkpoint.normalizer.dataset_manifest_sha256",
    )
    observation_dim = _positive_integer(
        metadata.get("observation_dim"),
        name="checkpoint.normalizer.observation_dim",
    )
    action_dim = _positive_integer(
        metadata.get("action_dim"),
        name="checkpoint.normalizer.action_dim",
    )
    observation_mean = _finite_vector(
        metadata.get("observation_mean"),
        name="checkpoint.normalizer.observation_mean",
        expected_length=observation_dim,
    )
    observation_std = _positive_finite_vector(
        metadata.get("observation_std"),
        name="checkpoint.normalizer.observation_std",
        expected_length=observation_dim,
    )
    action_mean = _finite_vector(
        metadata.get("action_mean"),
        name="checkpoint.normalizer.action_mean",
        expected_length=action_dim,
    )
    action_std = _positive_finite_vector(
        metadata.get("action_std"),
        name="checkpoint.normalizer.action_std",
        expected_length=action_dim,
    )
    epsilon = _positive_finite_float(
        metadata.get("epsilon"),
        name="checkpoint.normalizer.epsilon",
    )

    return FrozenStage1ANormalizerIdentity(
        dataset_manifest_sha256=manifest_sha256,
        observation_dim=observation_dim,
        action_dim=action_dim,
        observation_count=_positive_integer(
            metadata.get("observation_count"),
            name="checkpoint.normalizer.observation_count",
        ),
        action_count=_positive_integer(
            metadata.get("action_count"),
            name="checkpoint.normalizer.action_count",
        ),
        epsilon=epsilon,
        observation_mean=observation_mean,
        observation_std=observation_std,
        action_mean=action_mean,
        action_std=action_std,
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_finite_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric or null")

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric or null") from error

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite_float(value: Any, *, name: str) -> float:
    result = _optional_finite_float(value, name=name)
    if result is None or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite_vector(
    value: Any,
    *,
    name: str,
    expected_length: int,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    if len(value) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values")

    result = tuple(
        _optional_finite_float(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if any(item is None for item in result):
        raise ValueError(f"{name} must contain only finite numeric values")
    return tuple(float(item) for item in result)


def _positive_finite_vector(
    value: Any,
    *,
    name: str,
    expected_length: int,
) -> tuple[float, ...]:
    result = _finite_vector(value, name=name, expected_length=expected_length)
    if any(item <= 0.0 for item in result):
        raise ValueError(f"{name} must contain only positive values")
    return result


def _require_zero(value: Any, *, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error

    if numeric != 0.0:
        raise ValueError(f"{name} must be zero for strict unlabelled Stage 1A")


def _require_false(value: Any, *, name: str) -> None:
    if value is not False:
        raise ValueError(f"{name} must be false for strict unlabelled Stage 1A")


def _require_empty_mode_mapping(value: Any, *, name: str) -> None:
    if value not in ([], ()):
        raise ValueError(f"{name} must be empty for strict unlabelled Stage 1A")


def _normalize_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    normalized = value.strip().lower()

    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")

    return normalized
