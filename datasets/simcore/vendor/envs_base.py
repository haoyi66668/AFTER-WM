from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


class UnsupportedDynamicsEditError(RuntimeError):
    pass


class UnsupportedStateSnapshotError(RuntimeError):
    pass


@dataclass(slots=True)
class ResetResult:

    raw_observation: Any
    flat_observation: FloatArray
    info: dict[str, Any]


@dataclass(slots=True)
class StepResult:

    raw_observation: Any
    flat_observation: FloatArray
    model_action: FloatArray
    environment_action: FloatArray
    reward: float
    terminated: bool
    truncated: bool
    episode_end: bool
    info: dict[str, Any]


class EnvAdapter(ABC):

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        pass

    @property
    def supports_dynamics_edit(self) -> bool:

        return False

    @property
    def supports_full_state_snapshot(self) -> bool:

        return False

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass

    @abstractmethod
    def reset(self, seed: int) -> ResetResult:
        pass

    @abstractmethod
    def step(self, model_action: FloatArray) -> StepResult:
        pass

    def restore_base_dynamics(self) -> None:

        return

    def apply_absolute_dynamics(
        self,
        parameters: Mapping[str, float],
    ) -> None:

        if parameters:
            raise UnsupportedDynamicsEditError(
                f"{type(self).__name__} does not support dynamics editing"
            )

    def get_dynamics_metadata(self) -> dict[str, Any]:

        return {}

    def capture_integration_snapshot(self) -> Any:

        raise UnsupportedStateSnapshotError(
            f"{type(self).__name__} does not support full-state snapshots"
        )

    def restore_integration_snapshot(
        self,
        snapshot: Any,
        parameters: Mapping[str, float] | None = None,
    ) -> ResetResult:

        del snapshot, parameters
        raise UnsupportedStateSnapshotError(
            f"{type(self).__name__} does not support full-state snapshots"
        )

    def get_contact_summary(self) -> dict[str, Any]:

        return {}

    @abstractmethod
    def get_backend_metadata(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
