from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces.utils import flatdim, flatten

from .envs_base import EnvAdapter, ResetResult, StepResult

DYNAMICS_EDIT_SEMANTICS_VERSION = "gym_mujoco_contact_friction_v2"
FRICTION_SCOPE = "all_contact_geoms"
STATE_SNAPSHOT_SEMANTICS_VERSION = "mujoco_mjSTATE_INTEGRATION_v1"


def _canonical_array_sha256(array: np.ndarray) -> str:

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _snapshot_sha256(metadata: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class MujocoIntegrationSnapshot:

    integration_state: np.ndarray
    flat_observation: np.ndarray
    state_spec: int
    elapsed_steps: int
    environment_rng_state: dict[str, Any] | None
    action_space_rng_state: dict[str, Any] | None
    environment_id: str
    observation_dim: int
    action_dim: int
    model_sha256: str
    state_sha256: str
    observation_sha256: str
    snapshot_sha256: str
    gymnasium_version: str
    mujoco_version: str
    snapshot_semantics_version: str
    source_friction_scale: float

    def __post_init__(self) -> None:
        integration_state = (
            np.asarray(
                self.integration_state,
                dtype=np.float64,
            )
            .reshape(-1)
            .copy()
        )
        flat_observation = (
            np.asarray(
                self.flat_observation,
                dtype=np.float32,
            )
            .reshape(-1)
            .copy()
        )
        integration_state.setflags(write=False)
        flat_observation.setflags(write=False)
        object.__setattr__(self, "integration_state", integration_state)
        object.__setattr__(self, "flat_observation", flat_observation)
        object.__setattr__(
            self,
            "environment_rng_state",
            deepcopy(self.environment_rng_state),
        )
        object.__setattr__(
            self,
            "action_space_rng_state",
            deepcopy(self.action_space_rng_state),
        )

    def to_metadata(self) -> dict[str, Any]:

        return _json_safe(
            {
                "state_spec": self.state_spec,
                "elapsed_steps": self.elapsed_steps,
                "environment_rng_state": self.environment_rng_state,
                "action_space_rng_state": self.action_space_rng_state,
                "environment_id": self.environment_id,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "model_sha256": self.model_sha256,
                "state_sha256": self.state_sha256,
                "observation_sha256": self.observation_sha256,
                "snapshot_sha256": self.snapshot_sha256,
                "gymnasium_version": self.gymnasium_version,
                "mujoco_version": self.mujoco_version,
                "snapshot_semantics_version": self.snapshot_semantics_version,
                "state_snapshot_semantics": self.snapshot_semantics_version,
                "integration_state_sha256": self.state_sha256,
                "source_friction_scale": self.source_friction_scale,
            }
        )

    @classmethod
    def from_arrays(
        cls,
        integration_state: np.ndarray,
        flat_observation: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> MujocoIntegrationSnapshot:

        required = {
            "state_spec",
            "elapsed_steps",
            "environment_id",
            "observation_dim",
            "action_dim",
            "model_sha256",
            "state_sha256",
            "observation_sha256",
            "snapshot_sha256",
            "gymnasium_version",
            "mujoco_version",
            "snapshot_semantics_version",
            "source_friction_scale",
        }
        missing = required - set(metadata)
        if missing:
            raise KeyError(f"snapshot metadata is missing fields: {sorted(missing)}")

        return cls(
            integration_state=integration_state,
            flat_observation=flat_observation,
            state_spec=int(metadata["state_spec"]),
            elapsed_steps=int(metadata["elapsed_steps"]),
            environment_rng_state=deepcopy(metadata.get("environment_rng_state")),
            action_space_rng_state=deepcopy(metadata.get("action_space_rng_state")),
            environment_id=str(metadata["environment_id"]),
            observation_dim=int(metadata["observation_dim"]),
            action_dim=int(metadata["action_dim"]),
            model_sha256=str(metadata["model_sha256"]),
            state_sha256=str(metadata["state_sha256"]),
            observation_sha256=str(metadata["observation_sha256"]),
            snapshot_sha256=str(metadata["snapshot_sha256"]),
            gymnasium_version=str(metadata["gymnasium_version"]),
            mujoco_version=str(metadata["mujoco_version"]),
            snapshot_semantics_version=str(metadata["snapshot_semantics_version"]),
            source_friction_scale=float(metadata["source_friction_scale"]),
        )


class GymMujocoAdapter(EnvAdapter):

    def __init__(
        self,
        env_id: str,
        max_episode_steps: int,
        render_mode: str | None = None,
    ) -> None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be greater than 0.")

        make_kwargs: dict[str, Any] = {
            "max_episode_steps": max_episode_steps,
        }

        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode

        self._env_id = env_id
        self._max_episode_steps = max_episode_steps
        self._env = gym.make(env_id, **make_kwargs)

        if not isinstance(self._env.action_space, Box):
            raise TypeError(
                "only continuous Box action spaces are supported, "
                f"got {type(self._env.action_space)!r}."
            )

        if not np.isfinite(self._env.action_space.low).all():
            raise ValueError("all action-space lower bounds must be finite.")

        if not np.isfinite(self._env.action_space.high).all():
            raise ValueError("all action-space upper bounds must be finite.")

        self._observation_dim = int(flatdim(self._env.observation_space))
        self._action_dim = int(np.prod(self._env.action_space.shape))

        self._action_low = np.asarray(
            self._env.action_space.low,
            dtype=np.float32,
        ).reshape(-1)

        self._action_high = np.asarray(
            self._env.action_space.high,
            dtype=np.float32,
        ).reshape(-1)

        unwrapped = self._env.unwrapped

        if not hasattr(unwrapped, "model") or not hasattr(unwrapped, "data"):
            raise TypeError(
                f"{env_id} does not expose MuJoCo model/data; "
                "check that this is a Gymnasium MuJoCo environment."
            )

        self._model = unwrapped.model
        self._data = unwrapped.data

        self._base_geom_friction = self._model.geom_friction.copy()
        self._base_pair_friction = self._model.pair_friction.copy()

        contact_geom_mask = np.logical_or(
            np.asarray(self._model.geom_contype) != 0,
            np.asarray(self._model.geom_conaffinity) != 0,
        )

        pair_count = int(getattr(self._model, "npair", 0))

        if pair_count > 0:
            pair_geom_ids = np.concatenate(
                (
                    np.asarray(self._model.pair_geom1[:pair_count]),
                    np.asarray(self._model.pair_geom2[:pair_count]),
                )
            ).astype(np.int64, copy=False)
            contact_geom_mask[pair_geom_ids] = True

        self._contact_geom_mask = np.asarray(
            contact_geom_mask,
            dtype=np.bool_,
        ).copy()
        self._contact_geom_ids = np.flatnonzero(self._contact_geom_mask)

        if self._contact_geom_ids.size == 0:
            raise RuntimeError(
                f"{env_id} has no geom that can take part in contacts; "
                "the contact-friction edit cannot be applied."
            )

        self._floor_geom_id = int(
            mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "floor",
            )
        )

        self._current_friction_scale = 1.0
        self._last_flat_observation: np.ndarray | None = None
        self._model_sha256 = self._compute_model_sha256()

        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._integration_state_spec = int(state_spec)
        self._integration_state_size = int(mujoco.mj_stateSize(self._model, state_spec))

        if self._integration_state_size <= 0:
            raise RuntimeError("the mjSTATE_INTEGRATION state size must be greater than 0.")

    @property
    def observation_dim(self) -> int:
        return self._observation_dim

    @property
    def supports_dynamics_edit(self) -> bool:
        return True

    @property
    def supports_full_state_snapshot(self) -> bool:
        return True

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _flatten_observation(self, raw_observation: Any) -> np.ndarray:
        flat = flatten(
            self._env.observation_space,
            raw_observation,
        )

        flat = np.asarray(flat, dtype=np.float32).reshape(-1).copy()

        if flat.shape != (self.observation_dim,):
            raise ValueError(
                f"wrong flattened observation shape: expected {(self.observation_dim,)}, got {flat.shape}."
            )

        if not np.isfinite(flat).all():
            raise FloatingPointError("NaN or Inf in the observation.")

        return flat

    def _prepare_model_action(
        self,
        model_action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        model_action = np.asarray(
            model_action,
            dtype=np.float32,
        ).reshape(-1)

        if model_action.shape != (self.action_dim,):
            raise ValueError(
                "wrong model action shape: "
                f"expected {(self.action_dim,)}, got {model_action.shape}."
            )

        if not np.isfinite(model_action).all():
            raise FloatingPointError("NaN or Inf in the model action.")

        model_action = np.clip(model_action, -1.0, 1.0)

        environment_action = self._action_low + 0.5 * (model_action + 1.0) * (
            self._action_high - self._action_low
        )

        environment_action = environment_action.astype(
            self._env.action_space.dtype,
            copy=False,
        ).reshape(self._env.action_space.shape)

        return model_action.copy(), environment_action.copy()

    def reset(self, seed: int) -> ResetResult:
        self.restore_base_dynamics()

        self._env.action_space.seed(seed)
        raw_observation, info = self._env.reset(seed=seed)

        flat_observation = self._flatten_observation(raw_observation)
        self._last_flat_observation = flat_observation.copy()

        return ResetResult(
            raw_observation=raw_observation,
            flat_observation=flat_observation,
            info=dict(info),
        )

    def step(self, model_action: np.ndarray) -> StepResult:
        model_action, environment_action = self._prepare_model_action(model_action)

        (
            raw_next,
            reward,
            terminated,
            truncated,
            info,
        ) = self._env.step(environment_action)

        flat_next = self._flatten_observation(raw_next)
        self._last_flat_observation = flat_next.copy()
        reward = float(reward)

        if not np.isfinite(reward):
            raise FloatingPointError("NaN or Inf in the environment reward.")

        terminated = bool(terminated)
        truncated = bool(truncated)

        return StepResult(
            raw_observation=raw_next,
            flat_observation=flat_next,
            model_action=model_action,
            environment_action=np.asarray(
                environment_action,
                dtype=np.float32,
            )
            .reshape(-1)
            .copy(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_end=terminated or truncated,
            info=dict(info),
        )

    def _set_absolute_dynamics_without_forward(
        self,
        parameters: Mapping[str, float],
    ) -> None:
        allowed = {"friction_scale"}
        unknown = set(parameters) - allowed

        if unknown:
            raise KeyError(f"unknown dynamics parameters: {sorted(unknown)}")

        friction_scale = float(parameters.get("friction_scale", 1.0))

        if not np.isfinite(friction_scale) or friction_scale <= 0:
            raise ValueError("friction_scale must be finite and positive.")

        self._model.geom_friction[...] = self._base_geom_friction
        self._model.geom_friction[self._contact_geom_mask] = (
            self._base_geom_friction[self._contact_geom_mask] * friction_scale
        )

        self._model.pair_friction[...] = self._base_pair_friction * friction_scale
        self._current_friction_scale = friction_scale

    def restore_base_dynamics(self) -> None:
        self._set_absolute_dynamics_without_forward({"friction_scale": 1.0})

        mujoco.mj_forward(self._model, self._data)

    def apply_absolute_dynamics(
        self,
        parameters: Mapping[str, float],
    ) -> None:
        self._set_absolute_dynamics_without_forward(parameters)
        mujoco.mj_forward(self._model, self._data)

    @staticmethod
    def _update_digest_with_array(
        digest: Any,
        name: str,
        value: Any,
    ) -> None:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))

    def _compute_model_sha256(self) -> str:

        digest = hashlib.sha256()
        model_buffer = np.empty(
            int(mujoco.mj_sizeModel(self._model)),
            dtype=np.uint8,
        )
        mujoco.mj_saveModel(self._model, None, model_buffer)
        digest.update(b"compiled-mjb\0")
        digest.update(model_buffer.tobytes(order="C"))
        descriptor = {
            "environment_id": self._env_id,
            "nq": int(self._model.nq),
            "nv": int(self._model.nv),
            "na": int(self._model.na),
            "nu": int(self._model.nu),
            "nbody": int(self._model.nbody),
            "njnt": int(self._model.njnt),
            "ngeom": int(self._model.ngeom),
            "npair": int(self._model.npair),
            "nsite": int(self._model.nsite),
            "ntendon": int(self._model.ntendon),
            "nactuator": int(self._model.nactuator),
            "nsensor": int(self._model.nsensor),
            "timestep": float(self._model.opt.timestep),
        }
        digest.update(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
        )

        static_arrays = {
            "names": self._model.names,
            "body_parentid": self._model.body_parentid,
            "body_pos": self._model.body_pos,
            "body_quat": self._model.body_quat,
            "body_mass": self._model.body_mass,
            "body_inertia": self._model.body_inertia,
            "jnt_type": self._model.jnt_type,
            "jnt_bodyid": self._model.jnt_bodyid,
            "jnt_pos": self._model.jnt_pos,
            "dof_damping": self._model.dof_damping,
            "geom_type": self._model.geom_type,
            "geom_bodyid": self._model.geom_bodyid,
            "geom_pos": self._model.geom_pos,
            "geom_quat": self._model.geom_quat,
            "geom_size": self._model.geom_size,
            "geom_contype": self._model.geom_contype,
            "geom_conaffinity": self._model.geom_conaffinity,
            "actuator_trnid": self._model.actuator_trnid,
            "actuator_gear": self._model.actuator_gear,
            "base_geom_friction": self._base_geom_friction,
            "base_pair_friction": self._base_pair_friction,
        }

        for name, value in static_arrays.items():
            self._update_digest_with_array(digest, name, value)

        return digest.hexdigest()

    def _current_raw_observation(self) -> Any:
        getter = getattr(self._env.unwrapped, "_get_obs", None)

        if getter is None or not callable(getter):
            raise RuntimeError(
                f"{self._env_id} has no callable _get_obs; the state restore cannot be verified."
            )

        return getter()

    def _elapsed_steps(self) -> int:
        current: Any = self._env
        visited: set[int] = set()

        while id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "_elapsed_steps"):
                elapsed = current._elapsed_steps
                if elapsed is None:
                    raise RuntimeError("the environment has not been reset; elapsed_steps cannot be captured.")
                return int(elapsed)
            current = getattr(current, "env", None)
            if current is None:
                break

        raise RuntimeError("no TimeLimit elapsed_steps found in the wrapper chain.")

    def _restore_elapsed_steps(self, elapsed_steps: int) -> None:
        if elapsed_steps < 0 or elapsed_steps > self._max_episode_steps:
            raise ValueError(
                "snapshot elapsed_steps out of range: "
                f"{elapsed_steps} not in [0, {self._max_episode_steps}]"
            )

        current: Any = self._env
        visited: set[int] = set()
        restored = False

        while id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "_elapsed_steps"):
                current._elapsed_steps = elapsed_steps
                restored = True
            current = getattr(current, "env", None)
            if current is None:
                break

        if not restored:
            raise RuntimeError("no TimeLimit elapsed_steps found in the wrapper chain.")

    @staticmethod
    def _rng_state(rng: Any) -> dict[str, Any] | None:
        if rng is None:
            return None
        bit_generator = getattr(rng, "bit_generator", None)
        if bit_generator is None:
            raise TypeError("the environment RNG has no bit_generator.state.")
        return deepcopy(bit_generator.state)

    @staticmethod
    def _restore_rng_state(rng: Any, state: Mapping[str, Any] | None) -> None:
        if state is None:
            return
        if rng is None or not hasattr(rng, "bit_generator"):
            raise RuntimeError("the snapshot contains an RNG state but the environment has no compatible RNG.")
        rng.bit_generator.state = deepcopy(dict(state))

    def _snapshot_metadata_without_digest(
        self,
        *,
        state_sha256: str,
        observation_sha256: str,
        elapsed_steps: int,
        environment_rng_state: Mapping[str, Any] | None,
        action_space_rng_state: Mapping[str, Any] | None,
        source_friction_scale: float,
    ) -> dict[str, Any]:
        return {
            "state_spec": self._integration_state_spec,
            "elapsed_steps": elapsed_steps,
            "environment_rng_state": environment_rng_state,
            "action_space_rng_state": action_space_rng_state,
            "environment_id": self._env_id,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "model_sha256": self._model_sha256,
            "state_sha256": state_sha256,
            "observation_sha256": observation_sha256,
            "gymnasium_version": gym.__version__,
            "mujoco_version": mujoco.__version__,
            "snapshot_semantics_version": STATE_SNAPSHOT_SEMANTICS_VERSION,
            "source_friction_scale": source_friction_scale,
        }

    def capture_integration_snapshot(self) -> MujocoIntegrationSnapshot:
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration_state = np.empty(
            self._integration_state_size,
            dtype=np.float64,
        )
        mujoco.mj_getState(
            self._model,
            self._data,
            integration_state,
            state_spec,
        )

        if not np.isfinite(integration_state).all():
            raise FloatingPointError("NaN or Inf in mjSTATE_INTEGRATION.")

        raw_observation = self._current_raw_observation()
        flat_observation = self._flatten_observation(raw_observation)
        elapsed_steps = self._elapsed_steps()
        environment_rng_state = self._rng_state(self._env.unwrapped.np_random)
        action_space_rng_state = self._rng_state(self._env.action_space.np_random)
        state_sha256 = _canonical_array_sha256(integration_state)
        observation_sha256 = _canonical_array_sha256(flat_observation)
        metadata = self._snapshot_metadata_without_digest(
            state_sha256=state_sha256,
            observation_sha256=observation_sha256,
            elapsed_steps=elapsed_steps,
            environment_rng_state=environment_rng_state,
            action_space_rng_state=action_space_rng_state,
            source_friction_scale=self._current_friction_scale,
        )
        snapshot_digest = _snapshot_sha256(metadata)

        return MujocoIntegrationSnapshot(
            integration_state=integration_state,
            flat_observation=flat_observation,
            snapshot_sha256=snapshot_digest,
            **metadata,
        )

    def _validate_snapshot(self, snapshot: MujocoIntegrationSnapshot) -> None:
        if not isinstance(snapshot, MujocoIntegrationSnapshot):
            raise TypeError(
                f"snapshot must be a MujocoIntegrationSnapshot, got {type(snapshot)!r}."
            )
        if snapshot.snapshot_semantics_version != STATE_SNAPSHOT_SEMANTICS_VERSION:
            raise ValueError("incompatible snapshot semantics version.")
        if snapshot.environment_id != self._env_id:
            raise ValueError(
                f"snapshot environment_id does not match: {snapshot.environment_id!r}."
            )
        if snapshot.gymnasium_version != gym.__version__:
            raise ValueError("snapshot Gymnasium version does not match.")
        if snapshot.mujoco_version != mujoco.__version__:
            raise ValueError("snapshot MuJoCo version does not match.")
        if snapshot.model_sha256 != self._model_sha256:
            raise ValueError("snapshot model_sha256 does not match the current base model.")
        if snapshot.state_spec != self._integration_state_spec:
            raise ValueError("snapshot state_spec is not mjSTATE_INTEGRATION.")
        if snapshot.integration_state.shape != (self._integration_state_size,):
            raise ValueError(
                "wrong snapshot integration_state shape: "
                f"expected {(self._integration_state_size,)}, "
                f"got {snapshot.integration_state.shape}."
            )
        if snapshot.flat_observation.shape != (self.observation_dim,):
            raise ValueError("wrong snapshot flat_observation shape.")
        if snapshot.observation_dim != self.observation_dim:
            raise ValueError("snapshot observation_dim does not match.")
        if snapshot.action_dim != self.action_dim:
            raise ValueError("snapshot action_dim does not match.")
        if not np.isfinite(snapshot.integration_state).all():
            raise FloatingPointError("snapshot integration_state is not finite.")
        if not np.isfinite(snapshot.flat_observation).all():
            raise FloatingPointError("snapshot flat_observation is not finite.")

        state_sha256 = _canonical_array_sha256(snapshot.integration_state)
        observation_sha256 = _canonical_array_sha256(snapshot.flat_observation)
        if state_sha256 != snapshot.state_sha256:
            raise ValueError("snapshot state_sha256 check failed.")
        if observation_sha256 != snapshot.observation_sha256:
            raise ValueError("snapshot observation_sha256 check failed.")

        metadata = self._snapshot_metadata_without_digest(
            state_sha256=snapshot.state_sha256,
            observation_sha256=snapshot.observation_sha256,
            elapsed_steps=snapshot.elapsed_steps,
            environment_rng_state=snapshot.environment_rng_state,
            action_space_rng_state=snapshot.action_space_rng_state,
            source_friction_scale=snapshot.source_friction_scale,
        )
        if _snapshot_sha256(metadata) != snapshot.snapshot_sha256:
            raise ValueError("snapshot snapshot_sha256 check failed.")

    def restore_integration_snapshot(
        self,
        snapshot: MujocoIntegrationSnapshot,
        parameters: Mapping[str, float] | None = None,
    ) -> ResetResult:
        self._validate_snapshot(snapshot)
        requested_parameters = (
            {"friction_scale": snapshot.source_friction_scale}
            if parameters is None
            else parameters
        )
        self._set_absolute_dynamics_without_forward(requested_parameters)

        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(
            self._model,
            self._data,
            snapshot.integration_state,
            state_spec,
        )

        readback = np.empty(self._integration_state_size, dtype=np.float64)
        mujoco.mj_getState(
            self._model,
            self._data,
            readback,
            state_spec,
        )
        if not np.array_equal(readback, snapshot.integration_state):
            raise RuntimeError("mjSTATE_INTEGRATION read back after mj_setState is not exact.")

        self._restore_elapsed_steps(snapshot.elapsed_steps)
        self._restore_rng_state(
            self._env.unwrapped.np_random,
            snapshot.environment_rng_state,
        )
        self._restore_rng_state(
            self._env.action_space.np_random,
            snapshot.action_space_rng_state,
        )

        mujoco.mj_forward(self._model, self._data)
        raw_observation = self._current_raw_observation()
        flat_observation = self._flatten_observation(raw_observation)
        if not np.array_equal(flat_observation, snapshot.flat_observation):
            maximum_error = float(
                np.max(np.abs(flat_observation - snapshot.flat_observation))
            )
            raise RuntimeError(
                "observation differs after the full state restore: "
                f"max_abs_error={maximum_error:.9g}."
            )

        self._last_flat_observation = flat_observation.copy()
        return ResetResult(
            raw_observation=raw_observation,
            flat_observation=flat_observation,
            info={
                "restored_snapshot_sha256": snapshot.snapshot_sha256,
                "restored_state_sha256": snapshot.state_sha256,
                "friction_scale": self._current_friction_scale,
            },
        )

    def get_contact_summary(self) -> dict[str, Any]:
        contact_count = int(self._data.ncon)
        contact_pairs: list[tuple[int, int]] = []
        floor_contact_count = 0
        distances: list[float] = []
        floor_normal_force_sum = 0.0
        floor_tangential_force_sum = 0.0
        floor_utilizations: list[float] = []
        floor_utilization_weights: list[float] = []
        floor_friction_evaluated_contact_count = 0

        for index in range(contact_count):
            contact = self._data.contact[index]
            geom_pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            contact_pairs.append(geom_pair)
            is_floor_contact = (
                self._floor_geom_id >= 0 and self._floor_geom_id in geom_pair
            )
            floor_contact_count += int(is_floor_contact)
            distances.append(float(contact.dist))
            if is_floor_contact:
                wrench = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self._model, self._data, index, wrench)
                adhesion = float(getattr(contact, "adhesion", 0.0))
                normal_force = float(wrench[0]) + adhesion
                contact_dim = int(contact.dim)
                friction_dimensions = min(max(contact_dim - 1, 0), 5)
                friction_forces = wrench[1 : 1 + friction_dimensions]
                tangential_force = float(np.linalg.norm(friction_forces[:2]))
                floor_normal_force_sum += max(normal_force, 0.0)
                floor_tangential_force_sum += tangential_force
                friction = np.asarray(contact.friction, dtype=np.float64)[
                    :friction_dimensions
                ]
                if (
                    int(contact.efc_address) >= 0
                    and normal_force > 0.0
                    and friction_dimensions > 0
                    and np.isfinite(normal_force)
                    and np.isfinite(friction_forces).all()
                    and np.isfinite(friction).all()
                    and np.all(friction > 0.0)
                ):
                    scaled = friction_forces / friction
                    if bool(mujoco.mj_isPyramidal(self._model)):
                        utilization = float(np.linalg.norm(scaled, ord=1))
                    else:
                        utilization = float(np.linalg.norm(scaled, ord=2))
                    utilization /= normal_force
                    if np.isfinite(utilization) and utilization >= 0.0:
                        floor_utilizations.append(utilization)
                        floor_utilization_weights.append(normal_force)
                        floor_friction_evaluated_contact_count += 1

        contact_pairs.sort()
        signature_payload = json.dumps(
            contact_pairs,
            separators=(",", ":"),
        ).encode("ascii")
        signature_sha256 = hashlib.sha256(signature_payload).hexdigest()
        minimum_distance = None if not distances else min(distances)
        utilization_max = max(floor_utilizations, default=0.0)
        if floor_utilization_weights:
            utilization_weighted_mean = float(
                np.average(
                    np.asarray(floor_utilizations, dtype=np.float64),
                    weights=np.asarray(floor_utilization_weights, dtype=np.float64),
                )
            )
        else:
            utilization_weighted_mean = 0.0

        return {
            "contact_count": contact_count,
            "floor_contact_count": floor_contact_count,
            "contact_pairs": tuple(contact_pairs),
            "signature_sha256": signature_sha256,
            "minimum_distance": minimum_distance,
            "maximum_penetration": (
                0.0 if minimum_distance is None else max(0.0, -minimum_distance)
            ),
            "floor_contact_normal_force_sum": floor_normal_force_sum,
            "floor_contact_tangential_force_sum": floor_tangential_force_sum,
            "floor_friction_cone_utilization_max": utilization_max,
            "floor_friction_cone_utilization_weighted_mean": (
                utilization_weighted_mean
            ),
            "floor_friction_evaluated_contact_count": (
                floor_friction_evaluated_contact_count
            ),
        }

    def get_dynamics_metadata(self) -> dict[str, Any]:
        floor_friction: list[float] | None

        if self._floor_geom_id >= 0:
            floor_friction = (
                self._model.geom_friction[self._floor_geom_id].copy().tolist()
            )
        else:
            floor_friction = None

        contact_geom_friction = self._model.geom_friction[self._contact_geom_mask]
        pair_count = int(self._model.npair)
        pair_friction = self._model.pair_friction[:pair_count]

        active_contact_count = int(self._data.ncon)
        active_contact_friction: np.ndarray | None

        if active_contact_count > 0:
            active_contact_friction = np.stack(
                [
                    np.asarray(self._data.contact[index].friction).copy()
                    for index in range(active_contact_count)
                ],
                axis=0,
            )
        else:
            active_contact_friction = None

        return {
            "dynamics_edit_semantics_version": (DYNAMICS_EDIT_SEMANTICS_VERSION),
            "friction_scope": FRICTION_SCOPE,
            "friction_scale": self._current_friction_scale,
            "floor_friction": floor_friction,
            "contact_geom_count": int(self._contact_geom_ids.size),
            "contact_geom_friction_min": (contact_geom_friction.min(axis=0).tolist()),
            "contact_geom_friction_max": (contact_geom_friction.max(axis=0).tolist()),
            "explicit_contact_pair_count": pair_count,
            "explicit_pair_friction_min": (
                None if pair_count == 0 else pair_friction.min(axis=0).tolist()
            ),
            "explicit_pair_friction_max": (
                None if pair_count == 0 else pair_friction.max(axis=0).tolist()
            ),
            "active_contact_count": active_contact_count,
            "active_contact_friction_min": (
                None
                if active_contact_friction is None
                else active_contact_friction.min(axis=0).tolist()
            ),
            "active_contact_friction_max": (
                None
                if active_contact_friction is None
                else active_contact_friction.max(axis=0).tolist()
            ),
        }

    def get_backend_metadata(self) -> dict[str, Any]:
        return {
            "backend": "gymnasium",
            "environment_id": self._env_id,
            "gymnasium_version": gym.__version__,
            "mujoco_version": mujoco.__version__,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "action_low": self._action_low.tolist(),
            "action_high": self._action_high.tolist(),
            "max_episode_steps": self._max_episode_steps,
            "dynamics_edit_semantics_version": (DYNAMICS_EDIT_SEMANTICS_VERSION),
            "friction_scope": FRICTION_SCOPE,
            "contact_geom_count": int(self._contact_geom_ids.size),
            "explicit_contact_pair_count": int(self._model.npair),
            "supports_full_state_snapshot": True,
            "state_snapshot_semantics_version": STATE_SNAPSHOT_SEMANTICS_VERSION,
            "integration_state_spec": self._integration_state_spec,
            "integration_state_size": self._integration_state_size,
            "model_sha256": self._model_sha256,
        }

    def close(self) -> None:
        self._env.close()
