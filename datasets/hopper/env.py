from __future__ import annotations

import hashlib
import os

import gymnasium
import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

MODEL_XML = os.path.join(os.path.dirname(os.path.abspath(gymnasium.__file__)), "envs", "mujoco", "assets", "hopper.xml")
MODEL_SHA256 = "3ce93a055ffdcd83c0c701d2400768e40d2cbb9532f3c4ae33377c27f8b39f9e"
OBS_DIM = 11
ACT_DIM = 3
SERVO_KP = 800.0
SERVO_KV = 60.0
SERVO_FORCE = 200.0
HOME = np.array([-0.1, -0.1, 0.17], dtype=np.float64)


def check_model_file():
    with open(MODEL_XML, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != MODEL_SHA256:
        raise RuntimeError(f"hopper.xml differs from the Gymnasium 1.3.0 file: {got}")
    return got


class ServoHopperEnv(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

    def __init__(self, frame_skip: int = 25, reset_noise_scale: float = 5e-3, servo_kp: float = SERVO_KP,
                 servo_kv: float = SERVO_KV, home=tuple(HOME), cone: str = "elliptic", **kwargs):
        utils.EzPickle.__init__(self, frame_skip, reset_noise_scale, servo_kp, servo_kv, home, cone, **kwargs)
        if cone not in ("elliptic", "pyramidal"):
            raise ValueError(cone)
        self._cone = cone
        check_model_file()
        self._reset_noise_scale = float(reset_noise_scale)
        self._servo = (float(servo_kp), float(servo_kv))
        self._home = np.asarray(home, dtype=np.float64).copy()
        obs_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64)
        MujocoEnv.__init__(self, MODEL_XML, int(frame_skip), observation_space=obs_space, **kwargs)
        self.metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
                         "render_fps": int(np.round(1.0 / self.dt))}
        self.home_ctrl = self._home.copy()

    def _initialize_simulation(self):
        model, data = super()._initialize_simulation()
        convert_to_servos(model, *self._servo)
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if self._cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
        return model, data

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        return self._get_obs(), 0.0, False, False, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat[1:], self.data.qvel.flat]).astype(np.float64)

    def reset_model(self):
        c = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        self.data.ctrl[:] = self.home_ctrl
        self.set_state(qpos, qvel)
        return self._get_obs()


def convert_to_servos(model, kp=SERVO_KP, kv=SERVO_KV):
    if model.nu != ACT_DIM:
        raise RuntimeError("Hopper model must have 3 actuators")
    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0
    model.actuator_gainprm[:, 0] = kp
    model.actuator_biasprm[:, 1] = -kp
    model.actuator_biasprm[:, 2] = -kv
    model.actuator_gear[:] = 0.0
    model.actuator_gear[:, 0] = 1.0
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:] = [-SERVO_FORCE, SERVO_FORCE]
    model.actuator_ctrllimited[:] = 1
    for i in range(model.nu):
        j = int(model.actuator_trnid[i, 0])
        if not model.jnt_limited[j]:
            raise RuntimeError("servoed joint must be limited")
        model.actuator_ctrlrange[i] = model.jnt_range[j]
    return model
