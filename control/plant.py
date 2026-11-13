from __future__ import annotations

import numpy as np

from control import spec as SP

NOMINAL = {"friction_scale": 1.0, "gain_scale": 1.0}


class Plant:
    def __init__(self, namespace: str):
        from jshc import config as JC
        from simcore import gen as G
        self.protocol = JC.make_protocol(namespace)
        if not JC.shared_keys_ok(self.protocol):
            raise RuntimeError("JSHC protocol does not share the simulator keys")
        self.ad, self.mujoco = G.make_adapter(self.protocol)
        self.model, self.data = self.ad._model, self.ad._data
        self.frame_skip = int(self.ad._env.unwrapped.frame_skip)
        self.nq = int(self.model.nq)

    @staticmethod
    def masked(a, j):
        out = np.array(a, dtype=np.float32, copy=True)
        if int(j) >= 0:
            out[..., int(j)] = np.float32(0.0)
        return out

    def env_action(self, a):
        a = np.clip(np.asarray(a, np.float32), -1.0, 1.0)
        lo, hi = self.ad._action_low, self.ad._action_high
        return (lo + 0.5 * (a + 1.0) * (hi - lo)).astype(np.float32)

    def step(self, a_cmd, j):
        self.ad.apply_absolute_dynamics(dict(NOMINAL))
        applied = self.masked(a_cmd, j)
        res = self.ad.step_with_impulse(applied, None)
        if not np.array_equal(res.model_action, applied):
            raise ValueError("executed action differs from the masked command (clipping is not allowed)")
        if not np.isfinite(res.flat_observation).all():
            raise FloatingPointError("non-finite observation")
        return res, applied

    def restore_anchor(self, snap):
        self.ad.restore_integration_snapshot(snap, parameters=dict(NOMINAL))


def reward(x_velocity, applied):
    return float(x_velocity) - SP.CTRL_COST * float(np.sum(np.square(np.asarray(applied, np.float64))))
