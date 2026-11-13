from __future__ import annotations

import numpy as np

LOG_SCALE_LIMIT = 3.0
NON_STATIONARY_PERIOD = 100
NON_STATIONARY_INTERVAL = 10
TASK_NUM = 40
TEST_TASK_NUM = 40
LIST_LEN = 16
RAND_PARAMS = ("body_mass", "dof_damping_1_dim", "gravity")
TORSO_BODY = 1


def sample_tasks(rs: np.random.RandomState, n_tasks: int, init_gravity, log_scale_limit: float = LOG_SCALE_LIMIT):
    L = float(log_scale_limit)
    bound = lambda shape: np.array(1.5) ** rs.uniform(-L, L, size=shape)
    g0 = np.asarray(init_gravity, np.float64)
    out = []
    for _ in range(int(n_tasks)):
        new = {}
        new["body_mass"] = bound((1,))
        new["dof_damping_1_dim"] = bound((1,))
        new["gravity"] = np.multiply(g0, bound(g0.shape))
        out.append(new)
    return out


def make_pools(seed: int, init_gravity, log_scale_limit: float = LOG_SCALE_LIMIT):
    rs = np.random.RandomState(int(seed))
    train = sample_tasks(rs, TASK_NUM, init_gravity, log_scale_limit)
    test = sample_tasks(rs, TEST_TASK_NUM, init_gravity, log_scale_limit)
    return train, test


def episode_task_list(rng: np.random.Generator, n_pool: int):
    return [int(x) for x in rng.integers(0, int(n_pool), size=LIST_LEN)]


class Schedule:

    def __init__(self, period: int = NON_STATIONARY_PERIOD, interval: int = NON_STATIONARY_INTERVAL):
        self.period, self.interval = int(period), int(interval)
        self.cur_step_ind = 0

    def before_transition(self):
        self.cur_step_ind += 1
        if self.cur_step_ind % self.interval == 0:
            weight_origin = self.cur_step_ind / self.period
            return int(weight_origin)
        return None


def list_index_of_transition(j: int) -> int:
    return (int(j) + 1) // NON_STATIONARY_PERIOD


class Applier:

    def __init__(self, model):
        self.model = model
        self.init_mass = np.array(model.body_mass).copy()
        self.init_damping = np.array(model.dof_damping).copy()
        self.init_gravity = np.array(model.opt.gravity).copy()
        if model.body(TORSO_BODY).name != "torso":
            raise RuntimeError("body 1 is not the torso")

    def set_task(self, task):
        m = self.model
        m.body_mass[:] = self.init_mass
        m.body_mass[TORSO_BODY] = self.init_mass[TORSO_BODY] * float(task["body_mass"][0])
        m.dof_damping[:] = self.init_damping * float(task["dof_damping_1_dim"][0])
        m.opt.gravity[:] = np.asarray(task["gravity"], np.float64)

    def current(self):
        m = self.model
        return {"torso_mass": float(m.body_mass[TORSO_BODY]), "dof_damping": np.array(m.dof_damping).copy(),
                "gravity": np.array(m.opt.gravity).copy()}
