from __future__ import annotations

import hashlib

import torch

from control import spec as SP

POPSIZE, ELITES, ITERS, ALPHA, HORIZON = 500, 50, 5, 0.1, 30


def planner_config() -> dict:
    return {"method": "CEM (PETS)", "popsize": POPSIZE, "elites": ELITES, "iterations": ITERS, "alpha": ALPHA,
            "horizon": HORIZON, "init_var": "(ub-lb)^2/16", "sampling": "truncated normal [-2,2] x constrained std",
            "warm_start": "shift by one step, append box centre", "execute": "first action of final mean, replan every step",
            "box": SP.BOX, "cost": "sum_h (-v_hat_h + 0.05 ||a_h||^2), v_hat = predicted obs[8]", "ctrl_cost": SP.CTRL_COST}


def noise_seed(split: str, planner_seed: int, episode_index: int) -> int:
    h = hashlib.sha256(f"mpcjshc-cem|{split}|{int(planner_seed)}|{int(episode_index)}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def run_cost(vel, samples):
    return (-vel + SP.CTRL_COST * samples.square().sum(-1)).sum(-1)


class CEM:
    def __init__(self, act_dim=6, device="cpu"):
        self.device = device
        self.H, self.A = HORIZON, int(act_dim)
        self.ub = torch.full((self.A,), SP.BOX, dtype=torch.float32, device=device)
        self.lb = -self.ub
        self.init_var = ((self.ub - self.lb) ** 2 / 16.0).expand(self.H, self.A).clone()
        self.centre = (self.ub + self.lb) / 2.0

    def initial_mean(self):
        return self.centre.expand(self.H, self.A).clone()

    def shift(self, sol):
        return torch.cat([sol[1:], self.centre[None, :]], 0)

    def draw(self, gen: torch.Generator):
        eps = torch.empty(ITERS, POPSIZE, self.H, self.A, dtype=torch.float32)
        torch.nn.init.trunc_normal_(eps, 0.0, 1.0, -2.0, 2.0, generator=gen)
        return eps.to(self.device)

    def plan(self, cost_fn, mean0, gen: torch.Generator):
        eps_all = self.draw(gen)
        mean, best, mstd = self.plan_tensors(cost_fn, mean0.to(self.device), eps_all)
        return mean, {"best_cost_last_iter": float(best), "max_std": float(mstd)}

    def plan_tensors(self, cost_fn, mean0, eps_all):
        mean = mean0.clone()
        var = self.init_var.clone()
        best = None
        for i in range(ITERS):
            cvar = torch.minimum(torch.minimum(((mean - self.lb) / 2) ** 2, ((self.ub - mean) / 2) ** 2), var)
            samples = torch.clamp(mean + eps_all[i] * torch.sqrt(cvar), self.lb, self.ub)
            costs = cost_fn(samples).float()
            costs = torch.nan_to_num(costs, nan=1e9, posinf=1e9, neginf=1e9)
            vals, idx = torch.topk(costs, ELITES, largest=False)
            el = samples[idx]
            new_mean, new_var = el.mean(0), el.var(0, unbiased=False)
            mean = ALPHA * mean + (1 - ALPHA) * new_mean
            var = ALPHA * var + (1 - ALPHA) * new_var
            best = vals[0]
        return mean, best, var.sqrt().max()


class GraphedPlan:

    def __init__(self, cem, plan, am, asd, om_v, osd_v, enc_example, warmup=2):
        self.cem, self.plan = cem, plan
        dev = cem.device
        self.s_enc = tuple(x.detach().clone() for x in enc_example)
        self.s_eps = torch.zeros(ITERS, POPSIZE, cem.H, cem.A, device=dev)
        self.s_mean0 = cem.initial_mean().to(dev)
        self.am, self.asd, self.om_v, self.osd_v = am, asd, om_v, osd_v
        self.status = "eager"
        self.graph = None
        try:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(warmup):
                    self._body()
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.s_out = self._body()
            torch.cuda.synchronize()
            self.graph = g
            self.status = "graph"
        except Exception as exc:
            self.status = f"eager (capture failed: {type(exc).__name__}: {str(exc)[:200]})"
            torch.cuda.synchronize()

    def cost_fn(self, samples, enc=None):
        pred = self.plan.predict(self.s_enc if enc is None else enc, (samples - self.am) / self.asd)
        vel = pred[..., SP.VEL_INDEX] * self.osd_v + self.om_v
        return run_cost(vel, samples)

    def _body(self):
        return self.cem.plan_tensors(self.cost_fn, self.s_mean0, self.s_eps)

    def __call__(self, enc, eps_all, mean0):
        for s, x in zip(self.s_enc, enc):
            s.copy_(x)
        self.s_eps.copy_(eps_all)
        self.s_mean0.copy_(mean0)
        if self.graph is None:
            out = self._body()
        else:
            self.graph.replay()
            out = self.s_out
        mean, best, mstd = out
        return mean.clone(), {"best_cost_last_iter": float(best), "max_std": float(mstd)}
