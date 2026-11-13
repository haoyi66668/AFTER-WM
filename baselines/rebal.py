from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline import data as Dm
from pipeline import registry as P
from pipeline.eval_api import Forecast, QueryBatch

NAME = "rebal"
FAMILY = "baseline"

_O = P.OBS_DIM
_A = P.ACT_DIM


def _lin(i: int, o: int) -> int:
    return i * o + o


def param_count(cfg: dict) -> int:
    H, Hc = cfg["hidden"], cfg["cell_hidden"] or cfg["hidden"]
    cell = _lin(H + _O + _A, Hc) + 2 * Hc + _lin(Hc, H) + 2 * H
    head = _lin(H, _O)
    return cell + head


def default_config(budget_class: str = "baseline", **variant) -> dict:
    if budget_class != "baseline":
        raise ValueError(f"{NAME}: only budget_class 'baseline' is defined, got {budget_class!r}")
    cfg = dict(
        name=NAME, family=FAMILY, budget_class=budget_class, budget_target=P.BUDGET_BASELINE,
        hidden=None,
        cell_hidden=None,
        nonlinearity="tanh",
        lr=1e-2, optimizer="Adam", adam_eps=1e-8,
        grad_clip=None,
        delta_std_eps=1e-10,
        eval_chunk=240,
    )
    cfg.update(variant)
    if cfg["hidden"] is None:
        target = cfg["budget_target"]
        best = None
        for h in range(8, 4096):
            n = param_count(dict(cfg, hidden=h))
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (h, n)
        cfg["hidden"] = best[0]
    cfg["param_count_formula"] = param_count(cfg)
    return cfg


def _glorot_linear(i: int, o: int) -> nn.Linear:
    lin = nn.Linear(i, o)
    with torch.no_grad():
        nn.init.xavier_uniform_(lin.weight)
        lin.bias.zero_()
    return lin


class ReBALMLP(nn.Module):
    GROUPS = {"cell": ("inp", "ln_c", "out", "ln_h"), "head": ("head",)}

    def __init__(self, cfg: dict, delta_mean: np.ndarray, delta_std: np.ndarray):
        super().__init__()
        H = cfg["hidden"]
        Hc = cfg["cell_hidden"] or H
        self.H = H
        self.inp = _glorot_linear(H + _O + _A, Hc)
        self.ln_c = nn.LayerNorm(Hc)
        self.out = _glorot_linear(Hc, H)
        self.ln_h = nn.LayerNorm(H)
        self.head = _glorot_linear(H, _O)
        self.register_buffer("delta_mean", torch.as_tensor(np.asarray(delta_mean, np.float32)))
        self.register_buffer("delta_std", torch.as_tensor(np.asarray(delta_std, np.float32)))
        self.std_eps = float(cfg["delta_std_eps"])

    def proj_za(self, z, a):
        w = self.inp.weight
        return F.linear(torch.cat([z, a], -1), w[:, self.H:], self.inp.bias)

    def w_h_t(self):
        return self.inp.weight[:, : self.H].t()

    def cell(self, h, u, w_h_t):
        x = torch.tanh(self.ln_c(torch.addmm(u, h, w_h_t)))
        return self.ln_h(h + self.out(x))

    def denorm_delta(self, d):
        return d * (self.delta_std + self.std_eps) + self.delta_mean

    def norm_delta(self, d):
        return (d - self.delta_mean) / (self.delta_std + self.std_eps)

    def teacher_forced(self, z, a, h=None):
        B, T, _ = z.shape
        u = self.proj_za(z, a)
        wt = self.w_h_t()
        if h is None:
            h = z.new_zeros(B, self.H)
        hs = []
        for j in range(T):
            h = self.cell(h, u[:, j], wt)
            hs.append(h)
        return torch.stack(hs, 1), h


def _train_delta_stats(stats):
    obs, _ = Dm.load_visible("train")
    z = stats.norm_obs(obs).astype(np.float64)
    d = (z[:, 1:] - z[:, :-1]).reshape(-1, _O)
    return d.mean(0), d.std(0)


def build(cfg: dict, seed: int, device: str, stats) -> nn.Module:
    if stats is None:
        stats = Dm.train_stats()
    dmean, dstd = _train_delta_stats(stats)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = ReBALMLP(cfg, dmean, dstd)
    n = sum(p.numel() for p in model.parameters())
    assert n == param_count(cfg), (n, param_count(cfg))
    return model.to(device)


def make_optimizer(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=cfg["adam_eps"])


def _floats(**ts) -> dict:
    keys = list(ts)
    vals = torch.stack([ts[k].detach().float().reshape(()) for k in keys]).cpu().tolist()
    return dict(zip(keys, vals))


def fixed_chunk_forecast(fn, q: QueryBatch, chunk: int, dev) -> torch.Tensor:
    tens = [q.ctx_obs.to(dev, torch.float32), q.ctx_act.to(dev, torch.float32), q.fut_act.to(dev, torch.float32)]
    n_all = tens[0].shape[0]
    outs = []
    for s in range(0, n_all, chunk):
        part = [x[s:s + chunk] for x in tens]
        n = part[0].shape[0]
        if n < chunk:
            part = [torch.cat([x, x[-1:].expand(chunk - n, *x.shape[1:])], 0) for x in part]
        outs.append(fn(*part)[:n])
    return torch.cat(outs, 0)


def train_step(model, batch: dict, step: int, cfg: dict):
    model.train()
    obs = batch["win_obs"]
    act = batch["win_act"]
    T = act.shape[1]
    hs, _ = model.teacher_forced(obs[:, :T], act)
    pred = model.head(hs)
    target = model.norm_delta(obs[:, 1:T + 1] - obs[:, :T])
    loss = F.mse_loss(pred, target)
    with torch.no_grad():
        mse_ctx = F.mse_loss(pred[:, : P.CONTEXT], target[:, : P.CONTEXT])
        mse_fut = F.mse_loss(pred[:, P.CONTEXT:], target[:, P.CONTEXT:])
    return loss, _floats(loss=loss, mse_ctx=mse_ctx, mse_fut=mse_fut)


class _Forecaster:
    family = FAMILY
    budget_class = "baseline"
    needs_grad_at_eval = False
    stochastic = False

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.name = NAME

    def _mean(self, ctx_obs, ctx_act, fut_act):
        m = self.model
        _, h = m.teacher_forced(ctx_obs[:, :-1], ctx_act)
        z = ctx_obs[:, -1]
        wt = m.w_h_t()
        outs = []
        for k in range(fut_act.shape[1]):
            h = m.cell(h, m.proj_za(z, fut_act[:, k]), wt)
            z = z + m.denorm_delta(m.head(h))
            outs.append(z)
        return torch.stack(outs, 1)

    @torch.no_grad()
    def forecast(self, q: QueryBatch) -> Forecast:
        dev = next(self.model.parameters()).device
        mean = fixed_chunk_forecast(self._mean, q, int(self.cfg.get("eval_chunk", 240)), dev)
        return Forecast(mean=mean, comp_mean=mean[:, :, None, :],
                        comp_logw=torch.zeros(mean.shape[0], 1, device=dev, dtype=mean.dtype))

    def parameter_report(self) -> dict:
        return parameter_report(self.model, self.cfg)


def forecaster(model, cfg):
    return _Forecaster(model, cfg)


def parameter_report(model, cfg) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    groups = {g: sum(p.numel() for n in names for p in getattr(model, n).parameters())
              for g, names in ReBALMLP.GROUPS.items()}
    total = trainable + frozen
    assert sum(groups.values()) == total, (groups, total)
    target = cfg["budget_target"]
    return {"total": total, "trainable": trainable, "frozen": frozen, "stored_nongrad": 0,
            "buffers": buffers, "groups": groups, "budget_class": cfg["budget_class"], "budget_target": target,
            "within_budget": abs(total - target) <= P.BUDGET_TOL * target,
            "state": {"h": cfg["hidden"]},
            "note": "buffers = TRAIN delta mean/std (34 normaliser statistics, not counted in total)"}
