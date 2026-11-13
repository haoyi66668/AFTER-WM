from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from baselines import seq_utils as _G
from afterwm.core import METRIC_NAMES
from pipeline import data as Dm
from pipeline import registry as P
from pipeline.eval_api import Forecast, QueryBatch

NAME = "mlp"
FAMILY = "baseline"
K_GRID = (1, 5, 10)
K_NATIVE = 10
N_HIDDEN_LAYERS = 4

_O = P.OBS_DIM
_A = P.ACT_DIM

loss_from_pred = _G.loss_from_pred
fixed_chunk_forecast = _G.fixed_chunk_forecast


def _lin(i: int, o: int) -> int:
    return i * o + o


def param_count(cfg: dict) -> int:
    K, W, L = int(cfg["history_k"]), int(cfg["hidden"]), int(cfg["n_hidden_layers"])
    d_in = K * (_O + _A)
    return _lin(d_in, W) + (L - 1) * _lin(W, W) + _lin(W, _O)


def solve_hidden(cfg: dict) -> int:
    target = int(cfg["budget_target"])
    best = None
    for w in range(8, 4096):
        n = param_count(dict(cfg, hidden=w))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (w, n)
    return best[0]


def default_config(budget_class: str = "baseline", **variant) -> dict:
    if budget_class != "baseline":
        raise ValueError(f"{NAME}: only budget_class 'baseline' is defined, got {budget_class!r}")
    cfg = dict(
        name=NAME, family=FAMILY, budget_class=budget_class, budget_target=P.BUDGET_BASELINE,
        history_k=K_NATIVE,
        hidden=None,
        n_hidden_layers=N_HIDDEN_LAYERS,
        activation="SiLU",
        lr=3e-4, optimizer="Adam", adam_betas=[0.9, 0.999], adam_eps=1e-8,
        grad_clip=1.0,
        loss="afterwm.core.strict_loss(K=1, route_log_p=0) via baselines.seq_utils.loss_from_pred",
        eval_chunk=240,
    )
    cfg.update(variant)
    k = cfg["history_k"]
    if isinstance(k, bool) or int(k) != k or int(k) not in K_GRID:
        raise ValueError(f"{NAME}: history_k must be one of {K_GRID}, got {k!r}")
    cfg["history_k"] = int(k)
    if int(cfg["n_hidden_layers"]) != N_HIDDEN_LAYERS or cfg["activation"] != "SiLU":
        raise ValueError(f"{NAME}: the network has {N_HIDDEN_LAYERS} SiLU hidden layers")
    if cfg["hidden"] is None:
        cfg["hidden"] = solve_hidden(cfg)
    cfg["hidden"] = int(cfg["hidden"])
    cfg["n_hidden_layers"] = int(cfg["n_hidden_layers"])
    cfg["param_count_formula"] = param_count(cfg)
    return cfg


class MLPSeq(nn.Module):
    GROUPS = {"mlp": ("mlp",)}

    def __init__(self, cfg: dict, obs_std):
        super().__init__()
        K, W, L = int(cfg["history_k"]), int(cfg["hidden"]), int(cfg["n_hidden_layers"])
        self.K, self.W = K, W
        layers, d = [], K * (_O + _A)
        for _ in range(L):
            layers += [nn.Linear(d, W), nn.SiLU()]
            d = W
        layers.append(nn.Linear(W, _O))
        self.mlp = nn.Sequential(*layers)
        std = np.asarray(obs_std, dtype=np.float64)
        if std.shape != (_O,):
            raise ValueError(f"obs_std must have shape ({_O},), got {std.shape}")
        self.register_buffer("loss_own_std", torch.as_tensor(std.copy()), persistent=False)
        self.register_buffer("loss_common_std", torch.as_tensor(std.copy()), persistent=False)
        self.raw_scale = float(np.mean(std ** 2))

    def rollout(self, ctx_obs, ctx_act, fut_act):
        K = self.K
        n_ctx = ctx_obs.shape[1]
        zs = [ctx_obs[:, n_ctx - K + i] for i in range(K)]
        n_a = ctx_act.shape[1]
        acts = [ctx_act[:, n_a - (K - 1) + i] for i in range(K - 1)]
        outs = []
        for k in range(fut_act.shape[1]):
            acts_k = acts + [fut_act[:, k]]
            x = torch.cat([v for pair in zip(zs, acts_k) for v in pair], -1)
            z = zs[-1] + self.mlp(x)
            outs.append(z)
            zs = zs[1:] + [z]
            acts = acts_k[1:]
        return torch.stack(outs, 1)


def build(cfg: dict, seed: int, device: str, stats=None) -> nn.Module:
    if stats is None:
        stats = Dm.train_stats()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = MLPSeq(cfg, stats.obs_std)
    n = sum(p.numel() for p in model.parameters())
    assert n == param_count(cfg), (n, param_count(cfg))
    return model.to(device)


def make_optimizer(model, cfg):
    b1, b2 = (float(x) for x in cfg["adam_betas"])
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(b1, b2), eps=float(cfg["adam_eps"]),
                            weight_decay=0.0)


def train_step(model, batch: dict, step: int, cfg: dict):
    model.train()
    pred = model.rollout(batch["ctx_obs"], batch["ctx_act"], batch["fut_act"])
    loss, met = loss_from_pred(model, pred, batch["fut_obs"])
    vals = met.detach().double().cpu().tolist()
    return loss, dict(zip(METRIC_NAMES, vals))


class _Forecaster:
    family = FAMILY
    budget_class = "baseline"
    needs_grad_at_eval = False
    stochastic = False

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.name = NAME

    @torch.no_grad()
    def forecast(self, q: QueryBatch) -> Forecast:
        dev = next(self.model.parameters()).device
        mean = fixed_chunk_forecast(self.model.rollout, q, int(self.cfg.get("eval_chunk", 240)), dev)
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
              for g, names in MLPSeq.GROUPS.items()}
    total = trainable + frozen
    assert sum(groups.values()) == total, (groups, total)
    target = cfg["budget_target"]
    return {"total": total, "trainable": trainable, "frozen": frozen, "stored_nongrad": 0,
            "buffers": buffers, "groups": groups, "budget_class": cfg["budget_class"], "budget_target": target,
            "within_budget": abs(total - target) <= P.BUDGET_TOL * target,
            "state": {"history_k": cfg["history_k"], "hidden": cfg["hidden"]},
            "note": "buffers = strict-loss frame scales (2 x OBS_DIM TRAIN obs std, float64, non-persistent; not counted)"}
