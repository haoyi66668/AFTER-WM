from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, grad_and_value, vmap

from pipeline import paramcount as PC
from pipeline import registry as P
from pipeline.data import TrainStats, load_visible
from pipeline.eval_api import Forecast, QueryBatch

NAME = "grbal"
FAMILY = "baseline"
DIN = P.OBS_DIM + P.ACT_DIM
DOUT = P.OBS_DIM

DISCLOSURES = [
    "D1 offline meta-training on logged TRAIN with the shared schedule; no MPC/policy/reward/data collection",
    "D2 4 tasks of 32 consecutive transitions per schedule window (offsets 0/24/48/72; support 16, query 16) -> 128 tasks per outer update (official meta-batch 10, random starts)",
    "D3 width solved for the 700,000 +-1% parameter budget (official 512); Delta-normaliser constants are uncounted buffers",
    "D4 shared TRAIN normaliser for inputs (same data/formula as GrBAL's own obs/act normalisation); Delta stats over all TRAIN transitions in that frame",
    "D5 number of outer updates, learning-rate schedule and VAL interval set by the shared protocol; best VAL checkpoint by mean prefix nMSE H1/5/10/20/40 (official: rolling-average early stopping on a 10% split)",
    "D6 no native gradient clipping (as official); the shared training protocol clips the global gradient norm at 1.0",
    "D7 single adaptation at the forecast origin on the last 16 context transitions; 40-step open-loop rollout with theta' (official re-adapts before each env step)",
    "D8 inner lr 0.001 default (official); 1e-4, 1e-2 or 1e-1 when selected on VAL",
    "D9 PyTorch fp32 + torch.func; Adam (0.9, 0.999, 1e-8)",
    "D10 fixed-size padded evaluation chunks -> bitwise batch/order-invariant per-query outputs",
]


def mlp_param_count(hidden: int, n_hidden: int = 3, din: int = DIN, dout: int = DOUT) -> int:
    return (din * hidden + hidden) + (n_hidden - 1) * (hidden * hidden + hidden) + (hidden * dout + dout)


def solve_hidden(target: int, extra: int = 0, n_hidden: int = 3) -> int:
    return min(range(8, 4097), key=lambda h: (abs(mlp_param_count(h, n_hidden) + extra - target), h))


ALLOWED_INNER_LR = (1e-4, 1e-3, 1e-2, 1e-1)


def apply_variant(cfg: dict, variant: dict) -> dict:
    for k, v in variant.items():
        if k not in cfg and not k.startswith("dev_"):
            raise KeyError(f"unknown config key {k!r}")
        cfg[k] = v
    return cfg


def default_config(budget_class: str = "baseline", **variant) -> dict:
    if budget_class != "baseline":
        raise ValueError(f"{NAME} is a compared model; budget_class must be 'baseline', got {budget_class!r}")
    cfg = dict(
        model=NAME,
        budget_class=budget_class,
        n_hidden=3,
        hidden=solve_hidden(P.BUDGET_BASELINE),
        lr=1e-3,
        adam_betas=[0.9, 0.999],
        adam_eps=1e-8,
        grad_clip=None,
        inner_lr=1e-3,
        adapt_m=16,
        task_len=32,
        task_offsets=[0, 24, 48, 72],
        eval_chunk=240,
    )
    cfg = apply_variant(cfg, variant)
    if not cfg.get("dev_allow_offbudget") and float(cfg["inner_lr"]) not in ALLOWED_INNER_LR:
        raise ValueError(f"inner_lr must be one of {ALLOWED_INNER_LR} (protocol), got {cfg['inner_lr']}")
    return cfg


class MetaMLP(nn.Module):

    def __init__(self, hidden: int, n_hidden: int = 3, with_sigma2: bool = False):
        super().__init__()
        layers, d = [], DIN
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, DOUT))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.register_buffer("delta_mean", torch.zeros(DOUT))
        self.register_buffer("delta_std", torch.ones(DOUT))
        if with_sigma2:
            self.register_buffer("sigma2", torch.zeros(DOUT))

    def theta(self, detach: bool = False) -> dict:
        return {k: (v.detach() if detach else v) for k, v in self.net.named_parameters()}


_DELTA_CACHE: dict = {}


def train_delta_stats(stats: TrainStats):
    key = stats.sha256()
    if key not in _DELTA_CACHE:
        obs, _ = load_visible("train")
        d = np.diff(stats.norm_obs(obs).astype(np.float64), axis=1).reshape(-1, DOUT)
        _DELTA_CACHE[key] = (d.mean(0).astype(np.float32), np.maximum(d.std(0), 1e-6).astype(np.float32))
    return _DELTA_CACHE[key]


def _build(cfg: dict, seed: int, device, stats: TrainStats, with_sigma2: bool, budget_extra: int = 0) -> MetaMLP:
    torch.manual_seed(int(seed))
    model = MetaMLP(int(cfg["hidden"]), int(cfg["n_hidden"]), with_sigma2=with_sigma2)
    dm, ds = train_delta_stats(stats)
    with torch.no_grad():
        model.delta_mean.copy_(torch.from_numpy(dm))
        model.delta_std.copy_(torch.from_numpy(ds))
    if not cfg.get("dev_allow_offbudget"):
        chk = PC.budget_check(sum(p.numel() for p in model.parameters()) + budget_extra, "baseline")
        if not chk["within_budget"]:
            raise RuntimeError(f"parameter budget violated: {chk}")
    return model.to(device)


def build(cfg: dict, seed: int, device, stats: TrainStats) -> MetaMLP:
    return _build(cfg, seed, device, stats, with_sigma2=False)


def make_optimizer(model: MetaMLP, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.net.parameters(), lr=float(cfg["lr"]), betas=tuple(cfg["adam_betas"]),
                            eps=float(cfg["adam_eps"]))


def report(model: MetaMLP, per_query_copies: int, stored_nongrad: int, note: str) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = {n: p.numel() for n, p in model.named_parameters()}
    bufs = {n: b.numel() for n, b in model.named_buffers()}
    return PC.make_report(model, stored_nongrad=stored_nongrad, groups=groups, extra={
        "buffer_groups": bufs,
        "buffers_note": note,
        "budget": PC.budget_check(trainable + stored_nongrad, "baseline"),
        "per_query_state": {"fast_weight_copies": per_query_copies,
                            "fast_weight_values": per_query_copies * trainable,
                            "note": "functional per-query fast weights (state, not counted)"},
    })


def parameter_report(model: MetaMLP) -> dict:
    return report(model, 1, 0, "delta_mean/delta_std are normaliser constants (not counted)")


def _mse(net: nn.Module, p: dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r = functional_call(net, p, (x,)) - y
    return (r * r).mean()


def transitions(obs: torch.Tensor, act: torch.Tensor, delta_mean: torch.Tensor, delta_std: torch.Tensor):
    x = torch.cat([obs[:, :-1], act], dim=-1)
    y = (obs[:, 1:] - obs[:, :-1] - delta_mean) / delta_std
    return x, y


def make_tasks(win_obs, win_act, delta_mean, delta_std, offsets, task_len: int, m: int):
    x, y = transitions(win_obs, win_act, delta_mean, delta_std)
    n = x.shape[1]
    for o in offsets:
        if o < 0 or o + task_len > n:
            raise ValueError(f"task offset {o} (+{task_len}) outside window of {n} transitions")
    xs = torch.stack([x[:, o:o + task_len] for o in offsets], 1).flatten(0, 1)
    ys = torch.stack([y[:, o:o + task_len] for o in offsets], 1).flatten(0, 1)
    return xs[:, :m], ys[:, :m], xs[:, m:], ys[:, m:]


def maml_outer_loss(model: MetaMLP, sx, sy, qx, qy, inner_lr: float):
    net, p = model.net, model.theta()

    def per_task(sx_, sy_, qx_, qy_):
        g, pre = grad_and_value(lambda pp: _mse(net, pp, sx_, sy_))(p)
        p1 = {k: p[k] - inner_lr * g[k] for k in p}
        return _mse(net, p1, qx_, qy_), pre

    post, pre = vmap(per_task)(sx, sy, qx, qy)
    return post.mean(), pre.mean()


def train_step(model: MetaMLP, batch: dict, step: int, cfg: dict):
    sx, sy, qx, qy = make_tasks(batch["win_obs"], batch["win_act"], model.delta_mean, model.delta_std,
                                cfg["task_offsets"], int(cfg["task_len"]), int(cfg["adapt_m"]))
    post, pre_s = maml_outer_loss(model, sx, sy, qx, qy, float(cfg["inner_lr"]))
    with torch.no_grad():
        pre_q = _mse(model.net, model.theta(detach=True), qx, qy)
    return post, {"post_query_mse": post.detach(), "pre_query_mse": pre_q, "pre_support_mse": pre_s.detach(),
                  "n_tasks": float(sx.shape[0])}


def adapt_k_shot(net: nn.Module, p0: dict, x: torch.Tensor, y: torch.Tensor, lr: float) -> dict:
    g = vmap(grad(lambda pp, xx, yy: _mse(net, pp, xx, yy)), in_dims=(None, 0, 0))(p0, x, y)
    return {k: p0[k] - lr * g[k] for k in p0}


def batched_apply(net: nn.Module, pb: dict, x: torch.Tensor) -> torch.Tensor:
    return vmap(lambda pp, xx: functional_call(net, pp, (xx,)))(pb, x)


def rollout(net: nn.Module, pb: dict, z0: torch.Tensor, fut_act: torch.Tensor, delta_mean, delta_std) -> torch.Tensor:
    z, out = z0, []
    for k in range(fut_act.shape[1]):
        z = z + delta_mean + delta_std * batched_apply(net, pb, torch.cat([z, fut_act[:, k]], dim=-1))
        out.append(z)
    return torch.stack(out, dim=1)


class AdaptForecasterBase:


    family = FAMILY
    budget_class = "baseline"
    needs_grad_at_eval = True
    stochastic = False

    def __init__(self, model: MetaMLP, cfg: dict, name: str):
        self.model = model
        self.cfg = cfg
        self.name = name

    def forecast(self, q: QueryBatch) -> Forecast:
        return self.forecast_with_diagnostics(q)[0]

    def forecast_with_diagnostics(self, q: QueryBatch):
        dev = self.model.delta_mean.device
        out_dev = q.ctx_obs.device
        with torch.inference_mode(False):
            co = q.ctx_obs.detach().to(dev, torch.float32).clone()
            ca = q.ctx_act.detach().to(dev, torch.float32).clone()
            fa = q.fut_act.detach().to(dev, torch.float32).clone()
            B = co.shape[0]
            chunk = max(1, int(self.cfg.get("eval_chunk", 240)))
            means, diags = [], []
            for s in range(0, B, chunk):
                n = min(chunk, B - s)
                c_o, c_a, f_a = co[s:s + n], ca[s:s + n], fa[s:s + n]
                if n < chunk:
                    rep = torch.zeros(chunk - n, dtype=torch.long, device=dev)
                    c_o, c_a, f_a = (torch.cat([v, v[rep]], 0) for v in (c_o, c_a, f_a))
                m, d = self._forecast_chunk(c_o, c_a, f_a)
                means.append(m[:n])
                diags.append({k: v[:n] for k, v in d.items()})
            mean = torch.cat(means, 0).to(out_dev)
        diag = {k: np.concatenate([d[k] for d in diags]) for k in diags[0]} if diags else {}
        fc = Forecast(mean=mean, comp_mean=mean.unsqueeze(2), comp_logw=mean.new_zeros(B, 1))
        return fc, diag

    def _forecast_chunk(self, co, ca, fa):
        raise NotImplementedError


class GrBALForecaster(AdaptForecasterBase):
    def __init__(self, model: MetaMLP, cfg: dict):
        super().__init__(model, cfg, NAME)

    def parameter_report(self) -> dict:
        return parameter_report(self.model)

    def _forecast_chunk(self, co, ca, fa):
        model, cfg = self.model, self.cfg
        net, dm, ds = model.net, model.delta_mean, model.delta_std
        p0 = model.theta(detach=True)
        M = int(cfg["adapt_m"])
        x, y = transitions(co, ca, dm, ds)
        xs, ys = x[:, -M:], y[:, -M:]
        with torch.enable_grad():
            pa = adapt_k_shot(net, p0, xs, ys, float(cfg["inner_lr"]))
        with torch.no_grad():
            pred = rollout(net, pa, co[:, -1], fa, dm, ds)
            pre = ((functional_call(net, p0, (xs,)) - ys) ** 2).mean((1, 2))
            post = ((batched_apply(net, pa, xs) - ys) ** 2).mean((1, 2))
        return pred, {"adapt_pre_mse": pre.cpu().numpy(), "adapt_post_mse": post.cpu().numpy()}


def forecaster(model: MetaMLP, cfg: dict) -> GrBALForecaster:
    return GrBALForecaster(model, cfg)


def context_region_tasks(obs_n: np.ndarray, act_n: np.ndarray, task_len: int = 32, region_end: int = 192):
    offs = list(range(0, region_end - task_len + 1, task_len))
    return obs_n[:, :region_end + 1], act_n[:, :region_end], offs


def stacked_tasks(model: MetaMLP, obs_n: np.ndarray, act_n: np.ndarray, task_len: int):
    dev = model.delta_mean.device
    o, a, offs = context_region_tasks(obs_n, act_n, task_len)
    x, y = transitions(torch.from_numpy(np.ascontiguousarray(o)).to(dev),
                       torch.from_numpy(np.ascontiguousarray(a)).to(dev), model.delta_mean, model.delta_std)
    xs = torch.stack([x[:, s:s + task_len] for s in offs], 1).flatten(0, 1)
    ys = torch.stack([y[:, s:s + task_len] for s in offs], 1).flatten(0, 1)
    return xs, ys


def adaptation_audit(model: MetaMLP, cfg: dict, obs_n: np.ndarray, act_n: np.ndarray, chunk: int = 96) -> dict:
    net, p0 = model.net, model.theta(detach=True)
    M, L = int(cfg["adapt_m"]), int(cfg["task_len"])
    xs, ys = stacked_tasks(model, obs_n, act_n, L)
    acc = {"pre_support": [], "post_support": [], "pre_query": [], "post_query": []}
    for s in range(0, xs.shape[0], chunk):
        sx, sy, qx, qy = xs[s:s + chunk, :M], ys[s:s + chunk, :M], xs[s:s + chunk, M:], ys[s:s + chunk, M:]
        with torch.enable_grad():
            pa = adapt_k_shot(net, p0, sx, sy, float(cfg["inner_lr"]))
        with torch.no_grad():
            acc["pre_support"].append(((functional_call(net, p0, (sx,)) - sy) ** 2).mean((1, 2)))
            acc["post_support"].append(((batched_apply(net, pa, sx) - sy) ** 2).mean((1, 2)))
            acc["pre_query"].append(((functional_call(net, p0, (qx,)) - qy) ** 2).mean((1, 2)))
            acc["post_query"].append(((batched_apply(net, pa, qx) - qy) ** 2).mean((1, 2)))
    t = {k: torch.cat(v).double().cpu().numpy() for k, v in acc.items()}
    return {
        "n_tasks": int(xs.shape[0]),
        **{k: float(v.mean()) for k, v in t.items()},
        "frac_tasks_query_improved": float(np.mean(t["post_query"] < t["pre_query"])),
        "frac_tasks_support_improved": float(np.mean(t["post_support"] < t["pre_support"])),
    }
