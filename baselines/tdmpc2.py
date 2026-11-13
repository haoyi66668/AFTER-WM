from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline import registry as P
from pipeline.eval_api import Forecast, QueryBatch

NAME = "tdmpc2"
FAMILY = "baseline"

_O = P.OBS_DIM
_A = P.ACT_DIM


def _lin(i: int, o: int) -> int:
    return i * o + o


def _normed(i: int, o: int) -> int:
    return _lin(i, o) + 2 * o


def _mlp_count(i: int, dims, o: int, last_normed: bool) -> int:
    d = [i] + list(dims) + [o]
    n = sum(_normed(d[k], d[k + 1]) for k in range(len(d) - 2))
    return n + (_normed(d[-2], d[-1]) if last_normed else _lin(d[-2], d[-1]))


def param_count(cfg: dict) -> int:
    L, E, M = cfg["latent_dim"], cfg["enc_dim"], cfg["mlp_dim"]
    enc = _mlp_count(_O, max(cfg["num_enc_layers"] - 1, 1) * [E], L, True)
    dyn = _mlp_count(L + _A, 2 * [M], L, True)
    dec = _mlp_count(L, 2 * [cfg["dec_dim"] or M], _O, False)
    return enc + dyn + dec


def default_config(budget_class: str = "baseline", **variant) -> dict:
    if budget_class != "baseline":
        raise ValueError(f"{NAME}: only budget_class 'baseline' is defined, got {budget_class!r}")
    cfg = dict(
        name=NAME, family=FAMILY, budget_class=budget_class, budget_target=P.BUDGET_BASELINE,
        latent_dim=128, enc_dim=256, num_enc_layers=2, simnorm_dim=8,
        mlp_dim=None,
        dec_dim=None,
        horizon=3, rho=0.5, consistency_coef=20.0,
        dec_coef=1.0,
        lr=3e-4, enc_lr_scale=0.3, grad_clip=20.0, optimizer="Adam",
        eval_chunk=240,
    )
    cfg.update(variant)
    if cfg["mlp_dim"] is None:
        target = cfg["budget_target"]
        best = None
        for m in range(8, 4096):
            n = param_count(dict(cfg, mlp_dim=m))
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (m, n)
        cfg["mlp_dim"] = best[0]
    cfg["param_count_formula"] = param_count(cfg)
    return cfg


class SimNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        return F.softmax(x, dim=-1).view(*shp)


class NormedLinear(nn.Linear):
    def __init__(self, *args, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)

    def forward(self, x):
        return self.act(self.ln(super().forward(x)))


def mlp(in_dim, mlp_dims, out_dim, act=None) -> nn.Sequential:
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + list(mlp_dims) + [out_dim]
    layers = [NormedLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 2)]
    layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TDMPC2WorldModel(nn.Module):
    GROUPS = {"encoder": ("_encoder",), "dynamics": ("_dynamics",), "decoder_adapter": ("_decoder",)}

    def __init__(self, cfg: dict):
        super().__init__()
        L, E, M = cfg["latent_dim"], cfg["enc_dim"], cfg["mlp_dim"]
        sn = cfg["simnorm_dim"]
        self._encoder = mlp(_O, max(cfg["num_enc_layers"] - 1, 1) * [E], L, act=SimNorm(sn))
        self._dynamics = mlp(L + _A, 2 * [M], L, act=SimNorm(sn))
        self._decoder = mlp(L, 2 * [cfg["dec_dim"] or M], _O)
        self.apply(weight_init)

    def encode(self, obs):
        return self._encoder(obs)

    def next(self, z, a):
        return self._dynamics(torch.cat([z, a], dim=-1))

    def decode(self, z):
        return self._decoder(z)


def build(cfg: dict, seed: int, device: str, stats=None) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = TDMPC2WorldModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert n == param_count(cfg), (n, param_count(cfg))
    return model.to(device)


def make_optimizer(model, cfg):
    lr = cfg["lr"]
    return torch.optim.Adam([
        {"params": model._dynamics.parameters()},
        {"params": model._decoder.parameters()},
        {"params": model._encoder.parameters(), "lr": lr * cfg["enc_lr_scale"]},
    ], lr=lr)


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
    Hz = int(cfg["horizon"])
    rho = float(cfg["rho"])
    B, T, _ = obs.shape
    n_sub = T - Hz
    Z = model.encode(obs)
    Zt = Z.detach()
    z = Z[:, :n_sub]
    zs = [z]
    cons = 0.0
    for t in range(Hz):
        z = model.next(z, act[:, t:t + n_sub])
        cons = cons + F.mse_loss(z, Zt[:, t + 1:t + 1 + n_sub]) * rho ** t
        zs.append(z)
    cons = cons / Hz
    pred = model.decode(torch.stack(zs, 0))
    dec = 0.0
    for t in range(Hz + 1):
        dec = dec + F.mse_loss(pred[t], obs[:, t:t + n_sub]) * rho ** t
    dec = dec / (Hz + 1)
    loss = cfg["consistency_coef"] * cons + cfg["dec_coef"] * dec
    return loss, _floats(loss=loss, consistency=cons, decoder=dec)


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
        z = m.encode(ctx_obs[:, -1])
        zs = []
        for k in range(fut_act.shape[1]):
            z = m.next(z, fut_act[:, k])
            zs.append(z)
        return m.decode(torch.stack(zs, 1))

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
              for g, names in TDMPC2WorldModel.GROUPS.items()}
    total = trainable + frozen
    assert sum(groups.values()) == total, (groups, total)
    target = cfg["budget_target"]
    return {"total": total, "trainable": trainable, "frozen": frozen, "stored_nongrad": 0,
            "buffers": buffers, "groups": groups, "budget_class": cfg["budget_class"], "budget_target": target,
            "within_budget": abs(total - target) <= P.BUDGET_TOL * target,
            "state": {"latent": cfg["latent_dim"]},
            "note": "decoder_adapter is a non-native jointly trained head; counted in total"}
