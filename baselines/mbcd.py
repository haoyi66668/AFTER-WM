from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline import paramcount as PC
from pipeline import registry as P
from pipeline import run_io as R
from pipeline.data import TrainStats
from pipeline.eval_api import Forecast, QueryBatch

NAME = "mbcd"
FAMILY = "baseline"
P2_AWARE = True
DIN = P.OBS_DIM + P.ACT_DIM
DOUT = P.OBS_DIM
LOG2PI = math.log(2.0 * math.pi)
HOLDOUT_SEED = 20261012
HOLDOUT_FRAC = 0.1

DISCLOSURES = [
    "B1 offline library building on logged TRAIN via per-episode detector passes (no SAC/reward/online stream); library models armed immediately (official load() semantics)",
    "B2 novel segments / back-dating to the last zero of the winning CUSUM statistic replace remove_last_n(100)",
    "B3 M_r warm-started from M0, test-time model = copy of the current model (official: fresh); K_max 4; <=1 test-time model per query; refinement pass added",
    "B4 function-preserving scaler refit for warm-started models; test-time model keeps its source scaler",
    "B5 h_build 100 (official); test-time h from {10,20,50,100,200} and max_std = 99th pct of M0 var_mean, both on the fixed TRAIN holdout (label-free)",
    "B6 one update = one shared schedule row with bootstrap-count weights (official minibatch 256); holdout = fixed 29 TRAIN episodes (fallback: official random 10% split)",
    "B7 test-time fine-tune: fixed <=50 steps per call, no holdout/elite re-selection; library never trained at test time; test-time min_steps 16",
    "B8 VAL every val_every global updates + end of each phase; best.pt = best VAL state over all phases",
    "B9 no reward dimension; one normalised state delta per observation dimension",
    "B10 padded fixed-size eval chunks + fixed-size fine-tune groups; bootstrap draws hashed from QueryBatch.key",
    "B11 test-time fine-tune lr tied to cfg lr (ft_lr None); an LR sweep also scales it",
]


def member_params(hidden: int, n_hidden: int = 4) -> int:
    return (DIN * hidden + hidden) + (n_hidden - 1) * (hidden * hidden + hidden) + (hidden * 2 * DOUT + 2 * DOUT)


def library_params(hidden: int, n_members: int = 5, k_max: int = 4, n_hidden: int = 4) -> int:
    return k_max * (n_members * member_params(hidden, n_hidden) + 2 * DOUT)


def solve_hidden(target: int, n_members: int = 5, k_max: int = 4, n_hidden: int = 4) -> int:
    return min(range(8, 2049), key=lambda h: (abs(library_params(h, n_members, k_max, n_hidden) - target), h))


def holdout_episodes(n_train: int = 288) -> np.ndarray:
    rng = np.random.default_rng(HOLDOUT_SEED)
    k = int(round(HOLDOUT_FRAC * n_train))
    return np.sort(rng.choice(n_train, size=k, replace=False)).astype(np.int64)


def default_config(budget_class: str = "baseline", **variant) -> dict:
    if budget_class != "baseline":
        raise ValueError(f"{NAME} is a compared model; budget_class must be 'baseline', got {budget_class!r}")
    cfg = dict(
        model=NAME,
        budget_class=budget_class,
        lr_sweep=True,
        K_max=4,
        n_members=5,
        n_elites=2,
        n_hidden=4,
        hidden=solve_hidden(P.BUDGET_BASELINE),
        weight_decays=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4],
        max_logvar_init=0.5,
        min_logvar_init=-10.0,
        logvar_bound_coef=0.01,
        lr=1e-3,
        adam_betas=[0.9, 0.999],
        adam_eps=1e-8,
        grad_clip=None,
        max_updates=4000,
        init_frac=0.75,
        epoch_updates=25,
        max_epochs_since_update=5,
        improve_tol=0.01,
        min_holdout=50,
        holdout_ratio_fallback=0.1,
        holdout_cap=5000,
        library_init="warm_m0",
        min_novel=2000,
        min_refine=200,
        refine_passes=1,
        cusum_threshold_build=100.0,
        h_grid=[10.0, 20.0, 50.0, 100.0, 200.0],
        num_stds=2.0,
        max_std_percentile=99.0,
        logp_floor=1e-8,
        var_eps=1e-6,
        tt_create=True,
        tt_min_steps=16,
        ft_steps=50,
        ft_every=10,
        ft_min_buffer=10,
        ft_lr=None,
        ft_group=16,
        eval_chunk=240,
    )
    for k, v in variant.items():
        if k not in cfg and not k.startswith("dev_"):
            raise KeyError(f"unknown config key {k!r}")
        cfg[k] = v
    if cfg["library_init"] not in ("warm_m0", "fresh"):
        raise ValueError("library_init must be 'warm_m0' or 'fresh'")
    return cfg


class Ensemble(nn.Module):

    def __init__(self, n_members: int, hidden: int, n_hidden: int, n_elites: int, max_lv: float, min_lv: float):
        super().__init__()
        dims = [DIN] + [hidden] * n_hidden + [2 * DOUT]
        self.W = nn.ParameterList([nn.Parameter(torch.empty(n_members, dims[i], dims[i + 1]))
                                   for i in range(len(dims) - 1)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(n_members, 1, dims[i + 1]))
                                   for i in range(len(dims) - 1)])
        self.max_logvar = nn.Parameter(torch.full((1, DOUT), float(max_lv)))
        self.min_logvar = nn.Parameter(torch.full((1, DOUT), float(min_lv)))
        self.register_buffer("in_mu", torch.zeros(1, DIN))
        self.register_buffer("in_sigma", torch.ones(1, DIN))
        self.register_buffer("elites", torch.arange(n_elites, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for W in self.W:
                e, fi, fo = W.shape
                lim = math.sqrt(6.0 / (e * fi + e * fo))
                W.uniform_(-lim, lim)
            for b in self.b:
                b.zero_()

    def plist(self) -> list:
        return list(self.W) + list(self.b) + [self.max_logvar, self.min_logvar]


class MBCDLibrary(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.ens = nn.ModuleList([
            Ensemble(int(cfg["n_members"]), int(cfg["hidden"]), int(cfg["n_hidden"]), int(cfg["n_elites"]),
                     float(cfg["max_logvar_init"]), float(cfg["min_logvar_init"])) for _ in range(int(cfg["K_max"]))])
        self.register_buffer("n_active", torch.ones((), dtype=torch.long))
        self.register_buffer("h_test", torch.tensor(float(cfg["cusum_threshold_build"]), dtype=torch.float64))
        self.register_buffer("max_std", torch.tensor(float("inf"), dtype=torch.float64))


def ens_forward(Ws, bs, max_lv, min_lv, mu, sigma, x):
    h = (x - mu) / sigma
    n = len(Ws)
    for i in range(n - 1):
        h = torch.relu(torch.matmul(h, Ws[i]) + bs[i])
    out = torch.matmul(h, Ws[n - 1]) + bs[n - 1]
    mean, raw = out[..., :DOUT], out[..., DOUT:]
    lv = max_lv - F.softplus(max_lv - raw)
    lv = min_lv + F.softplus(lv - min_lv)
    return mean, lv


def ens_mean_only(Ws, bs, mu, sigma, x):
    h = (x - mu) / sigma
    n = len(Ws)
    for i in range(n - 1):
        h = torch.relu(torch.matmul(h, Ws[i]) + bs[i])
    return (torch.matmul(h, Ws[n - 1]) + bs[n - 1])[..., :DOUT]


def ens_call(ens: Ensemble, x):
    return ens_forward(list(ens.W), list(ens.b), ens.max_logvar, ens.min_logvar, ens.in_mu, ens.in_sigma, x)


def nll_terms(mean, lv, y, w):
    per = ((mean - y) ** 2 * torch.exp(-lv)).mean(-1) + lv.mean(-1)
    return (w * per).sum(-1) / w.sum(-1).clamp_min(1.0)


def decay_terms(Ws, decays):
    tot = 0.0
    for W, d in zip(Ws, decays):
        tot = tot + float(d) * 0.5 * (W * W).sum()
    return tot


def moment_stats(mean, lv, y, num_stds: float, log_floor, var_eps: float):
    m = mean.double()
    v = lv.double().exp()
    mu = m.mean(-3)
    var = (m * m + v).mean(-3) - mu * mu + var_eps
    logdet = var.log().sum(-1)
    k = mean.shape[-1]
    lp = -0.5 * (k * LOG2PI + logdet + ((y.double() - mu) ** 2 / var).sum(-1))
    ln = -0.5 * (k * LOG2PI + logdet + float(num_stds) ** 2 * k)
    if log_floor is not None:
        lp = torch.logaddexp(lp, torch.full_like(lp, log_floor))
        ln = torch.logaddexp(ln, torch.full_like(ln, log_floor))
    vm = m.std(-3, unbiased=False).norm(dim=-1)
    return lp, vm, ln


def _log_floor(cfg):
    f = cfg.get("logp_floor")
    return None if f is None or float(f) <= 0 else math.log(float(f))


def build(cfg: dict, seed: int, device, stats: TrainStats) -> MBCDLibrary:
    torch.manual_seed(int(seed))
    model = MBCDLibrary(cfg)
    if not cfg.get("dev_allow_offbudget"):
        chk = PC.budget_check(sum(p.numel() for p in model.parameters()), "baseline")
        if not chk["within_budget"]:
            raise RuntimeError(f"parameter budget violated: {chk}")
    return model.to(device)


def make_optimizer(model: MBCDLibrary, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.ens[0].parameters(), lr=float(cfg["lr"]), betas=tuple(cfg["adam_betas"]),
                            eps=float(cfg["adam_eps"]))


def _window_batch(batch: dict):
    z = batch["win_obs"][:, :-1]
    x = torch.cat([z, batch["win_act"]], -1).reshape(-1, DIN)
    y = (batch["win_obs"][:, 1:] - z).reshape(-1, DOUT)
    return x, y


def train_step(model: MBCDLibrary, batch: dict, step: int, cfg: dict):
    ens = model.ens[0]
    x, y = _window_batch(batch)
    mean, lv = ens_call(ens, x)
    w = torch.ones(mean.shape[0], mean.shape[1], device=x.device)
    loss_e = nll_terms(mean, lv, y, w)
    loss = loss_e.sum() + decay_terms(list(ens.W), cfg["weight_decays"]) \
        + float(cfg["logvar_bound_coef"]) * (ens.max_logvar.sum() - ens.min_logvar.sum())
    return loss, {"nll_mean": loss_e.detach().mean()}


@torch.no_grad()
def library_stats(model: MBCDLibrary, n_models: int, X: torch.Tensor, Y: torch.Tensor, cfg: dict, chunk: int = 65536):
    lf = _log_floor(cfg)
    outs = [[], [], []]
    for s in range(0, X.shape[0], chunk):
        xs, ys = X[s:s + chunk], Y[s:s + chunk]
        parts = [[], [], []]
        for k in range(n_models):
            mean, lv = ens_call(model.ens[k], xs)
            for i, t in enumerate(moment_stats(mean, lv, ys, cfg["num_stds"], lf, cfg["var_eps"])):
                parts[i].append(t)
        for i in range(3):
            outs[i].append(torch.stack(parts[i], -1))
    return tuple(torch.cat(o, 0) for o in outs)


@torch.no_grad()
def cusum_pass(lp, vm, ln, h: float, max_std: float, novel_as_current: bool = False):
    n_ep, T, K = lp.shape
    dev = lp.device
    cur = torch.zeros(n_ep, dtype=torch.long, device=dev)
    S = torch.zeros(n_ep, K, dtype=torch.float64, device=dev)
    Sn = torch.zeros(n_ep, dtype=torch.float64, device=dev)
    lzS = torch.full((n_ep, K), -1, dtype=torch.long, device=dev)
    lzN = torch.full((n_ep,), -1, dtype=torch.long, device=dev)
    changed = torch.zeros(n_ep, dtype=torch.bool, device=dev)
    novel = torch.zeros(n_ep, dtype=torch.bool, device=dev)
    labels = torch.zeros(n_ep, T, dtype=torch.long, device=dev)
    pos = torch.arange(T, device=dev)[None, :]
    ar = torch.arange(K, device=dev)[None, :]
    n_alarm = torch.zeros(n_ep, dtype=torch.long, device=dev)
    n_new = torch.zeros_like(n_alarm)
    n_switch = torch.zeros_like(n_alarm)
    for j in range(T):
        S = torch.where(changed[:, None], torch.zeros_like(S), S)
        Sn = torch.where(changed, torch.zeros_like(Sn), Sn)
        lzS = torch.where(changed[:, None], torch.full_like(lzS, j - 1), lzS)
        lzN = torch.where(changed, torch.full_like(lzN, j - 1), lzN)
        lpj, vmj, lnj = lp[:, j], vm[:, j], ln[:, j]
        lpc = lpj.gather(1, cur[:, None])[:, 0]
        vmc = vmj.gather(1, cur[:, None])[:, 0]
        lnc = lnj.gather(1, cur[:, None])[:, 0]
        gate = vmc < max_std
        upd = gate[:, None] & (ar != cur[:, None])
        S = torch.where(upd, (S + lpj - lpc[:, None]).clamp_min(0.0), S)
        lzS = torch.where(upd & (S == 0), torch.full_like(lzS, j), lzS)
        Sn = torch.where(gate, (Sn + lnc - lpc).clamp_min(0.0), Sn)
        lzN = torch.where(gate & (Sn == 0), torch.full_like(lzN, j), lzN)
        smax, w = S.max(1)
        alarm = torch.maximum(smax, Sn) > h
        is_new = alarm & (Sn >= smax)
        is_sw = alarm & ~is_new
        labels[:, j] = cur if novel_as_current else torch.where(novel, torch.full_like(cur, -1), cur)
        if bool(is_sw.any()):
            lzw = lzS.gather(1, w[:, None])[:, 0]
            m = is_sw[:, None] & (pos > lzw[:, None]) & (pos <= j)
            labels = torch.where(m, w[:, None].expand_as(labels), labels)
            cur = torch.where(is_sw, w, cur)
            novel = novel & ~is_sw
        if bool(is_new.any()) and not novel_as_current:
            m = is_new[:, None] & (pos > lzN[:, None]) & (pos <= j)
            labels = torch.where(m, torch.full_like(labels, -1), labels)
            novel = novel | is_new
        changed = alarm
        n_alarm += alarm.long()
        n_new += is_new.long()
        n_switch += is_sw.long()
    return labels, {"alarms": int(n_alarm.sum()), "new_alarms": int(n_new.sum()), "switch_alarms": int(n_switch.sum()),
                    "episodes_with_alarm": int((n_alarm > 0).sum())}


_MOD = 2147483647


def _mix(x: torch.Tensor) -> torch.Tensor:
    for a in (48271, 69621, 16807):
        x = (x * a) % _MOD
        x = (x ^ (x >> 11)) % _MOD
    return x


def hash_uniform(key: torch.Tensor, a: torch.Tensor, n_e: int, n_d: int) -> torch.Tensor:
    dev = key.device
    x = _mix((key.remainder(_MOD) + 1) % _MOD)
    x = _mix((x + 7919 * (a.remainder(100000) + 1)) % _MOD)
    e = torch.arange(n_e, device=dev, dtype=torch.long)
    d = torch.arange(n_d, device=dev, dtype=torch.long)
    x = _mix((x[:, None] + 104729 * (e[None, :] + 1)) % _MOD)
    x = _mix((x[:, :, None] + 1299709 * (d[None, None, :] + 1)) % _MOD)
    return x.double() / float(_MOD)


def bootstrap_counts(key, call_idx, inbuf, n_e: int) -> torch.Tensor:
    Q, L = inbuf.shape
    blen = inbuf.sum(1)
    u = hash_uniform(key, call_idx, n_e, L)
    rank1 = (u * blen[:, None, None].double()).floor().long() + 1
    csum = inbuf.long().cumsum(1)
    hit = (csum[:, None, None, :] == rank1[..., None]) & inbuf[:, None, None, :]
    valid = torch.arange(L, device=inbuf.device)[None, None, :] < blen[:, None, None]
    return (hit & valid[..., None]).sum(2).to(torch.float32)


class MBCDForecaster:
    name = NAME
    family = FAMILY
    budget_class = "baseline"
    needs_grad_at_eval = True
    stochastic = True

    def __init__(self, model: MBCDLibrary, cfg: dict, h_override: float | None = None, create: bool | None = None):
        self.model = model
        self.cfg = cfg
        self.h_override = h_override
        self.create = bool(cfg.get("tt_create", True)) if create is None else bool(create)

    def parameter_report(self) -> dict:
        return parameter_report(self.model, self.cfg)

    def forecast(self, q: QueryBatch) -> Forecast:
        return self.forecast_with_diagnostics(q)[0]

    def forecast_with_diagnostics(self, q: QueryBatch):
        dev = self.model.n_active.device
        out_dev = q.ctx_obs.device
        with torch.inference_mode(False):
            co = q.ctx_obs.detach().to(dev, torch.float32).clone()
            ca = q.ctx_act.detach().to(dev, torch.float32).clone()
            fa = q.fut_act.detach().to(dev, torch.float32).clone()
            ky = q.key.detach().to(dev, torch.long).clone()
            B = co.shape[0]
            chunk = max(1, int(self.cfg.get("eval_chunk", 240)))
            comps, diags = [], []
            for s in range(0, B, chunk):
                n = min(chunk, B - s)
                c_o, c_a, f_a, k_y = co[s:s + n], ca[s:s + n], fa[s:s + n], ky[s:s + n]
                if n < chunk:
                    rep = torch.zeros(chunk - n, dtype=torch.long, device=dev)
                    c_o, c_a, f_a, k_y = (torch.cat([v, v[rep]], 0) for v in (c_o, c_a, f_a, k_y))
                cm, d = self._forecast_chunk(c_o, c_a, f_a, k_y)
                comps.append(cm[:n].detach())
                diags.append({k: v[:n] for k, v in d.items()})
            comp_mean = torch.cat(comps, 0).to(out_dev)
        K = comp_mean.shape[2]
        comp_logw = torch.full((B, K), -math.log(K), dtype=comp_mean.dtype, device=out_dev)
        fc = Forecast(mean=comp_mean.mean(2), comp_mean=comp_mean, comp_logw=comp_logw)
        diag = {k: np.concatenate([d[k] for d in diags]) for k in diags[0]} if diags else {}
        return fc, diag

    def _forecast_chunk(self, co, ca, fa, key):
        model, cfg = self.model, self.cfg
        dev = co.device
        Kl = int(model.n_active)
        T = Kl
        M = Kl + 1
        Q, L = co.shape[0], co.shape[1] - 1
        E = int(cfg["n_members"])
        h = float(self.h_override if self.h_override is not None else model.h_test)
        max_std = float(model.max_std)
        lf = _log_floor(cfg)
        x = torch.cat([co[:, :-1], ca], -1)
        y = co[:, 1:] - co[:, :-1]
        lib = model.ens[:Kl]
        lp_l, vm_l, ln_l = library_stats(model, Kl, x.reshape(-1, DIN), y.reshape(-1, DOUT), cfg)
        lp_l, vm_l, ln_l = (t.reshape(Q, L, Kl) for t in (lp_l, vm_l, ln_l))
        nl = len(lib[0].W)
        LW = [torch.stack([e.W[i].detach() for e in lib]) for i in range(nl)]
        Lb = [torch.stack([e.b[i].detach() for e in lib]) for i in range(nl)]
        Lmax = torch.stack([e.max_logvar.detach() for e in lib])
        Lmin = torch.stack([e.min_logvar.detach() for e in lib])
        Lmu = torch.stack([e.in_mu for e in lib])
        Lsig = torch.stack([e.in_sigma for e in lib])
        Lel = torch.stack([e.elites for e in lib])
        TW = [torch.zeros((Q,) + w.shape[1:], device=dev) for w in LW]
        Tb = [torch.zeros((Q,) + b.shape[1:], device=dev) for b in Lb]
        Tmax = torch.zeros(Q, 1, DOUT, device=dev)
        Tmin = torch.zeros(Q, 1, DOUT, device=dev)
        Tmu = torch.zeros(Q, 1, DIN, device=dev)
        Tsig = torch.ones(Q, 1, DIN, device=dev)
        Tel = torch.zeros(Q, Lel.shape[1], dtype=torch.long, device=dev)
        tparams = TW + Tb + [Tmax, Tmin]
        adam_m = [torch.zeros_like(p) for p in tparams]
        adam_v = [torch.zeros_like(p) for p in tparams]
        adam_t = torch.zeros(Q, dtype=torch.long, device=dev)
        created = torch.zeros(Q, dtype=torch.bool, device=dev)
        counterT = torch.zeros(Q, dtype=torch.long, device=dev)
        inbuf = torch.zeros(Q, L, dtype=torch.bool, device=dev)
        ncalls = torch.zeros(Q, dtype=torch.long, device=dev)
        cur = torch.zeros(Q, dtype=torch.long, device=dev)
        S = torch.zeros(Q, M, dtype=torch.float64, device=dev)
        Sn = torch.zeros(Q, dtype=torch.float64, device=dev)
        lzS = torch.full((Q, M), -1, dtype=torch.long, device=dev)
        lzN = torch.full((Q,), -1, dtype=torch.long, device=dev)
        changed = torch.zeros(Q, dtype=torch.bool, device=dev)
        n_alarm = torch.zeros(Q, dtype=torch.long, device=dev)
        n_create = torch.zeros_like(n_alarm)
        n_switch = torch.zeros_like(n_alarm)
        pos = torch.arange(L, device=dev)[None, :]
        arM = torch.arange(M, device=dev)[None, :]
        lib_active = arM < Kl
        tt_min = int(cfg["tt_min_steps"])
        for tau in range(L):
            S = torch.where(changed[:, None], torch.zeros_like(S), S)
            Sn = torch.where(changed, torch.zeros_like(Sn), Sn)
            lzS = torch.where(changed[:, None], torch.full_like(lzS, tau - 1), lzS)
            lzN = torch.where(changed, torch.full_like(lzN, tau - 1), lzN)
            with torch.no_grad():
                mT, lvT = ens_forward(TW, Tb, Tmax[:, None], Tmin[:, None], Tmu[:, None], Tsig[:, None],
                                      x[:, None, tau:tau + 1, :])
                lpT, vmT, lnT = moment_stats(mT, lvT, y[:, tau:tau + 1], cfg["num_stds"], lf, cfg["var_eps"])
            lp = torch.cat([lp_l[:, tau], lpT[:, :1]], 1)
            vm = torch.cat([vm_l[:, tau], vmT[:, :1]], 1)
            ln = torch.cat([ln_l[:, tau], lnT[:, :1]], 1)
            active = lib_active | ((arM == T) & created[:, None])
            lpc = lp.gather(1, cur[:, None])[:, 0]
            vmc = vm.gather(1, cur[:, None])[:, 0]
            lnc = ln.gather(1, cur[:, None])[:, 0]
            armed = (cur < Kl) | (counterT > tt_min)
            gate = (vmc < max_std) & armed
            upd = gate[:, None] & active & (arM != cur[:, None])
            S = torch.where(upd, (S + lp - lpc[:, None]).clamp_min(0.0), S)
            lzS = torch.where(upd & (S == 0), torch.full_like(lzS, tau), lzS)
            Sn = torch.where(gate, (Sn + lnc - lpc).clamp_min(0.0), Sn)
            lzN = torch.where(gate & (Sn == 0), torch.full_like(lzN, tau), lzN)
            smax, w = torch.where(active, S, torch.full_like(S, -1.0)).max(1)
            alarm = torch.maximum(smax, Sn) > h
            is_new = alarm & (Sn >= smax)
            is_sw = alarm & ~is_new
            if not self.create:
                is_new = torch.zeros_like(is_new)
            if bool(is_new.any()):
                src = cur.clamp(max=Kl - 1)
                fromT = cur == T

                def pick(lib_t, t_t):
                    s = lib_t[src]
                    s = torch.where(fromT.view((Q,) + (1,) * (s.dim() - 1)), t_t, s)
                    return torch.where(is_new.view((Q,) + (1,) * (s.dim() - 1)), s, t_t)
                TW = [pick(a, b) for a, b in zip(LW, TW)]
                Tb = [pick(a, b) for a, b in zip(Lb, Tb)]
                Tmax = pick(Lmax, Tmax)
                Tmin = pick(Lmin, Tmin)
                Tmu = pick(Lmu, Tmu)
                Tsig = pick(Lsig, Tsig)
                Tel = pick(Lel, Tel)
                tparams = TW + Tb + [Tmax, Tmin]
                adam_m = [torch.where(is_new.view((Q,) + (1,) * (p.dim() - 1)), torch.zeros_like(p), m)
                          for p, m in zip(tparams, adam_m)]
                adam_v = [torch.where(is_new.view((Q,) + (1,) * (p.dim() - 1)), torch.zeros_like(p), v)
                          for p, v in zip(tparams, adam_v)]
                adam_t = torch.where(is_new, torch.zeros_like(adam_t), adam_t)
                inbuf = torch.where(is_new[:, None], (pos > lzN[:, None]) & (pos <= tau), inbuf)
                counterT = torch.where(is_new, torch.zeros_like(counterT), counterT)
                created = created | is_new
                cur = torch.where(is_new, torch.full_like(cur, T), cur)
            if bool(is_sw.any()):
                swT = is_sw & (w == T)
                lzT = lzS[:, T]
                inbuf = inbuf | (swT[:, None] & (pos > lzT[:, None]) & (pos <= tau))
                cur = torch.where(is_sw, w, cur)
            changed = alarm
            n_alarm += alarm.long()
            n_create += is_new.long()
            n_switch += is_sw.long()
            isT = cur == T
            inbuf[:, tau] = inbuf[:, tau] | isT
            counterT = counterT + isT.long()
            blen = inbuf.sum(1)
            train_now = isT & (blen > int(cfg["ft_min_buffer"])) & (alarm | (blen % int(cfg["ft_every"]) == 0))
            if bool(train_now.any()):
                tparams, adam_m, adam_v, adam_t = self._finetune_queries(tparams, adam_m, adam_v, adam_t, Tmu, Tsig,
                                                                         x, y, inbuf, train_now, key, ncalls)
                ncalls = ncalls + train_now.long()
                TW, Tb = tparams[:nl], tparams[nl:2 * nl]
                Tmax, Tmin = tparams[2 * nl], tparams[2 * nl + 1]
        with torch.no_grad():
            z0 = co[:, -1]
            H = fa.shape[1]
            nel = Lel.shape[1]
            EW = [torch.stack([LW[i][k][Lel[k]] for k in range(Kl)]) for i in range(nl)]
            Eb = [torch.stack([Lb[i][k][Lel[k]] for k in range(Kl)]) for i in range(nl)]
            muL, sgL = Lmu.view(Kl, 1, 1, DIN), Lsig.view(Kl, 1, 1, DIN)
            qi = torch.arange(Q, device=dev)[:, None]
            TEW = [w_[qi, Tel] for w_ in TW]
            TEb = [b_[qi, Tel] for b_ in Tb]
            zl = z0[:, None, None, :].expand(Q, Kl, nel, DOUT).contiguous()
            zt = z0[:, None, :].expand(Q, nel, DOUT).contiguous()
            out_l, out_t = [], []
            for k in range(H):
                a = fa[:, k]
                inp = torch.cat([zl, a[:, None, None, :].expand(Q, Kl, nel, P.ACT_DIM)], -1)[..., None, :]
                zl = zl + ens_mean_only(EW, Eb, muL, sgL, inp)[..., 0, :]
                inpt = torch.cat([zt, a[:, None, :].expand(Q, nel, P.ACT_DIM)], -1)[..., None, :]
                zt = zt + ens_mean_only(TEW, TEb, Tmu[:, None], Tsig[:, None], inpt)[..., 0, :]
                out_l.append(zl)
                out_t.append(zt)
            rl = torch.stack(out_l, 1)
            rt = torch.stack(out_t, 1)
            idx = cur.clamp(max=Kl - 1).view(Q, 1, 1, 1, 1).expand(Q, H, 1, nel, DOUT)
            sel_l = rl.gather(2, idx)[:, :, 0]
            comp = torch.where((cur == T).view(Q, 1, 1, 1), rt, sel_l)
        diag = {"final_model": cur.cpu().numpy(), "created": created.cpu().numpy(), "n_alarms": n_alarm.cpu().numpy(),
                "n_create": n_create.cpu().numpy(), "n_switch": n_switch.cpu().numpy(),
                "n_ft_calls": ncalls.cpu().numpy(), "buffer_len": inbuf.sum(1).cpu().numpy()}
        return comp, diag

    def _finetune_queries(self, tparams, adam_m, adam_v, adam_t, Tmu, Tsig, x, y, inbuf, train_now, key, ncalls):
        G = max(1, int(self.cfg.get("ft_group", 16)))
        idx = torch.nonzero(train_now)[:, 0]
        nreal = int(idx.numel())
        pad = (-nreal) % G
        if pad:
            idx = torch.cat([idx, idx[:1].expand(pad)])
        tparams = [p.clone() for p in tparams]
        adam_m = [p.clone() for p in adam_m]
        adam_v = [p.clone() for p in adam_v]
        adam_t = adam_t.clone()
        for g0 in range(0, int(idx.numel()), G):
            gi = idx[g0:g0 + G]
            real = min(G, nreal - g0)
            new_p, new_m, new_v, new_t = self._finetune_group(
                [p[gi] for p in tparams], [m[gi] for m in adam_m], [v[gi] for v in adam_v], adam_t[gi], Tmu[gi],
                Tsig[gi], x[gi], y[gi], inbuf[gi], key[gi], ncalls[gi])
            ri = gi[:real]
            for lst, new in ((tparams, new_p), (adam_m, new_m), (adam_v, new_v)):
                for i in range(len(lst)):
                    lst[i][ri] = new[i][:real]
            adam_t[ri] = new_t[:real]
        return tparams, adam_m, adam_v, adam_t

    def _finetune_group(self, params0, m0, v0, t0, Tmu, Tsig, x, y, inbuf, key, ncalls):
        cfg = self.cfg
        G = x.shape[0]
        nl = (len(params0) - 2) // 2
        E = params0[0].shape[1]
        lr = float(cfg["ft_lr"] if cfg.get("ft_lr") is not None else cfg["lr"])
        b1, b2 = (float(v) for v in cfg["adam_betas"])
        eps = float(cfg["adam_eps"])
        coef = float(cfg["logvar_bound_coef"])
        decays = cfg["weight_decays"]
        w = bootstrap_counts(key, ncalls, inbuf, E)
        xin = x[:, None]
        yin = y[:, None]
        params = [p.detach().clone().requires_grad_(True) for p in params0]
        m = [t.clone() for t in m0]
        v = [t.clone() for t in v0]
        t = t0.clone()
        with torch.enable_grad():
            for _ in range(int(cfg["ft_steps"])):
                Ws, bs = params[:nl], params[nl:2 * nl]
                mx, mn = params[2 * nl], params[2 * nl + 1]
                mean, lv = ens_forward(Ws, bs, mx[:, None], mn[:, None], Tmu[:, None], Tsig[:, None], xin)
                loss_q = nll_terms(mean, lv, yin, w).sum(-1)
                for W_, d in zip(Ws, decays):
                    loss_q = loss_q + float(d) * 0.5 * (W_ * W_).sum(dim=(1, 2, 3))
                loss_q = loss_q + coef * (mx.sum(dim=(1, 2)) - mn.sum(dim=(1, 2)))
                grads = torch.autograd.grad(loss_q.sum(), params)
                t = t + 1
                tf = t.double()
                bc1 = (1.0 - b1 ** tf).float()
                bc2 = (1.0 - b2 ** tf).float()
                new_params = []
                for i, (p, g) in enumerate(zip(params, grads)):
                    shp = (G,) + (1,) * (p.dim() - 1)
                    m[i] = b1 * m[i] + (1 - b1) * g
                    v[i] = b2 * v[i] + (1 - b2) * g * g
                    step = lr * (m[i] / bc1.view(shp)) / ((v[i] / bc2.view(shp)).sqrt() + eps)
                    new_params.append((p.detach() - step).requires_grad_(True))
                params = new_params
        return [p.detach() for p in params], [a.detach() for a in m], [a.detach() for a in v], t


def forecaster(model: MBCDLibrary, cfg: dict) -> MBCDForecaster:
    return MBCDForecaster(model, cfg)


def parameter_report(model: MBCDLibrary, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    groups = {f"ens{k}": sum(p.numel() for p in e.parameters()) for k, e in enumerate(model.ens)}
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    per_model = groups["ens0"]
    return PC.make_report(model, stored_nongrad=0, groups=groups, extra={
        "buffers_note": "per-ensemble input scaler (2x23), elite indices, n_active, h_test, max_std: normaliser "
                        "constants / bookkeeping (not counted)",
        "budget": PC.budget_check(trainable, "baseline"),
        "active_library_models": int(model.n_active),
        "h_test": float(model.h_test), "max_std": float(model.max_std),
        "per_query_state": {"test_time_model_values": per_model, "adam_moment_values": 2 * per_model,
                            "note": "functional per-query test-time ensemble + Adam moments (state, not counted)"},
    })


class _Trainer:

    def __init__(self, cfg, seed, out_dir, device, stats, schedule, data):
        from pipeline import evaluate as E
        from pipeline import trainer as TR
        self.TR, self.E = TR, E
        self.ctx = TR.current_context()
        self.cfg, self.seed, self.out_dir, self.device, self.stats = cfg, int(seed), out_dir, device, stats
        self.module = sys.modules[__name__]
        self.model = build(cfg, seed, device, stats)
        n_ep = int(data.obs_n.shape[0])
        self.n_ep = n_ep
        obs = torch.from_numpy(np.ascontiguousarray(data.obs_n)).to(device)
        act = torch.from_numpy(np.ascontiguousarray(data.act_n)).to(device)
        self.X = torch.cat([obs[:, :-1], act], -1).contiguous()
        self.Y = (obs[:, 1:] - obs[:, :-1]).contiguous()
        self.Tn = int(self.X.shape[1])
        self.hold = holdout_episodes(n_ep)
        self.is_hold = np.zeros(n_ep, bool)
        self.is_hold[self.hold] = True
        self.train_eps = np.flatnonzero(~self.is_hold)
        ep, tt = schedule
        self.sched_ep = torch.from_numpy(np.asarray(ep)).to(device)
        self.sched_j = torch.from_numpy(np.asarray(tt)[:, :, None] - P.CONTEXT
                                        + np.arange(P.CONTEXT + P.MAX_H)[None, None, :]).to(device)
        self.U = int(min(int(cfg["max_updates"]), int(self.ctx.steps), int(ep.shape[0])))
        self.val_every = int(self.ctx.val_every)
        self.ckpt_every = int(self.ctx.ckpt_every)
        K = int(cfg["K_max"])
        lab = np.zeros((n_ep, self.Tn), np.int16)
        lab[self.is_hold] = 0
        self.st = {"stage": "m0", "r": 1, "ref_i": 0, "ref_k": 0, "global_step": 0, "used": [0] * K,
                   "row_ptr": [0] * K, "adam": [None] * K, "call": None, "labels": lab, "ref_labels": None,
                   "history": [], "last_val_step": -1, "phase_val": []}
        self.planned = None

    def save(self):
        self.ctx.save_checkpoint({"model": self.model.state_dict(), "step": int(self.st["global_step"]),
                                  "st": self.st})

    def try_resume(self):
        ck = self.ctx.load_checkpoint()
        if ck is None:
            self.ctx.reconcile_after_resume(0)
            return
        self.model.load_state_dict(ck["model"])
        self.st = ck["st"]
        self.E.restore_rng(ck["rng"])
        self.ctx.reconcile_after_resume(int(ck["step"]))
        print(json.dumps({"stage": "resumed", "step": int(ck["step"]), "mbcd_stage": self.st["stage"],
                          "best_step": self.ctx.best_step, "best_val": self.ctx.best_val}), flush=True)

    def stop_requested(self):
        return bool(self.TR._Stop.flag)

    def terminate(self):
        self.save()
        print(json.dumps({"stage": "terminated", "signal": self.TR._Stop.signum,
                          "checkpoint_step": int(self.st["global_step"])}), flush=True)
        raise SystemExit(self.TR.EXIT_TERMINATED)

    def validate(self, component):
        s = int(self.st["global_step"])
        self.ctx.validate(s, self.module, self.model, self.cfg, component=component)
        self.st["last_val_step"] = s

    def record(self, rec):
        rec = dict(rec, global_step=int(self.st["global_step"]), time=R.now_iso())
        self.st["history"].append(rec)
        print(json.dumps({"stage": "mbcd_phase", **{k: v for k, v in rec.items() if not isinstance(v, (list, dict))
                                                     or k in ("label_counts", "alarms")}}), flush=True)

    def flat(self, mask_np):
        m = torch.from_numpy(mask_np.reshape(-1)).to(self.device)
        return self.X.reshape(-1, DIN)[m], self.Y.reshape(-1, DOUT)[m]

    def detector_labels(self, n_models, h, novel_as_current):
        X = self.X.reshape(-1, DIN)
        Y = self.Y.reshape(-1, DOUT)
        lp, vm, ln = library_stats(self.model, n_models, X, Y, self.cfg)
        sh = (self.n_ep, self.Tn, n_models)
        labels, cnt = cusum_pass(lp.reshape(sh), vm.reshape(sh), ln.reshape(sh), float(h), float(self.model.max_std),
                                 novel_as_current=novel_as_current)
        lab = labels.cpu().numpy().astype(np.int16)
        tr, ho = lab[~self.is_hold], lab[self.is_hold]
        counts = {"train": {str(int(v)): int((tr == v).sum()) for v in np.unique(tr)},
                  "holdout": {str(int(v)): int((ho == v).sum()) for v in np.unique(ho)}}
        return lab, cnt, counts

    def train_call(self, k, call_id, train_mask, hold_mask, budget, refit_scaler):
        cfg, model, dev, st = self.cfg, self.model, self.device, self.st
        ens = model.ens[k]
        E = int(cfg["n_members"])
        call_idx = sum(1 for c in st["history"] if c.get("kind") == "train_call" and c.get("k") == k)
        rng = np.random.default_rng([self.seed, 7, k, call_idx])
        tr_mask = train_mask.copy()
        ho_mask = hold_mask.copy()
        fallback = False
        if int(ho_mask.sum()) < int(cfg["min_holdout"]):
            idx = np.flatnonzero(tr_mask.reshape(-1))
            nh = min(int(len(idx) * float(cfg["holdout_ratio_fallback"])), int(cfg["holdout_cap"]))
            pick = rng.permutation(len(idx))[:nh]
            ho_mask = np.zeros_like(tr_mask)
            ho_mask.reshape(-1)[idx[pick]] = True
            tr_mask.reshape(-1)[idx[pick]] = False
            fallback = True
        n_tr = int(tr_mask.sum())
        Xh, Yh = self.flat(ho_mask)
        idx = np.flatnonzero(tr_mask.reshape(-1))
        draws = rng.integers(len(idx), size=(E, len(idx)))
        cnt = np.zeros((E, self.n_ep * self.Tn), np.float32)
        for e in range(E):
            cnt[e, idx] = np.bincount(draws[e], minlength=len(idx)).astype(np.float32)
        cnt = torch.from_numpy(cnt.reshape(E, self.n_ep, self.Tn)).to(dev)
        call = st["call"]
        if call is None or call.get("id") != call_id:
            if refit_scaler:
                self._refit_scaler(ens, tr_mask, preserve=(call_id != "m0"))
            call = {"id": call_id, "k": k, "done": 0, "best": [1e10] * E, "since": 0, "epochs": 0,
                    "snap": [p.detach().clone() for p in list(ens.W) + list(ens.b)], "hold_mse": None,
                    "stopped": False}
            st["call"] = call
        opt = torch.optim.Adam(ens.parameters(), lr=float(cfg["lr"]), betas=tuple(cfg["adam_betas"]),
                               eps=float(cfg["adam_eps"]))
        if st["adam"][k] is not None:
            opt.load_state_dict(st["adam"][k])
        decays = cfg["weight_decays"]
        coef = float(cfg["logvar_bound_coef"])
        ep_upd = max(1, int(cfg["epoch_updates"]))
        t_call = time.time()
        while call["done"] < budget and st["row_ptr"][k] < self.U and not call["stopped"]:
            t0 = time.time()
            r = st["row_ptr"][k]
            st["row_ptr"][k] = r + 1
            ep = self.sched_ep[r]
            jj = self.sched_j[r]
            w = cnt[:, ep[:, None], jj].reshape(E, -1)
            if not bool((w.sum(1) > 0).any()):
                continue
            x = self.X[ep[:, None], jj].reshape(-1, DIN)
            y = self.Y[ep[:, None], jj].reshape(-1, DOUT)
            mean, lv = ens_call(ens, x)
            loss_e = nll_terms(mean, lv, y, w)
            loss = loss_e.sum() + decay_terms(list(ens.W), decays) + coef * (ens.max_logvar.sum() - ens.min_logvar.sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            lr_f = None
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(list(ens.parameters()), float(cfg["grad_clip"]))
            if self.ctx.p2 is not None:
                lr_f = self.ctx.lr_factor(int(st["used"][k]) + 1)
                for g in opt.param_groups:
                    g["lr"] = float(cfg["lr"]) * lr_f
            opt.step()
            lf = float(loss.detach())
            if not math.isfinite(lf):
                raise FloatingPointError(f"nonfinite MBCD loss (ensemble {k}, call {call_id})")
            call["done"] += 1
            st["used"][k] += 1
            st["global_step"] += 1
            gs = st["global_step"]
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize()
            self.ctx.log_train(gs, lf, {"k": k, "nll_member_mean": float(loss_e.detach().mean()),
                                        "batch_weight": float(w.sum() / E)}, time.time() - t0, component=f"M{k}",
                               **({"lr": opt.param_groups[0]["lr"], "lr_factor": lr_f} if lr_f is not None else {}))
            if call["done"] % ep_upd == 0:
                self._epoch_check(ens, call, Xh, Yh)
            if gs % self.val_every == 0:
                st["adam"][k] = opt.state_dict()
                self.validate(f"M{k}")
            if gs % self.ckpt_every == 0 or self.stop_requested():
                st["adam"][k] = opt.state_dict()
                self.save()
            if gs % 10 == 0:
                self.ctx.progress(gs, phase=f"train_{call_id}", component=f"M{k}", steps=self._planned())
            if self.stop_requested():
                self.terminate()
        if call["done"] % ep_upd != 0 and not call["stopped"] and call["done"] > 0:
            self._epoch_check(ens, call, Xh, Yh)
        with torch.no_grad():
            for p, s in zip(list(ens.W) + list(ens.b), call["snap"]):
                p.copy_(s)
            mse = self._hold_mse(ens, Xh, Yh)
            ens.elites.copy_(torch.argsort(mse)[: ens.elites.numel()].to(ens.elites.device))
        st["adam"][k] = opt.state_dict()
        rec = {"kind": "train_call", "k": k, "call": call_id, "updates": call["done"], "budget": budget,
               "early_stopped": bool(call["stopped"]), "n_train": n_tr, "n_holdout": int(ho_mask.sum()),
               "holdout_fallback_random10pct": fallback, "holdout_mse": [float(v) for v in mse.cpu()],
               "elites": [int(v) for v in ens.elites.cpu()], "seconds": time.time() - t_call,
               "used_total": int(st["used"][k])}
        st["call"] = None
        self.record(rec)
        return rec

    def _planned(self):
        return int(self.cfg["K_max"]) * self.U

    @torch.no_grad()
    def _hold_mse(self, ens, Xh, Yh):
        if Xh.shape[0] == 0:
            return torch.zeros(ens.W[0].shape[0], device=Xh.device)
        tot = torch.zeros(ens.W[0].shape[0], dtype=torch.float64, device=Xh.device)
        for s in range(0, Xh.shape[0], 65536):
            mean, _ = ens_call(ens, Xh[s:s + 65536])
            tot += ((mean - Yh[s:s + 65536]) ** 2).mean(-1).sum(-1).double()
        return (tot / Xh.shape[0]).float()

    @torch.no_grad()
    def _epoch_check(self, ens, call, Xh, Yh):
        mse = self._hold_mse(ens, Xh, Yh).cpu().tolist()
        updated = False
        for i, cur in enumerate(mse):
            best = call["best"][i]
            if (best - cur) / best > float(self.cfg["improve_tol"]):
                call["best"][i] = cur
                for p, s in zip(list(ens.W) + list(ens.b), call["snap"]):
                    s[i].copy_(p[i])
                updated = True
        call["since"] = 0 if updated else call["since"] + 1
        call["epochs"] += 1
        call["hold_mse"] = mse
        if call["since"] > int(self.cfg["max_epochs_since_update"]):
            call["stopped"] = True
        ens.elites.copy_(torch.argsort(torch.tensor(mse))[: ens.elites.numel()].to(ens.elites.device))

    @torch.no_grad()
    def _refit_scaler(self, ens, tr_mask, preserve: bool):
        Xt, _ = self.flat(tr_mask)
        mu = Xt.mean(0, keepdim=True)
        sd = Xt.std(0, unbiased=False, keepdim=True)
        sd = torch.where(sd < 1e-12, torch.ones_like(sd), sd)
        if preserve:
            old_mu, old_sd = ens.in_mu.clone(), ens.in_sigma.clone()
            W0 = ens.W[0]
            ens.b[0].add_(torch.matmul(((mu - old_mu) / old_sd), W0))
            W0.mul_((sd / old_sd).reshape(1, DIN, 1))
        ens.in_mu.copy_(mu)
        ens.in_sigma.copy_(sd)

    @torch.no_grad()
    def _init_from(self, dst, src):
        for a, b in zip(dst.parameters(), src.parameters()):
            a.copy_(b)
        dst.in_mu.copy_(src.in_mu)
        dst.in_sigma.copy_(src.in_sigma)
        dst.elites.copy_(src.elites)

    def holdout_queries(self):
        from pipeline.data import context_keys, windows
        obs_n = self.data_obs
        act_n = self.data_act
        e = np.repeat(self.hold, len(P.EVAL_STARTS))
        t = np.tile(np.asarray(P.EVAL_STARTS), len(self.hold))
        w = windows(obs_n, act_n, e, t, P.MAX_H)
        return {"ctx_obs": np.ascontiguousarray(w["ctx_obs"]), "ctx_act": np.ascontiguousarray(w["ctx_act"]),
                "fut_act": np.ascontiguousarray(w["fut_act"]), "fut_obs": np.ascontiguousarray(w["fut_obs"]),
                "keys": context_keys(w["ctx_obs"], w["ctx_act"])}

    def select_threshold(self):
        from pipeline import scorer as S
        E = self.E
        hq = self.holdout_queries()
        own, common, raw = S.frame_scales(self.stats)
        table = []
        for hv in self.cfg["h_grid"]:
            fm = MBCDForecaster(self.model, self.cfg, h_override=float(hv))
            t0 = time.time()
            out = E.forecast_arrays(fm, hq, self.device, 720)
            cm, lw = E.components(out)
            val, per_h, fin = S.selection_metric(cm, lw, hq["fut_obs"], own, common, raw)
            table.append({"h": float(hv), "holdout_metric": val, "per_h": {k: v for k, v in per_h.items()
                                                                            if k.startswith("nMSE")},
                          "finite": fin, "seconds": time.time() - t0})
            print(json.dumps({"stage": "mbcd_select_h", "h": float(hv), "holdout_metric": val}), flush=True)
        vals = [r["holdout_metric"] if math.isfinite(r["holdout_metric"]) else float("inf") for r in table]
        grid = [float(v) for v in self.cfg["h_grid"]]
        vmin = min(vals)
        tied = [i for i, v in enumerate(vals) if v == vmin]
        best = min(tied, key=lambda i: (abs(math.log(grid[i] / float(self.cfg["cusum_threshold_build"]))), -grid[i]))
        self.model.h_test.fill_(grid[best])
        return table, grid[best]

    def run(self, data):
        self.data_obs, self.data_act = data.obs_n, data.act_n
        cfg, st, model = self.cfg, self.st, self.model
        K = int(cfg["K_max"])
        init_budget = max(1, int(round(float(cfg["init_frac"]) * self.U)))
        all_hold = np.zeros((self.n_ep, self.Tn), bool)
        all_hold[self.is_hold] = True
        all_train = ~all_hold
        while st["stage"] != "done":
            stage = st["stage"]
            if stage == "m0":
                self.train_call(0, "m0", all_train, all_hold, init_budget, refit_scaler=True)
                Xh, Yh = self.flat(all_hold)
                _, vm, _ = library_stats(model, 1, Xh, Yh, cfg)
                ms = float(np.percentile(vm[:, 0].cpu().numpy(), float(cfg["max_std_percentile"])))
                model.max_std.fill_(ms)
                model.n_active.fill_(1)
                self.record({"kind": "m0_done", "max_std": ms})
                st["stage"] = "build"
                self.validate("M0_final")
                self.save()
            elif stage == "build":
                r = int(st["r"])
                if r >= K:
                    st["stage"] = "refine"
                    continue
                if st["call"] is None or st["call"].get("id") != f"build{r}":
                    lab, cnt, counts = self.detector_labels(r, float(cfg["cusum_threshold_build"]), False)
                    st["labels"] = lab
                    n_novel = int(((lab == -1) & all_train).sum())
                    self.record({"kind": "build_pass", "r": r, "alarms": cnt, "label_counts": counts,
                                 "n_novel_train": n_novel})
                    if n_novel < int(cfg["min_novel"]):
                        self.record({"kind": "build_stop", "r": r, "reason": f"novel {n_novel} < {cfg['min_novel']}"})
                        st["stage"] = "refine"
                        self.save()
                        continue
                    if cfg["library_init"] == "warm_m0":
                        self._init_from(model.ens[r], model.ens[0])
                    st["adam"][r] = None
                    model.n_active.fill_(r + 1)
                lab = st["labels"]
                self.train_call(r, f"build{r}", (lab == -1) & all_train, (lab == -1) & all_hold, init_budget,
                                refit_scaler=True)
                st["r"] = r + 1
                self.validate(f"M{r}_final")
                self.save()
            elif stage == "refine":
                i = int(st["ref_i"])
                if i >= int(cfg["refine_passes"]):
                    st["stage"] = "select"
                    continue
                na = int(model.n_active)
                if st["ref_labels"] is None:
                    lab, cnt, counts = self.detector_labels(na, float(cfg["cusum_threshold_build"]), True)
                    st["ref_labels"] = lab
                    self.record({"kind": "refine_pass", "i": i, "alarms": cnt, "label_counts": counts})
                lab = st["ref_labels"]
                for k in range(int(st["ref_k"]), na):
                    left = self.U - int(st["used"][k])
                    budget = left // (int(cfg["refine_passes"]) - i)
                    trm = (lab == k) & all_train
                    if budget > 0 and int(trm.sum()) >= int(cfg["min_refine"]):
                        self.train_call(k, f"refine{i}_{k}", trm, (lab == k) & all_hold, budget, refit_scaler=True)
                    else:
                        self.record({"kind": "refine_skip", "k": k, "budget": budget, "n": int(trm.sum())})
                    st["ref_k"] = k + 1
                    self.save()
                st["ref_i"] = i + 1
                st["ref_k"] = 0
                st["ref_labels"] = None
                self.validate(f"refine{i}_final")
                self.save()
            elif stage == "select":
                table, hsel = self.select_threshold()
                self.record({"kind": "select", "h_test": hsel, "table": table})
                st["stage"] = "done"
                self.validate("final")
                self.save()
        lib = {"schema": "mbcd_library_v1", "n_active": int(model.n_active), "h_test": float(model.h_test),
               "max_std": float(model.max_std), "updates_used": st["used"], "U": self.U,
               "holdout_episodes": self.hold.tolist(), "history": st["history"], "disclosures": DISCLOSURES}
        R.atomic_json(self.ctx.p("LIBRARY.json"), lib)
        return lib


def custom_train(cfg, seed, out_dir, device, stats, schedule, data):
    tr = _Trainer(cfg, seed, out_dir, device, stats, schedule, data)
    tr.try_resume()
    lib = tr.run(data)
    tr.ctx.complete({"custom_train": True, "n_active": lib["n_active"], "h_test": lib["h_test"],
                     "max_std": lib["max_std"], "updates_used": lib["updates_used"],
                     "global_updates": int(tr.st["global_step"])})
    print(json.dumps({"stage": "complete", "best_step": tr.ctx.best_step, "best_val": tr.ctx.best_val,
                      "wall_s": round(tr.ctx.wall(), 1), "n_active": lib["n_active"], "h_test": lib["h_test"]}),
          flush=True)
