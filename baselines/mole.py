from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from pipeline import registry as P
from pipeline import run_io as R
from pipeline.data import TrainStats

from . import grbal as G

NAME = "mole"
FAMILY = "baseline"
STORED_SIGMA2 = G.DOUT

DISCLOSURES = [
    "M1 reimplementation from the paper (arXiv 1812.07671); no official code exists",
    "M2 theta* = second-order MAML (GrBAL code path) with eta 0.01, outer Adam 1e-3 times the protocol learning-rate multiplier, K 16 (D_tr 16 / D_val next 16); 128 tasks per update from the shared schedule windows; MLP with 3 ReLU hidden layers of budget-solved width (paper 3x500); offline on TRAIN; no native clipping (the shared training protocol clips the global gradient norm at 1.0); protocol VAL selection",
    "M3 online EM restarted from {theta*} per query over the 64 context transitions; functional per-query fast weights",
    "M4 per-dimension sigma^2 = k-shot residual variance of theta* on VAL context-region tasks, floor 1e-4, recalibrated before each VAL evaluation and stored as a buffer",
    "M5 M-step with MSE-normalised gradient (beta 0.01) instead of grad[-log p]",
    "M6 new-task candidate adapted from theta* on the 16 transitions preceding x_t (theta* itself for t<=16); added iff largest posterior and < T_max=5 tasks (cap not in paper); else dropped and posteriors renormalised",
    "M7 at t=1 the single task theta* takes the first transition (no candidate)",
    "M8 T* = argmax likelihood under updated parameters at the last context step; forecast = 40-step rollout of theta(T*) (K=1)",
    "M9 budget: trainable parameters + one stored sigma^2 per observation dimension within 700,000 +-1%; Delta-normaliser constants are uncounted buffers",
    "M10 fixed-size padded evaluation chunks and fixed T_max slots -> bitwise batch/order-invariant outputs",
]


def default_config(budget_class: str = "baseline", **variant) -> dict:
    if budget_class != "baseline":
        raise ValueError(f"{NAME} is a compared model; budget_class must be 'baseline', got {budget_class!r}")
    cfg = dict(
        model=NAME,
        budget_class=budget_class,
        n_hidden=3,
        hidden=G.solve_hidden(P.BUDGET_BASELINE, extra=STORED_SIGMA2),
        lr=1e-3,
        adam_betas=[0.9, 0.999],
        adam_eps=1e-8,
        grad_clip=None,
        inner_lr=1e-2,
        adapt_m=16,
        task_len=32,
        task_offsets=[0, 24, 48, 72],
        beta=1e-2,
        alpha_crp=1.0,
        T_max=5,
        sigma2_floor=1e-4,
        em_dynamic_slots=False,
        eval_chunk=48,
        calib_chunk=96,
    )
    return G.apply_variant(cfg, variant)


def build(cfg: dict, seed: int, device, stats: TrainStats) -> G.MetaMLP:
    return G._build(cfg, seed, device, stats, with_sigma2=True, budget_extra=STORED_SIGMA2)


make_optimizer = G.make_optimizer
train_step = G.train_step


def parameter_report(model: G.MetaMLP, cfg: dict | None = None) -> dict:
    return G.report(model, int((cfg or {}).get("T_max", 5)), STORED_SIGMA2,
                    "sigma2 (17) counted as stored non-gradient; delta_mean/delta_std are normaliser constants (not counted)")


def calibrate_sigma2(model: G.MetaMLP, cfg: dict, obs_n: np.ndarray, act_n: np.ndarray) -> dict:
    net, p0 = model.net, model.theta(detach=True)
    M, L = int(cfg["adapt_m"]), int(cfg["task_len"])
    xs, ys = G.stacked_tasks(model, obs_n, act_n, L)
    tot = torch.zeros(G.DOUT, dtype=torch.float64, device=xs.device)
    n = 0
    ch = int(cfg.get("calib_chunk", 96))
    for s in range(0, xs.shape[0], ch):
        with torch.enable_grad():
            pa = G.adapt_k_shot(net, p0, xs[s:s + ch, :M], ys[s:s + ch, :M], float(cfg["inner_lr"]))
        with torch.no_grad():
            r = G.batched_apply(net, pa, xs[s:s + ch, M:]) - ys[s:s + ch, M:]
            tot += (r * r).sum((0, 1)).double()
            n += r.shape[0] * r.shape[1]
    s2 = (tot / n).clamp_min(float(cfg["sigma2_floor"])).float()
    with torch.no_grad():
        model.sigma2.copy_(s2)
    return {"sigma2": s2.cpu().tolist(), "n_tasks": int(xs.shape[0]), "n_points": int(n),
            "source": "VAL transitions 0..191 (union of VAL query contexts); k-shot residuals of theta*"}


def after_step(model: G.MetaMLP, step: int, cfg: dict) -> None:
    try:
        from pipeline import trainer as TR
        ctx = TR.current_context()
    except (ImportError, RuntimeError):
        return
    s = int(step) + 1
    if not (s % ctx.val_every == 0 or s == ctx.steps):
        return
    from pipeline import evaluate as E
    sq = E.split_queries("val", E.get_stats())
    t0 = time.time()
    info = calibrate_sigma2(model, cfg, sq["obs_n"], sq["act_n"])
    info.update({"step": s, "seconds": time.time() - t0})
    R.append_jsonl(ctx.p("SIGMA2.jsonl"), info)
    print(json.dumps({"stage": "mole_sigma2", "step": s, "min": min(info["sigma2"]), "max": max(info["sigma2"]),
                      "seconds": round(info["seconds"], 2)}), flush=True)


def _mse_sse(net, p, x, y, m):
    r = functional_call(net, p, (x,)) - y
    sse = ((r * r) * m.unsqueeze(-1)).sum(0)
    return sse.sum() / (m.sum().clamp_min(1.0) * y.shape[-1]), sse


def online_em(net, p0: dict, X: torch.Tensor, Y: torch.Tensor, sigma2: torch.Tensor, cfg: dict):

    C, L, _ = X.shape
    K, Tm = int(cfg["adapt_m"]), int(cfg["T_max"])
    beta, alpha = float(cfg["beta"]), float(cfg["alpha_crp"])
    dev, dt = X.device, X.dtype
    pad = 2 * K
    Xp = torch.cat([X.new_zeros(C, pad, X.shape[2]), X], 1)
    Yp = torch.cat([Y.new_zeros(C, pad, Y.shape[2]), Y], 1)
    Mp = torch.cat([X.new_zeros(pad), X.new_ones(L)])
    inv_s2 = (1.0 / sigma2).to(dt)
    log_norm = float(-0.5 * torch.log(2.0 * math.pi * sigma2.double()).sum())
    log_alpha = math.log(alpha)

    f_exist = vmap(vmap(grad(lambda pp, xx, yy, mm: _mse_sse(net, pp, xx, yy, mm), has_aux=True),
                        in_dims=(0, None, None, None)), in_dims=(0, 0, 0, None))
    f_new_grad = vmap(grad(lambda pp, xx, yy, mm: _mse_sse(net, pp, xx, yy, mm)[0]), in_dims=(None, 0, 0, None))
    f_sse = vmap(lambda pp, xx, yy, mm: _mse_sse(net, pp, xx, yy, mm)[1], in_dims=(0, 0, 0, None))
    f_sse2 = vmap(vmap(lambda pp, xx, yy, mm: _mse_sse(net, pp, xx, yy, mm)[1], in_dims=(0, None, None, None)),
                  in_dims=(0, 0, 0, None))

    theta = {k: v.unsqueeze(0).unsqueeze(0).expand(C, Tm, *v.shape).clone() for k, v in p0.items()}
    active = torch.zeros(C, Tm, dtype=torch.bool, device=dev)
    active[:, 0] = True
    counts = torch.zeros(C, Tm, dtype=dt, device=dev)
    n_created = torch.zeros(C, dtype=torch.long, device=dev)
    first_create = torch.full((C,), -1, dtype=torch.long, device=dev)
    dyn = bool(cfg.get("em_dynamic_slots", False))
    n_act = 1 if dyn else Tm
    xb = yb = mb = None
    for tau in range(1, L + 1):
        b0 = pad + tau - K
        xb, yb, mb = Xp[:, b0:b0 + K], Yp[:, b0:b0 + K], Mp[b0:b0 + K]
        nb = float(min(tau, K))
        th = {k: v[:, :n_act] for k, v in theta.items()}
        g, sse = f_exist(th, xb, yb, mb)
        act = active[:, :n_act]
        ll = (-0.5 * (sse * inv_s2).sum(-1) + nb * log_norm).masked_fill(~act, float("-inf"))
        created = None
        if tau == 1:
            pi = torch.zeros(C, n_act, dtype=dt, device=dev)
            pi[:, 0] = 1.0
        else:
            n0 = pad + tau - 2 * K
            xn, yn, mn = Xp[:, n0:n0 + K], Yp[:, n0:n0 + K], Mp[n0:n0 + K]
            gn = f_new_grad(p0, xn, yn, mn)
            th_new = {k: p0[k] - beta * gn[k] for k in p0}
            ll_new = -0.5 * (f_sse(th_new, xb, yb, mb) * inv_s2).sum(-1) + nb * log_norm
            log_den = torch.log(counts.sum(1) + alpha)
            lp_ex = torch.log(counts[:, :n_act]).masked_fill(~act, float("-inf")) - log_den[:, None]
            logits_ex = lp_ex + ll
            logit_new = log_alpha - log_den + ll_new
            full = torch.softmax(torch.cat([logits_ex, logit_new[:, None]], 1), 1)
            created = (active.sum(1) < Tm) & (full[:, -1] > full[:, :-1].max(1).values)
            pi = torch.where(created[:, None], full[:, :-1], torch.softmax(logits_ex, 1))
            pi_new = torch.where(created, full[:, -1], torch.zeros_like(full[:, -1]))
        for k in theta:
            w = pi.view((C, n_act) + (1,) * (theta[k].dim() - 2))
            theta[k][:, :n_act].sub_(beta * w * g[k])
        counts[:, :n_act] += pi
        if created is not None and bool(created.any()):
            slot = active.sum(1)
            onehot = (torch.arange(Tm, device=dev)[None, :] == slot[:, None]) & created[:, None]
            for k in theta:
                sel = onehot.view((C, Tm) + (1,) * (theta[k].dim() - 2))
                theta[k] = torch.where(sel, th_new[k].unsqueeze(1), theta[k])
            active = active | onehot
            counts = counts + onehot.to(dt) * pi_new[:, None]
            first_create = torch.where(created & (first_create < 0), torch.full_like(first_create, tau), first_create)
            n_created += created.long()
            if dyn:
                n_act = int(active.sum(1).max())
    th = {k: v[:, :n_act] for k, v in theta.items()}
    ll_fin = (-0.5 * (f_sse2(th, xb, yb, mb) * inv_s2).sum(-1)).masked_fill(~active[:, :n_act], float("-inf"))
    tstar = ll_fin.argmax(1)
    idx = torch.arange(C, device=dev)
    theta_star = {k: v[idx, tstar] for k, v in theta.items()}
    diag = {"n_tasks": active.sum(1), "t_star": tstar, "n_created": n_created, "first_create_step": first_create,
            "count_tstar": counts[idx, tstar]}
    return theta_star, diag


class MOLeForecaster(G.AdaptForecasterBase):
    def __init__(self, model: G.MetaMLP, cfg: dict):
        super().__init__(model, cfg, NAME)

    def parameter_report(self) -> dict:
        return parameter_report(self.model, self.cfg)

    def _forecast_chunk(self, co, ca, fa):
        model, cfg = self.model, self.cfg
        if not bool((model.sigma2 > 0).all()):
            raise RuntimeError("MOLe sigma^2 is not calibrated (train with pipeline.trainer, whose after_step hook calibrates it)")
        net, dm, ds = model.net, model.delta_mean, model.delta_std
        p0 = model.theta(detach=True)
        x, y = G.transitions(co, ca, dm, ds)
        with torch.enable_grad():
            th_star, d = online_em(net, p0, x, y, model.sigma2, cfg)
        with torch.no_grad():
            pred = G.rollout(net, th_star, co[:, -1], fa, dm, ds)
            M = int(cfg["adapt_m"])
            pre = ((functional_call(net, p0, (x[:, -M:],)) - y[:, -M:]) ** 2).mean((1, 2))
            post = ((G.batched_apply(net, th_star, x[:, -M:]) - y[:, -M:]) ** 2).mean((1, 2))
        diag = {k: v.cpu().numpy() for k, v in d.items()}
        diag["adapt_pre_mse"] = pre.cpu().numpy()
        diag["adapt_post_mse"] = post.cpu().numpy()
        return pred, diag


def forecaster(model: G.MetaMLP, cfg: dict) -> MOLeForecaster:
    return MOLeForecaster(model, cfg)
