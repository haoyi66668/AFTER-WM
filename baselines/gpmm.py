from __future__ import annotations

import hashlib
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

NAME = "gpmm"
FAMILY = "baseline"
DIN = P.OBS_DIM + P.ACT_DIM
DOUT = P.OBS_DIM
LOG2PI = math.log(2.0 * math.pi)
HOLDOUT_SEED = 20261012
HOLDOUT_FRAC = 0.1
N_HYPER = DIN + 3

DISCLOSURES = [
    "G1 offline library from one streaming pass over TRAIN-train (no MPC/reward); K_max 5; per-query restart with first-point prior = R column sums",
    "G2 assignment likelihood = noisy posterior predictive averaged over the outputs (official latent; option assign_lik='latent')",
    "G3 MLL refit (5 Adam steps lr 0.1, early stopping) at 25/50/100/200/400 points, then every 500 added points, and at each sparsification, on <=1000 random points of the component",
    "G4 sparsification at n > M_cap (solved for the parameter budget): keep 90%, best of 50 random subsets by the exact conditional log-likelihood of the dropped points (official ELBO)",
    "G5 fp64 Cholesky; fp32 whitening matrix and large GEMMs; fp64 small Schur complements; jitter 1e-6 outputscale; numerical noise floor 1e-3 x outputscale (also in the MLL); guard noisy variance >= 0.5 noise (counted)",
    "G6 one trainer step = one streamed episode; VAL every 32 episodes + end; no SGD learning rate (lr_sweep False)",
    "G7 query-local new component with data-driven prior hyper-parameters (no per-query MLL fit); at most one",
    "G8 padded fixed-size eval chunks; weight-derived factor cache keyed by a state hash",
]


def stored_values(k_max: int, m_cap: int) -> int:
    return k_max * m_cap * (DIN + DOUT) + k_max * k_max + (1 + 3 * DOUT) + DOUT


def total_values(k_max: int, m_cap: int) -> int:
    return k_max * DOUT * N_HYPER + stored_values(k_max, m_cap)


def solve_mcap(target: int, k_max: int = 5) -> int:
    return min(range(16, 100000), key=lambda m: (abs(total_values(k_max, m) - target), m))


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
        lr_sweep=False,
        K_max=5,
        M_cap=solve_mcap(P.BUDGET_BASELINE, 5),
        keep_frac=0.9,
        alpha=1.5,
        window_prob=0.001,
        self_prob=1.0,
        burnin=5,
        gp_lr=0.1,
        gp_iter=5,
        early_stop_tol=1e-4,
        refit_every=500,
        refit_early=[25, 50, 100, 200, 400],
        fit_max_points=1000,
        init_noise_frac=0.01,
        noise_floor_frac=1e-4,
        noise_os_floor=1e-3,
        init_subsample=4000,
        assign_lik="noisy",
        factor_dtype="float32",
        jitter=1e-6,
        sparsify_candidates=50,
        stream_block=64,
        out_group=4,
        eval_chunk=240,
        test_local_new=1,
        val_every_episodes=32,
        ckpt_every_episodes=32,
        grad_clip=None,
    )
    for k, v in variant.items():
        if k not in cfg and not k.startswith("dev_"):
            raise KeyError(f"unknown config key {k!r} (GPMM has no SGD learning rate; lr_sweep is False)")
        cfg[k] = v
    if cfg["assign_lik"] not in ("noisy", "latent"):
        raise ValueError("assign_lik must be 'noisy' or 'latent'")
    if cfg["factor_dtype"] not in ("float32", "float64"):
        raise ValueError("factor_dtype must be 'float32' or 'float64'")
    return cfg


def _fdtype(cfg):
    return torch.float64 if cfg["factor_dtype"] == "float64" else torch.float32


def inv_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return torch.where(x > 20, x, torch.log(torch.expm1(x.clamp_min(1e-12))))


def noise_transform(raw_noise, os_, noise_lb, os_floor: float):
    return noise_lb + float(os_floor) * os_ + F.softplus(raw_noise)


class GPMMLibrary(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        K, M = int(cfg["K_max"]), int(cfg["M_cap"])
        f64 = torch.float64
        self.raw_ls = nn.Parameter(torch.zeros(K, DOUT, DIN, dtype=f64))
        self.raw_os = nn.Parameter(torch.zeros(K, DOUT, dtype=f64))
        self.raw_noise = nn.Parameter(torch.zeros(K, DOUT, dtype=f64))
        self.const = nn.Parameter(torch.zeros(K, DOUT, dtype=f64))
        self.register_buffer("X", torch.zeros(K, M, DIN))
        self.register_buffer("Y", torch.zeros(K, M, DOUT))
        self.register_buffer("n_points", torch.zeros(K, dtype=torch.long))
        self.register_buffer("n_active", torch.zeros((), dtype=torch.long))
        self.register_buffer("R", torch.zeros(K, K, dtype=f64))
        self.register_buffer("prior_ls", torch.ones((), dtype=f64))
        self.register_buffer("prior_os", torch.ones(DOUT, dtype=f64))
        self.register_buffer("prior_noise", torch.full((DOUT,), 0.01, dtype=f64))
        self.register_buffer("prior_const", torch.zeros(DOUT, dtype=f64))
        self.register_buffer("noise_lb", torch.full((DOUT,), 1e-4, dtype=f64))
        self.noise_os_floor = float(cfg.get("noise_os_floor", 0.0))

    def hyper(self, k: int):
        ls = F.softplus(self.raw_ls[k])
        os_ = F.softplus(self.raw_os[k])
        noise = noise_transform(self.raw_noise[k], os_, self.noise_lb, self.noise_os_floor)
        return ls, os_, noise, self.const[k]

    def prior_hyper(self):
        return self.prior_ls.expand(DOUT, DIN), self.prior_os, self.prior_noise, self.prior_const


def build(cfg: dict, seed: int, device, stats: TrainStats) -> GPMMLibrary:
    torch.manual_seed(int(seed))
    model = GPMMLibrary(cfg)
    if not cfg.get("dev_allow_offbudget"):
        chk = PC.budget_check(total_values(int(cfg["K_max"]), int(cfg["M_cap"])), "baseline")
        if not chk["within_budget"]:
            raise RuntimeError(f"parameter budget violated: {chk}")
    return model.to(device)


def make_optimizer(model: GPMMLibrary, cfg: dict):
    return torch.optim.Adam(model.parameters(), lr=float(cfg["gp_lr"]))


def train_step(model, batch, step, cfg):
    raise RuntimeError("GPMM has no minibatch training step; pipeline.trainer uses custom_train")


def parameter_report(model: GPMMLibrary, cfg: dict) -> dict:
    K, M = int(cfg["K_max"]), int(cfg["M_cap"])
    stored = stored_values(K, M)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n = model.n_points.cpu().tolist()
    return PC.make_report(model, stored_nongrad=stored,
                          groups={"gp_hyperparameters": trainable},
                          stored_nongrad_groups={"inducing_X_capacity": K * M * DIN, "inducing_Y_capacity": K * M * DOUT,
                                                 "sticky_R": K * K, "data_driven_prior": 1 + 3 * DOUT,
                                                 "noise_floor": DOUT},
                          extra={"budget": PC.budget_check(trainable + stored, "baseline"),
                                 "buffers_note": "n_points / n_active are bookkeeping (not counted); stored data is "
                                                 "counted at full capacity K_max x M_cap",
                                 "points_used": n, "n_active": int(model.n_active),
                                 "stored_values_used": int(sum(n) * (DIN + DOUT)),
                                 "per_query_state": {"note": "per-query Schur factors over <= 64 context points per "
                                                             "component and a copy of R (state, not counted)"}})


def kern(A, B, ls, os_):
    a = A / ls.unsqueeze(-2)
    b = B / ls.unsqueeze(-2)
    d2 = (a * a).sum(-1).unsqueeze(-1) + (b * b).sum(-1).unsqueeze(-2) - 2.0 * torch.matmul(a, b.transpose(-1, -2))
    return os_.unsqueeze(-1).unsqueeze(-1) * torch.exp(-0.5 * d2.clamp_min(0.0))


def eff_noise(noise, os_, jitter):
    return noise + float(jitter) * os_


def factorize(X, Y, ls, os_, noise, c, jitter: float, fdtype, group: int):
    n = int(X.shape[0])
    dev = X.device
    W = torch.zeros(DOUT, n, n, dtype=fdtype, device=dev)
    beta = torch.zeros(DOUT, n, dtype=torch.float64, device=dev)
    if n == 0:
        return W, beta
    X64, Y64 = X.double(), Y.double()
    ne = eff_noise(noise, os_, jitter).double()
    for g0 in range(0, DOUT, group):
        sl = slice(g0, min(DOUT, g0 + group))
        K = kern(X64, X64, ls[sl].double(), os_[sl].double())
        K.diagonal(dim1=-2, dim2=-1).add_(ne[sl].unsqueeze(-1))
        L, info = torch.linalg.cholesky_ex(K)
        extra = 1e-6
        while bool((info > 0).any()):
            if extra > 1e-2:
                raise RuntimeError("GP Cholesky failed even with jitter 1e-2 x outputscale")
            K.diagonal(dim1=-2, dim2=-1).add_(extra * os_[sl].double().unsqueeze(-1))
            L, info = torch.linalg.cholesky_ex(K)
            extra *= 10
        del K
        r = (Y64[:, sl] - c[sl].double()).T.unsqueeze(-1)
        beta[sl] = torch.linalg.solve_triangular(L, r, upper=False)[..., 0]
        eye = torch.eye(n, dtype=fdtype, device=dev).expand(L.shape[0], n, n)
        W[sl] = torch.linalg.solve_triangular(L.to(fdtype), eye, upper=False)
        del L
    return W, beta


def ctx_stats(W, beta, XD, Xc, ls, os_, c, group: int, fdtype, return_V: bool = False):
    Q, L, _ = Xc.shape
    n = int(XD.shape[0])
    dev = Xc.device
    mu = torch.empty(Q, DOUT, L, dtype=torch.float64, device=dev)
    C = torch.empty(Q, DOUT, L, L, dtype=torch.float64, device=dev)
    Vs = [] if return_V else None
    flat = Xc.reshape(Q * L, DIN).to(fdtype)
    XDf = XD.to(fdtype)
    for g0 in range(0, DOUT, group):
        sl = slice(g0, min(DOUT, g0 + group))
        g = sl.stop - sl.start
        lsg, osg = ls[sl].to(fdtype), os_[sl].to(fdtype)
        Kqq = kern(Xc.to(fdtype).unsqueeze(1), Xc.to(fdtype).unsqueeze(1), lsg.unsqueeze(0), osg.unsqueeze(0))
        if n > 0:
            Kdq = kern(XDf, flat, lsg, osg)
            V = torch.matmul(W[sl], Kdq)
            del Kdq
            m_ = torch.matmul(beta[sl].to(fdtype).unsqueeze(1), V)[:, 0]
            mu[:, sl] = (c[sl].double().unsqueeze(-1) + m_.double()).reshape(g, Q, L).permute(1, 0, 2)
            Vq = V.reshape(g, n, Q, L)
            VV = torch.einsum("gnqi,gnqj->qgij", Vq, Vq)
            C[:, sl] = (Kqq - VV).double()
            if return_V:
                Vs.append(V)
            del V, Vq, VV
        else:
            mu[:, sl] = c[sl].double().view(1, g, 1).expand(Q, g, L)
            C[:, sl] = Kqq.double()
    if return_V:
        return mu, C, (torch.cat(Vs, 0) if Vs else None)
    return mu, C


def append_factor(W_nn, V_A, L22, fdtype):
    a = int(L22.shape[-1])
    eye = torch.eye(a, dtype=torch.float64, device=L22.device).expand(L22.shape[0], a, a)
    U = torch.linalg.solve_triangular(L22, eye, upper=False)
    rows = None
    if W_nn is not None and W_nn.shape[-1] > 0:
        rows = -torch.matmul(U.to(fdtype), torch.matmul(V_A.transpose(1, 2), W_nn))
    return rows, U.to(fdtype)


def log_normal_mean(y, m, var):
    return (-0.5 * (LOG2PI + torch.log(var) + (y - m) ** 2 / var)).mean(-1)


def assign_sequence(mu, C, y, noise, lpn, Rm, exists, npts, newest, prev, next_slot, cfg, prior_noise=None,
                    stop_fn=None):

    Q, Kc, D, B = mu.shape
    dev = mu.device
    f64 = torch.float64
    L = torch.eye(B, dtype=f64, device=dev).expand(Q, Kc, D, B, B).clone()
    u = torch.zeros(Q, Kc, D, B, dtype=f64, device=dev)
    member = torch.zeros(Q, Kc, B, dtype=torch.bool, device=dev)
    assign = torch.full((Q, B), -1, dtype=torch.long, device=dev)
    lp_ass = torch.zeros(Q, B, dtype=f64, device=dev)
    created = torch.zeros(Q, B, dtype=torch.bool, device=dev)
    Rm, exists, npts = Rm.clone(), exists.clone(), npts.clone()
    newest, prev, next_slot = newest.clone(), prev.clone(), next_slot.clone()
    alpha = float(cfg["alpha"])
    log_alpha = math.log(alpha)
    burnin = int(cfg["burnin"])
    wp, sp = float(cfg["window_prob"]), float(cfg["self_prob"])
    latent = cfg["assign_lik"] == "latent"
    qi = torch.arange(Q, device=dev)
    ark = torch.arange(Kc, device=dev)
    ninf = torch.tensor(float("-inf"), dtype=f64, device=dev)
    n_done = B
    n_guard = torch.zeros(Q, Kc, dtype=torch.long, device=dev)
    for i in range(B):
        ci = C[..., :, i] * member.unsqueeze(2).to(f64)
        lvec = torch.linalg.solve_triangular(L, ci.unsqueeze(-1), upper=False)[..., 0]
        m_i = mu[..., i] + (lvec * u).sum(-1)
        vlat = C[..., i, i] - (lvec * lvec).sum(-1)
        vnoisy = vlat + noise
        guard = vnoisy < 0.5 * noise
        n_guard += guard.long().sum(-1)
        vnoisy = torch.where(guard, 0.5 * noise.expand_as(vnoisy), vnoisy)
        if latent:
            fallback = prior_noise if prior_noise is not None else noise
            var = torch.where(vlat > 0, vlat, fallback.expand_as(vlat))
        else:
            var = vnoisy.clamp_min(1e-12)
        yi = y[:, i].unsqueeze(1)
        lp = log_normal_mean(yi, m_i, var)
        has_prev = prev >= 0
        prow = Rm[qi, prev.clamp_min(0)]
        wts = torch.where(has_prev.unsqueeze(1), prow, (Rm * exists.unsqueeze(2).to(f64)).sum(1))
        log_old = torch.where(exists & (wts > 0), torch.log(wts.clamp_min(1e-300)) + lp, ninf)
        nb = npts[qi, newest.clamp_min(0)]
        burn_ok = (newest < 0) | (nb >= burnin)
        can_new = (next_slot < Kc) & burn_ok
        ns = next_slot.clamp(max=Kc - 1)
        log_new = torch.where(can_new, log_alpha + lpn[:, i], ninf)
        full = torch.cat([log_old, log_new.unsqueeze(1)], 1)
        ok = torch.isfinite(full).any(1)
        ksel = full.argmax(1)
        is_new = (ksel == Kc) & ok
        kstar = torch.where(is_new, ns, ksel.clamp(max=Kc - 1))
        kstar = torch.where(ok, kstar, torch.where(has_prev, prev, torch.zeros_like(prev)))
        rho_full = torch.softmax(full, 1)
        rho_old = torch.softmax(log_old, 1)
        rho_old = torch.where(torch.isfinite(rho_old), rho_old, torch.zeros_like(rho_old))
        add_new = rho_full[:, :Kc] + F.one_hot(ns, Kc).to(f64) * rho_full[:, Kc:]
        add_row = torch.where(is_new.unsqueeze(1), add_new, rho_old)
        upd = (has_prev & ok).to(f64)
        Rm = Rm + F.one_hot(prev.clamp_min(0), Kc).to(f64).unsqueeze(2) * (add_row * upd.unsqueeze(1)).unsqueeze(1)
        ex_after = exists | (F.one_hot(ns, Kc).bool() & is_new.unsqueeze(1))
        init_row = torch.where(ark.unsqueeze(0) == ns.unsqueeze(1), torch.full_like(add_row, sp),
                               torch.where(ex_after, torch.full_like(add_row, wp), torch.zeros_like(add_row)))
        rowsel = (F.one_hot(ns, Kc).bool() & is_new.unsqueeze(1)).unsqueeze(2)
        Rm = torch.where(rowsel, init_row.unsqueeze(1), Rm)
        exists = ex_after
        newest = torch.where(is_new, ns, newest)
        next_slot = torch.where(is_new, next_slot + 1, next_slot)
        sel = F.one_hot(kstar, Kc).bool()
        lii = vnoisy.clamp_min(1e-300).sqrt()
        new_row = lvec.clone()
        new_row[..., i] = lii
        cur_row = L[..., i, :]
        L[..., i, :] = torch.where(sel.view(Q, Kc, 1, 1), new_row, cur_row)
        r_i = y[:, i].unsqueeze(1) - mu[..., i]
        ui = (r_i - (lvec * u).sum(-1)) / lii
        u[..., i] = torch.where(sel.unsqueeze(-1), ui, u[..., i])
        member[..., i] = sel
        npts = npts + sel.long()
        assign[:, i] = kstar
        lp_ass[:, i] = lp.gather(1, kstar.unsqueeze(1))[:, 0]
        created[:, i] = is_new
        prev = kstar
        if stop_fn is not None and stop_fn(i, int(kstar[0]), npts[0]):
            n_done = i + 1
            break
    return {"assign": assign, "L": L, "u": u, "member": member, "R": Rm, "exists": exists, "npts": npts,
            "newest": newest, "prev": prev, "next_slot": next_slot, "n_done": n_done, "lp": lp_ass,
            "created": created, "n_guard": n_guard}


def mll_loss(raw, Xs, Ys, noise_lb, jitter, os_floor: float = 0.0):
    raw_ls, raw_os, raw_noise, const = raw
    ls, os_ = F.softplus(raw_ls), F.softplus(raw_os)
    noise = noise_transform(raw_noise, os_, noise_lb, os_floor)
    n = Xs.shape[0]
    K = kern(Xs, Xs, ls, os_)
    K = K + torch.diag_embed((eff_noise(noise, os_, jitter)).unsqueeze(-1).expand(DOUT, n))
    L, info = torch.linalg.cholesky_ex(K)
    if bool((info > 0).any()):
        K = K + torch.diag_embed((1e-4 * os_).unsqueeze(-1).expand(DOUT, n))
        L = torch.linalg.cholesky(K)
    r = (Ys - const).T.unsqueeze(-1)
    a = torch.linalg.solve_triangular(L, r, upper=False)[..., 0]
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    ll = -0.5 * (a * a).sum(-1) - 0.5 * logdet - 0.5 * n * LOG2PI
    return -(ll.sum() / n)


def state_fingerprint(model: GPMMLibrary) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class GPMMForecaster:
    name = NAME
    family = FAMILY
    budget_class = "baseline"
    needs_grad_at_eval = False
    stochastic = False

    def __init__(self, model: GPMMLibrary, cfg: dict):
        self.model, self.cfg = model, cfg
        self.fd = _fdtype(cfg)
        self.Kl = int(model.n_active)
        self.fac = self._factors()

    def parameter_report(self) -> dict:
        return parameter_report(self.model, self.cfg)

    @torch.no_grad()
    def _factors(self):
        model, cfg = self.model, self.cfg
        fp = state_fingerprint(model)
        cache = getattr(model, "_gpmm_cache", None)
        if cache is not None and cache.get("fp") == fp and cache.get("dtype") == str(self.fd):
            return cache["fac"]
        fac = []
        for k in range(self.Kl):
            n = int(model.n_points[k])
            ls, os_, noise, c = (t.detach() for t in model.hyper(k))
            W, beta = factorize(model.X[k, :n], model.Y[k, :n], ls, os_, noise, c, float(cfg["jitter"]), self.fd,
                                int(cfg["out_group"]))
            fac.append({"W": W, "beta": beta, "X": model.X[k, :n], "n": n})
        model._gpmm_cache = {"fp": fp, "dtype": str(self.fd), "fac": fac}
        return fac

    def forecast(self, q: QueryBatch) -> Forecast:
        return self.forecast_with_diagnostics(q)[0]

    @torch.no_grad()
    def forecast_with_diagnostics(self, q: QueryBatch):
        dev = self.model.R.device
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
        return Forecast(mean=mean, comp_mean=mean.unsqueeze(2), comp_logw=mean.new_zeros(B, 1)), diag

    def _forecast_chunk(self, co, ca, fa):
        model, cfg, fd = self.model, self.cfg, self.fd
        dev = co.device
        Q, Lc = co.shape[0], co.shape[1] - 1
        Kl = self.Kl
        nloc = min(int(cfg["test_local_new"]), int(cfg["K_max"]) - Kl) if Kl > 0 else 1
        nloc = max(0, nloc)
        Kc = Kl + nloc
        grp = int(cfg["out_group"])
        jit = float(cfg["jitter"])
        x = torch.cat([co[:, :-1], ca], -1)
        y = (co[:, 1:] - co[:, :-1]).double()
        mu = torch.empty(Q, Kc, DOUT, Lc, dtype=torch.float64, device=dev)
        C = torch.empty(Q, Kc, DOUT, Lc, Lc, dtype=torch.float64, device=dev)
        noise = torch.empty(1, Kc, DOUT, dtype=torch.float64, device=dev)
        hyp = []
        for k in range(Kl):
            ls, os_, nz, c = (t.detach() for t in model.hyper(k))
            hyp.append((ls, os_, c))
            f = self.fac[k]
            mu[:, k], C[:, k] = ctx_stats(f["W"], f["beta"], f["X"], x, ls, os_, c, grp, fd)
            noise[0, k] = eff_noise(nz, os_, jit)
        pls, pos_, pnz, pc = model.prior_hyper()
        for k in range(Kl, Kc):
            hyp.append((pls, pos_, pc))
            mu[:, k], C[:, k] = ctx_stats(None, None, x.new_zeros(0, DIN), x, pls, pos_, pc, grp, fd)
            noise[0, k] = eff_noise(pnz, pos_, jit)
        pvar = pos_ + (eff_noise(pnz, pos_, jit) if cfg["assign_lik"] == "noisy" else 0.0)
        lpn = log_normal_mean(y, pc, pvar)
        Rm = torch.zeros(Q, Kc, Kc, dtype=torch.float64, device=dev)
        Rm[:, :Kl, :Kl] = model.R[:Kl, :Kl]
        exists = torch.zeros(Q, Kc, dtype=torch.bool, device=dev)
        exists[:, :Kl] = True
        npts = torch.zeros(Q, Kc, dtype=torch.long, device=dev)
        npts[:, :Kl] = model.n_points[:Kl]
        newest = torch.full((Q,), Kl - 1, dtype=torch.long, device=dev)
        prev = torch.full((Q,), -1, dtype=torch.long, device=dev)
        next_slot = torch.full((Q,), Kl if nloc > 0 else Kc, dtype=torch.long, device=dev)
        res = assign_sequence(mu, C, y, noise, lpn, Rm, exists, npts, newest, prev, next_slot, cfg,
                              prior_noise=pnz.view(1, 1, DOUT).expand(1, Kc, DOUT))
        kstar = res["assign"][:, -1]
        gamma = torch.linalg.solve_triangular(res["L"].transpose(-1, -2), res["u"].unsqueeze(-1), upper=True)[..., 0]
        bmp = []
        flat = x.reshape(Q * Lc, DIN).to(fd)
        for k in range(Kl):
            f = self.fac[k]
            n = f["n"]
            ls, os_, c = hyp[k]
            parts = []
            for g0 in range(0, DOUT, grp):
                sl = slice(g0, min(DOUT, g0 + grp))
                g = sl.stop - sl.start
                V = torch.matmul(f["W"][sl], kern(f["X"].to(fd), flat, ls[sl].to(fd), os_[sl].to(fd)))
                psi = torch.einsum("gnqi,qgi->qgn", V.reshape(g, n, Q, Lc), gamma[:, k, sl].to(fd))
                parts.append(f["beta"][sl].to(fd).unsqueeze(0) - psi)
                del V
            bmp.append(torch.cat(parts, 1))
        z = co[:, -1]
        H = fa.shape[1]
        outs = []
        xq = x.to(fd)
        for step in range(H):
            xin = torch.cat([z, fa[:, step]], -1)
            mk = torch.empty(Q, Kc, DOUT, dtype=torch.float64, device=dev)
            for k in range(Kc):
                ls, os_, c = hyp[k]
                kA = kern(xq.unsqueeze(1), xin.to(fd).view(Q, 1, 1, DIN), ls.to(fd).unsqueeze(0),
                          os_.to(fd).unsqueeze(0))[..., 0]
                corr = (kA.double() * gamma[:, k]).sum(-1)
                if k < Kl:
                    f = self.fac[k]
                    main = []
                    for g0 in range(0, DOUT, grp):
                        sl = slice(g0, min(DOUT, g0 + grp))
                        v = torch.matmul(f["W"][sl], kern(f["X"].to(fd), xin.to(fd), ls[sl].to(fd), os_[sl].to(fd)))
                        main.append(torch.einsum("gnq,qgn->qg", v, bmp[k][:, sl]))
                    mk[:, k] = c.double() + torch.cat(main, 1).double() + corr
                else:
                    mk[:, k] = c.double() + corr
            dz = mk.gather(1, kstar.view(Q, 1, 1).expand(Q, 1, DOUT))[:, 0]
            z = z + dz.to(z.dtype)
            outs.append(z)
        pred = torch.stack(outs, 1)
        a = res["assign"]
        diag = {"final_component": kstar.cpu().numpy(), "n_switches": (a[:, 1:] != a[:, :-1]).sum(1).cpu().numpy(),
                "created_local": res["created"].any(1).cpu().numpy(),
                "first_component": a[:, 0].cpu().numpy(), "assign_loglik_mean": res["lp"].mean(1).cpu().numpy(),
                "n_guard": res["n_guard"].sum(1).cpu().numpy()}
        return pred, diag


def forecaster(model: GPMMLibrary, cfg: dict) -> GPMMForecaster:
    return GPMMForecaster(model, cfg)


class _Stream:

    def __init__(self, cfg, seed, out_dir, device, stats, data):
        from pipeline import evaluate as E
        from pipeline import trainer as TR
        self.TR, self.E = TR, E
        self.ctx = TR.current_context()
        self.cfg, self.seed, self.device, self.stats = cfg, int(seed), device, stats
        self.module = sys.modules[__name__]
        self.model = build(cfg, seed, device, stats)
        self.fd = _fdtype(cfg)
        self.obs_np, self.act_np = data.obs_n, data.act_n
        obs = torch.from_numpy(np.ascontiguousarray(data.obs_n)).to(device)
        act = torch.from_numpy(np.ascontiguousarray(data.act_n)).to(device)
        self.Xall = torch.cat([obs[:, :-1], act], -1).contiguous()
        self.Yall = (obs[:, 1:] - obs[:, :-1]).contiguous()
        self.n_ep, self.Tn = int(self.Xall.shape[0]), int(self.Xall.shape[1])
        self.hold = holdout_episodes(self.n_ep)
        mask = np.ones(self.n_ep, bool)
        mask[self.hold] = False
        self.train_eps = np.flatnonzero(mask)
        self.order = np.random.default_rng([self.seed, 11]).permutation(self.train_eps)
        self.n_stream = int(min(len(self.order), int(self.ctx.steps)))
        self.val_every = max(1, min(int(cfg["val_every_episodes"]), int(self.ctx.val_every)))
        self.ckpt_every = max(1, min(int(cfg["ckpt_every_episodes"]), int(self.ctx.ckpt_every)))
        K, M = int(cfg["K_max"]), int(cfg["M_cap"])
        self.K, self.M = K, M
        cap = M + 1
        self.Xw = torch.zeros(K, cap, DIN, device=device)
        self.Yw = torch.zeros(K, cap, DOUT, device=device)
        self.cap = cap
        self.W = [torch.zeros(DOUT, cap, cap, dtype=self.fd, device=device) for _ in range(K)]
        self.beta = torch.zeros(K, DOUT, cap, dtype=torch.float64, device=device)
        self.n = [0] * K
        self.st = {"ep_pos": 0, "prev": -1, "newest": -1, "added": [0] * K, "adam": [None] * K,
                   "prev_loss": [float("inf")] * K, "fit_rng": None, "counts": {"switches": 0, "created": 0,
                                                                            "refits": 0, "sparsify": 0,
                                                                            "points": 0}, "history": []}
        self.fit_rng = np.random.default_rng([self.seed, 13])

    @torch.no_grad()
    def init_prior(self):
        cfg, model = self.cfg, self.model
        Xt = self.Xall[torch.from_numpy(self.train_eps).to(self.device)].reshape(-1, DIN).double()
        Yt = self.Yall[torch.from_numpy(self.train_eps).to(self.device)].reshape(-1, DOUT).double()
        rng = np.random.default_rng([self.seed, 17])
        sub = torch.from_numpy(rng.choice(Xt.shape[0], size=min(int(cfg["init_subsample"]), Xt.shape[0]),
                                          replace=False)).to(self.device)
        Xs = Xt[sub]
        d = torch.cdist(Xs, Xs)
        iu = torch.triu_indices(Xs.shape[0], Xs.shape[0], 1, device=self.device)
        med = float(d[iu[0], iu[1]].median())
        var = Yt.var(0, unbiased=False)
        model.prior_ls.fill_(med)
        model.prior_os.copy_(var)
        model.prior_noise.copy_(float(cfg["init_noise_frac"]) * var)
        model.prior_const.copy_(Yt.mean(0))
        model.noise_lb.copy_(float(cfg["noise_floor_frac"]) * var)
        model.raw_ls.copy_(inv_softplus(torch.full_like(model.raw_ls, med)))
        model.raw_os.copy_(inv_softplus(var).expand_as(model.raw_os))
        model.raw_noise.copy_(inv_softplus(model.prior_noise - model.noise_lb - model.noise_os_floor * var)
                              .expand_as(model.raw_noise))
        model.const.copy_(model.prior_const.expand_as(model.const))
        return {"median_pairwise_distance": med, "var_delta": var.cpu().tolist(), "mean_delta": Yt.mean(0).cpu().tolist()}

    @torch.no_grad()
    def sync_model(self):
        m = self.model
        for k in range(self.K):
            n = self.n[k]
            m.X[k].zero_()
            m.Y[k].zero_()
            m.X[k, :n] = self.Xw[k, :n]
            m.Y[k, :n] = self.Yw[k, :n]
            m.n_points[k] = n
        m.n_active.fill_(sum(1 for v in self.n if v > 0))

    def refactor_all(self):
        for k in range(self.K):
            if self.n[k] > 0:
                self.refactor(k)

    def free_factors(self):
        self.W = None
        self.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def restore_factors(self):
        cache = getattr(self.model, "_gpmm_cache", None)
        ok = cache is not None and cache.get("fp") == state_fingerprint(self.model) and cache.get("dtype") == str(self.fd)
        self.W = []
        for k in range(self.K):
            buf = torch.zeros(DOUT, self.cap, self.cap, dtype=self.fd, device=self.device)
            n = self.n[k]
            if ok and k < len(cache["fac"]) and n > 0:
                f = cache["fac"][k]
                buf[:, :n, :n] = f["W"]
                self.beta[k].zero_()
                self.beta[k][:, :n] = f["beta"]
                f["W"] = None
            self.W.append(buf)
        self.clear_cache()
        if not ok:
            self.refactor_all()

    def clear_cache(self):
        if hasattr(self.model, "_gpmm_cache"):
            del self.model._gpmm_cache

    def save(self):
        self.st["fit_rng"] = self.fit_rng.bit_generator.state
        self.ctx.save_checkpoint({"model": self.model.state_dict(), "step": int(self.st["ep_pos"]), "st": self.st})

    def try_resume(self):
        ck = self.ctx.load_checkpoint()
        if ck is None:
            self.ctx.reconcile_after_resume(0)
            return False
        self.model.load_state_dict(ck["model"])
        self.st = ck["st"]
        self.fit_rng.bit_generator.state = self.st["fit_rng"]
        self.E.restore_rng(ck["rng"])
        self.ctx.reconcile_after_resume(int(ck["step"]))
        m = self.model
        for k in range(self.K):
            n = int(m.n_points[k])
            self.n[k] = n
            self.Xw[k, :n] = m.X[k, :n]
            self.Yw[k, :n] = m.Y[k, :n]
        self.refactor_all()
        print(json.dumps({"stage": "resumed", "episode": int(ck["step"]), "best_step": self.ctx.best_step,
                          "best_val": self.ctx.best_val}), flush=True)
        return True

    @torch.no_grad()
    def refactor(self, k):
        n = self.n[k]
        ls, os_, noise, c = (t.detach() for t in self.model.hyper(k))
        W, beta = factorize(self.Xw[k, :n], self.Yw[k, :n], ls, os_, noise, c, float(self.cfg["jitter"]), self.fd,
                            int(self.cfg["out_group"]))
        self.W[k].zero_()
        self.beta[k].zero_()
        self.W[k][:, :n, :n] = W
        self.beta[k][:, :n] = beta

    def refit(self, k):
        cfg, model = self.cfg, self.model
        n = self.n[k]
        m = min(n, int(cfg["fit_max_points"]))
        idx = np.sort(self.fit_rng.choice(n, size=m, replace=False))
        it = torch.from_numpy(idx).to(self.device)
        Xs, Ys = self.Xw[k, it].double(), self.Yw[k, it].double()
        raw = [model.raw_ls[k].detach().clone().requires_grad_(True), model.raw_os[k].detach().clone().requires_grad_(True),
               model.raw_noise[k].detach().clone().requires_grad_(True), model.const[k].detach().clone().requires_grad_(True)]
        opt = torch.optim.Adam(raw, lr=float(cfg["gp_lr"]))
        if self.st["adam"][k] is not None:
            opt.load_state_dict(self.st["adam"][k])
        tol = float(cfg["early_stop_tol"])
        steps = 0
        with torch.enable_grad():
            for _ in range(int(cfg["gp_iter"])):
                loss = mll_loss(raw, Xs, Ys, model.noise_lb, float(cfg["jitter"]), model.noise_os_floor)
                lf = float(loss.detach())
                pl = float(self.st["prev_loss"][k])
                if abs(pl - lf) < tol and pl - lf > 0:
                    break
                self.st["prev_loss"][k] = lf
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                steps += 1
        with torch.no_grad():
            model.raw_ls[k].copy_(raw[0])
            model.raw_os[k].copy_(raw[1])
            model.raw_noise[k].copy_(raw[2])
            model.const[k].copy_(raw[3])
        self.st["adam"][k] = opt.state_dict()
        self.st["counts"]["refits"] += 1
        return {"k": k, "n_fit": m, "steps": steps, "loss": self.st["prev_loss"][k]}

    @torch.no_grad()
    def sparsify(self, k):
        cfg = self.cfg
        n = self.n[k]
        keep = int(round(float(cfg["keep_frac"]) * self.M))
        nd = n - keep
        W = self.W[k][:, :n, :n]
        beta = self.beta[k][:, :n]
        best, best_s = None, -float("inf")
        for _ in range(int(cfg["sparsify_candidates"])):
            D = np.sort(self.fit_rng.choice(n, size=nd, replace=False))
            dt = torch.from_numpy(D).to(self.device)
            G = W[:, :, dt]
            Pdd = torch.matmul(G.transpose(1, 2), G).double()
            aD = torch.matmul(G.transpose(1, 2).double(), beta.unsqueeze(-1))[..., 0]
            Lp, info = torch.linalg.cholesky_ex(Pdd)
            if bool((info > 0).any()):
                continue
            sol = torch.cholesky_solve(aD.unsqueeze(-1), Lp)[..., 0]
            score = float((-0.5 * (aD * sol).sum(-1) + torch.log(torch.diagonal(Lp, dim1=-2, dim2=-1)).sum(-1)).sum())
            if score > best_s:
                best_s, best = score, D
        if best is None:
            best = np.sort(self.fit_rng.choice(n, size=nd, replace=False))
        keep_idx = np.setdiff1d(np.arange(n), best)
        kt = torch.from_numpy(keep_idx).to(self.device)
        Xk, Yk = self.Xw[k, kt].clone(), self.Yw[k, kt].clone()
        self.Xw[k].zero_()
        self.Yw[k].zero_()
        self.Xw[k, :keep] = Xk
        self.Yw[k, :keep] = Yk
        self.n[k] = keep
        self.st["counts"]["sparsify"] += 1
        return {"k": k, "dropped": nd, "score": best_s}

    def process_episode(self, e):
        cfg, model = self.cfg, self.model
        K, B0 = self.K, int(cfg["stream_block"])
        f64 = torch.float64
        jit = float(cfg["jitter"])
        refit_every = int(cfg["refit_every"])
        early = set(int(v) for v in cfg.get("refit_early") or [])
        j = 0
        ep_lp, ep_sw, ep_new, ep_guard = [], 0, 0, 0
        events = []
        while j < self.Tn:
            B = min(B0, self.Tn - j)
            Xb = self.Xall[e, j:j + B]
            Yb = self.Yall[e, j:j + B]
            mu = torch.empty(1, K, DOUT, B, dtype=f64, device=self.device)
            C = torch.empty(1, K, DOUT, B, B, dtype=f64, device=self.device)
            noise = torch.empty(1, K, DOUT, dtype=f64, device=self.device)
            Vs = {}
            for k in range(K):
                n = self.n[k]
                if n > 0:
                    ls, os_, nz, c = (t.detach() for t in model.hyper(k))
                    m_, C_, V_ = ctx_stats(self.W[k][:, :n, :n], self.beta[k][:, :n], self.Xw[k, :n], Xb[None], ls,
                                           os_, c, DOUT, self.fd, return_V=True)
                    Vs[k] = V_
                else:
                    ls, os_, nz, c = model.prior_hyper()
                    m_, C_ = ctx_stats(None, None, Xb.new_zeros(0, DIN), Xb[None], ls, os_, c, DOUT, self.fd)
                mu[0, k], C[0, k] = m_[0], C_[0]
                noise[0, k] = eff_noise(nz, os_, jit)
            pls, pos_, pnz, pc = model.prior_hyper()
            yb = Yb.double()[None]
            pvar = pos_ + (eff_noise(pnz, pos_, jit) if cfg["assign_lik"] == "noisy" else 0.0)
            lpn = log_normal_mean(yb, pc, pvar)
            exists = torch.tensor([[v > 0 for v in self.n]], device=self.device)
            npts = torch.tensor([self.n], dtype=torch.long, device=self.device)
            nact = int(exists.sum())
            left = [refit_every - self.st["added"][k] for k in range(K)]
            base_n = list(self.n)

            def stop_fn(i, ks, npts_row, base_n=base_n, left=left):
                nk = int(npts_row[ks])
                return nk > self.M or (nk - base_n[ks]) >= left[ks] or nk in early
            res = assign_sequence(mu, C, yb, noise, lpn, model.R[None, :, :].clone(), exists, npts,
                                  torch.tensor([self.st["newest"]], device=self.device),
                                  torch.tensor([self.st["prev"]], device=self.device),
                                  torch.tensor([nact], device=self.device), cfg,
                                  prior_noise=pnz.view(1, 1, DOUT).expand(1, K, DOUT), stop_fn=stop_fn)
            nd = res["n_done"]
            a = res["assign"][0, :nd].cpu().numpy()
            for k in np.unique(a):
                k = int(k)
                A = np.flatnonzero(a == k)
                At = torch.from_numpy(A).to(self.device)
                na = len(A)
                n = self.n[k]
                L22 = res["L"][0, k][:, At][:, :, At]
                rows, U = append_factor(self.W[k][:, :n, :n] if n > 0 else None,
                                        Vs[k][:, :, At] if n > 0 else None, L22, self.fd)
                if rows is not None:
                    self.W[k][:, n:n + na, :n] = rows
                self.W[k][:, n:n + na, n:n + na] = U
                self.beta[k][:, n:n + na] = res["u"][0, k][:, At]
                self.Xw[k, n:n + na] = Xb[At]
                self.Yw[k, n:n + na] = Yb[At]
                self.n[k] = n + na
                self.st["added"][k] += na
            model.R.copy_(res["R"][0])
            self.st["prev"] = int(res["prev"][0])
            self.st["newest"] = int(res["newest"][0])
            ep_lp.append(res["lp"][0, :nd])
            ep_sw += int((res["assign"][0, 1:nd] != res["assign"][0, :nd - 1]).sum()) if nd > 1 else 0
            ep_new += int(res["created"][0, :nd].sum())
            ep_guard += int(res["n_guard"].sum())
            last = int(a[-1])
            if self.n[last] > self.M:
                ev = self.sparsify(last)
                ev.update(self.refit(last))
                self.refactor(last)
                self.st["added"][last] = 0
                events.append({"kind": "sparsify+refit", **ev})
            elif self.st["added"][last] >= refit_every or self.n[last] in early:
                ev = self.refit(last)
                self.refactor(last)
                self.st["added"][last] = 0
                events.append({"kind": "refit", **ev})
            j += nd
        self.st["counts"]["switches"] += ep_sw
        self.st["counts"]["created"] += ep_new
        self.st["counts"]["points"] += self.Tn
        lp = torch.cat(ep_lp)
        return {"mean_assign_loglik": float(lp.mean()), "switches": ep_sw, "created": ep_new, "events": len(events),
                "guard_hits": ep_guard}

    def validate(self, step):
        self.sync_model()
        self.free_factors()
        try:
            self.ctx.validate(step, self.module, self.model, self.cfg, component="stream")
        finally:
            self.restore_factors()

    def run(self):
        cfg = self.cfg
        if self.st["ep_pos"] == 0 and not self.st["history"]:
            info = self.init_prior()
            self.st["history"].append({"kind": "prior", **info})
            print(json.dumps({"stage": "gpmm_prior", "median_pairwise_distance": info["median_pairwise_distance"]}),
                  flush=True)
        while self.st["ep_pos"] < self.n_stream:
            t0 = time.time()
            e = int(self.order[self.st["ep_pos"]])
            info = self.process_episode(e)
            self.st["ep_pos"] += 1
            s = int(self.st["ep_pos"])
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t0
            metrics = {"n_active": sum(1 for v in self.n if v > 0), **{f"n{k}": self.n[k] for k in range(self.K)},
                       "switches": info["switches"], "created": info["created"], "events": info["events"],
                       "guard_hits": info["guard_hits"],
                       "refits_total": self.st["counts"]["refits"], "sparsify_total": self.st["counts"]["sparsify"]}
            self.ctx.log_train(s, -info["mean_assign_loglik"], metrics, dt, component="stream")
            self.ctx.progress(s, phase="stream", component="stream", steps=self.n_stream)
            is_val = (s % self.val_every == 0) or s == self.n_stream
            if is_val:
                self.validate(s)
            if is_val or s % self.ckpt_every == 0 or self.TR._Stop.flag:
                if not is_val:
                    self.refactor_all()
                    self.sync_model()
                self.save()
            if self.TR._Stop.flag and s < self.n_stream:
                print(json.dumps({"stage": "terminated", "signal": self.TR._Stop.signum, "checkpoint_step": s}),
                      flush=True)
                raise SystemExit(self.TR.EXIT_TERMINATED)
        return self.final_checks()

    @torch.no_grad()
    def final_checks(self):
        from pipeline import scorer as S
        from pipeline.data import context_keys, windows
        model, cfg = self.model, self.cfg
        self.sync_model()
        out = {"n_active": int(model.n_active), "n_points": model.n_points.cpu().tolist(),
               "R": model.R.cpu().tolist(), "counts": self.st["counts"]}
        hyp = []
        for k in range(int(model.n_active)):
            ls, os_, nz, c = model.hyper(k)
            hyp.append({"median_lengthscale": float(ls.median()), "min_lengthscale": float(ls.min()),
                        "median_noise_over_os": float((nz / os_).median()), "median_os": float(os_.median())})
        out["hyper"] = hyp
        Xh = self.Xall[torch.from_numpy(self.hold).to(self.device)].reshape(-1, DIN)
        sub = torch.from_numpy(np.random.default_rng([self.seed, 19]).choice(Xh.shape[0], size=min(2000, Xh.shape[0]),
                                                                             replace=False)).to(self.device)
        Xs = Xh[sub]
        best = torch.zeros(DOUT, Xs.shape[0], device=self.device)
        for k in range(int(model.n_active)):
            n = int(model.n_points[k])
            ls, os_, _, _ = model.hyper(k)
            for g0 in range(0, DOUT, 4):
                sl = slice(g0, min(DOUT, g0 + 4))
                kv = kern(model.X[k, :n], Xs, ls[sl].float(), torch.ones_like(os_[sl]).float())
                best[sl] = torch.maximum(best[sl], kv.max(1).values)
        ks = float(best.median())
        out["kernel_sanity"] = {"median_max_k_over_os": ks, "per_output_median": best.median(1).values.cpu().tolist(),
                                "passed": ks >= 0.1}
        e = np.repeat(self.hold, len(P.EVAL_STARTS))
        t = np.tile(np.asarray(P.EVAL_STARTS), len(self.hold))
        w = windows(self.obs_np, self.act_np, e, t, P.MAX_H)
        hq = {"ctx_obs": np.ascontiguousarray(w["ctx_obs"]), "ctx_act": np.ascontiguousarray(w["ctx_act"]),
              "fut_act": np.ascontiguousarray(w["fut_act"]), "keys": context_keys(w["ctx_obs"], w["ctx_act"])}
        own, common, raw = S.frame_scales(self.stats)
        self.free_factors()
        try:
            fm = forecaster(model, cfg)
            res = self.E.forecast_arrays(fm, hq, self.device, 720)
            del fm
        finally:
            self.clear_cache()
        cm, lw = self.E.components(res)
        g_val, g_ph, _ = S.selection_metric(cm, lw, w["fut_obs"], own, common, raw)
        zeros = np.zeros_like(w["fut_obs"])[:, :, None]
        m_val, m_ph, _ = S.selection_metric(zeros, np.zeros((len(e), 1)), w["fut_obs"], own, common, raw)
        pers = np.repeat(w["ctx_obs"][:, -1:, :], P.MAX_H, 1)[:, :, None]
        p_val, p_ph, _ = S.selection_metric(pers, np.zeros((len(e), 1)), w["fut_obs"], own, common, raw)
        h1 = "nMSE_train__prefix_H1"
        out["holdout"] = {"queries": int(len(e)), "gpmm": g_val, "mean_predictor": m_val, "persistence": p_val,
                          "gpmm_H1": g_ph[h1], "mean_H1": m_ph[h1], "persistence_H1": p_ph[h1],
                          "gpmm_per_h": {k: v for k, v in g_ph.items() if k.startswith("nMSE")},
                          "beats_mean_and_persistence": bool(g_val < m_val and g_val < p_val),
                          "beats_mean_and_persistence_H1": bool(g_ph[h1] < m_ph[h1] and g_ph[h1] < p_ph[h1])}
        print(json.dumps({"stage": "gpmm_checks", "kernel_sanity": ks, **{k: v for k, v in out["holdout"].items()
                                                                          if not isinstance(v, dict)}}), flush=True)
        return out


def custom_train(cfg, seed, out_dir, device, stats, schedule, data):
    s = _Stream(cfg, seed, out_dir, device, stats, data)
    s.try_resume()
    t0 = time.time()
    checks = s.run()
    lib = {"schema": "gpmm_library_v1", "episodes_streamed": int(s.st["ep_pos"]), "stream_order_seed": [int(seed), 11],
           "holdout_episodes": s.hold.tolist(), "checks": checks, "history": s.st["history"],
           "disclosures": DISCLOSURES, "seconds_this_attempt": time.time() - t0}
    R.atomic_json(s.ctx.p("LIBRARY.json"), lib)
    s.ctx.complete({"custom_train": True, "n_active": checks["n_active"], "n_points": checks["n_points"],
                    "kernel_sanity": checks["kernel_sanity"]["median_max_k_over_os"],
                    "holdout": {k: v for k, v in checks["holdout"].items() if not isinstance(v, dict)}})
    print(json.dumps({"stage": "complete", "best_step": s.ctx.best_step, "best_val": s.ctx.best_val,
                      "wall_s": round(s.ctx.wall(), 1), "n_active": checks["n_active"]}), flush=True)
