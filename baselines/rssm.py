from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline import registry as P
from pipeline.eval_api import Forecast, QueryBatch

from . import keyed_sampling as KS

NAME = "rssm"
FAMILY = "baseline"

_O = P.OBS_DIM
_A = P.ACT_DIM


def _lin(i: int, o: int) -> int:
    return i * o + o


def param_count(cfg: dict) -> int:
    D, Hh, U = cfg["deter"], cfg["hidden"], cfg["units"]
    SK = cfg["stoch"] * cfg["classes"]
    Hc = cfg["core_hidden"]
    rms = lambda d: d
    enc = _lin(_O, U) + rms(U) + (cfg["enc_layers"] - 1) * (_lin(U, U) + rms(U))
    dynin = (_lin(D, Hh) + rms(Hh)) + (_lin(SK, Hh) + rms(Hh)) + (_lin(_A, Hh) + rms(Hh))
    core = _lin(D + 3 * Hh, Hc) + rms(Hc) + _lin(Hc, D)
    post = sum(_lin(D + U if i == 0 else Hh, Hh) + rms(Hh) for i in range(cfg["obs_layers"])) + _lin(Hh, SK)
    prior = sum(_lin(D if i == 0 else Hh, Hh) + rms(Hh) for i in range(cfg["img_layers"])) + _lin(Hh, SK)
    dec = _lin(SK + D, U) + rms(U) + (cfg["dec_layers"] - 1) * (_lin(U, U) + rms(U)) + _lin(U, _O)
    return enc + dynin + core + post + prior + dec


def default_config(budget_class: str = "baseline", **variant) -> dict:
    if budget_class != "baseline":
        raise ValueError(f"{NAME}: only budget_class 'baseline' is defined, got {budget_class!r}")
    cfg = dict(
        name=NAME, family=FAMILY, budget_class=budget_class, budget_target=P.BUDGET_BASELINE,
        deter=512, hidden=64, units=64, stoch=32, classes=4,
        enc_layers=3, dec_layers=3, img_layers=2, obs_layers=1,
        core_hidden=None,
        core_out_scale=0.1,
        unimix=0.01, norm_eps=1e-4, free_nats=1.0,
        loss_scales={"rec": 1.0, "dyn": 1.0, "rep": 0.1},
        optimizer="LaPropAGC", lr=4e-5, agc=0.3, agc_pmin=1e-3, beta1=0.9, beta2=0.999, opt_eps=1e-20,
        warmup=1000, weight_decay=0.0,
        grad_clip=None,
        eval_latent="mode",
        eval_chunk=240,
    )
    cfg.update(variant)
    if cfg["core_hidden"] is None:
        target = cfg["budget_target"]
        best = None
        for hc in range(8, 8192):
            n = param_count(dict(cfg, core_hidden=hc))
            if best is None or abs(n - target) < abs(best[1] - target):
                best = (hc, n)
        cfg["core_hidden"] = best[0]
    cfg["param_count_formula"] = param_count(cfg)
    return cfg


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class RMSNorm(nn.Module):


    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))
        self._pad = (-int(d)) % 4
        self._d = int(d)

    def forward(self, x):
        if self._pad:
            ms = F.pad(x.square(), (0, self._pad)).sum(-1, keepdim=True) / self._d
            return x * torch.rsqrt(ms + self.eps) * self.scale
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.scale


def dlinear(i: int, o: int, outscale: float = 1.0) -> nn.Linear:
    lin = nn.Linear(i, o)
    with torch.no_grad():
        nn.init.trunc_normal_(lin.weight, mean=0.0, std=1.0, a=-2.0, b=2.0)
        lin.weight.mul_(1.1368 * math.sqrt(1.0 / i) * outscale)
        lin.bias.zero_()
    return lin


class LNA(nn.Module):

    def __init__(self, i: int, o: int, eps: float):
        super().__init__()
        self.lin = dlinear(i, o)
        self.norm = RMSNorm(o, eps)

    def forward(self, x):
        return F.silu(self.norm(self.lin(x)))


class DreamerMLP(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        D, Hh, U, S, K, Hc = (cfg["deter"], cfg["hidden"], cfg["units"], cfg["stoch"], cfg["classes"],
                              cfg["core_hidden"])
        eps = cfg["norm_eps"]
        self.D, self.S, self.K = D, S, K
        self.unimix = float(cfg["unimix"])
        self.enc = nn.Sequential(*[LNA(_O if i == 0 else U, U, eps) for i in range(cfg["enc_layers"])])
        self.dynin0 = LNA(D, Hh, eps)
        self.dynin1 = LNA(S * K, Hh, eps)
        self.dynin2 = LNA(_A, Hh, eps)
        self.core_hid = LNA(D + 3 * Hh, Hc, eps)
        self.core_out = dlinear(Hc, D, outscale=float(cfg.get("core_out_scale", 1.0)))
        self.obs_net = nn.Sequential(*[LNA(D + U if i == 0 else Hh, Hh, eps) for i in range(cfg["obs_layers"])])
        self.obs_logit = dlinear(Hh, S * K)
        self.img_net = nn.Sequential(*[LNA(D if i == 0 else Hh, Hh, eps) for i in range(cfg["img_layers"])])
        self.img_logit = dlinear(Hh, S * K)
        self.dec_net = nn.Sequential(*[LNA(S * K + D if i == 0 else U, U, eps) for i in range(cfg["dec_layers"])])
        self.dec_out = dlinear(U, _O)

    GROUPS = {
        "encoder": ("enc",),
        "rssm_core": ("dynin0", "dynin1", "dynin2", "core_hid", "core_out"),
        "posterior": ("obs_net", "obs_logit"),
        "prior": ("img_net", "img_logit"),
        "decoder": ("dec_net", "dec_out"),
    }

    def tokens(self, z):
        return self.enc(symlog(z))

    def act_branch(self, a):
        return self.dynin2(a / torch.clamp(torch.abs(a), min=1.0))

    def core(self, deter, stoch, x2):
        x0 = self.dynin0(deter)
        x1 = self.dynin1(stoch)
        h = self.core_hid(torch.cat([deter, x0, x1, x2], -1))
        return torch.tanh(deter + self.core_out(h))

    def logp(self, logits):
        lg = logits.view(*logits.shape[:-1], self.S, self.K)
        probs = torch.softmax(lg, -1) * (1.0 - self.unimix) + self.unimix / self.K
        return torch.log(probs)

    def post_logp(self, deter, tok):
        return self.logp(self.obs_logit(self.obs_net(torch.cat([deter, tok], -1))))

    def prior_logp(self, deter):
        return self.logp(self.img_logit(self.img_net(deter)))

    def sample_st(self, logp, gumbel):
        idx = torch.argmax(logp + gumbel, -1)
        onehot = F.one_hot(idx, self.K).to(logp.dtype)
        probs = torch.exp(logp)
        return (onehot + probs - probs.detach()).flatten(-2)

    def mode(self, logp):
        return F.one_hot(torch.argmax(logp, -1), self.K).to(logp.dtype).flatten(-2)

    def decode(self, stoch, deter):
        return self.dec_out(self.dec_net(torch.cat([stoch, deter], -1)))

    def observe(self, z, prevact, sample: bool, keep: bool, gumbel_fn=None):
        B, T, _ = z.shape
        tok = self.tokens(z)
        x2 = self.act_branch(prevact)
        if sample:
            u = torch.rand(B, T, self.S, self.K, device=z.device, dtype=z.dtype).clamp_(min=1e-20)
            gumbel = -torch.log(-torch.log(u))
        deter = z.new_zeros(B, self.D)
        stoch = z.new_zeros(B, self.S * self.K)
        deters, stochs, logps = [], [], []
        for j in range(T):
            deter = self.core(deter, stoch, x2[:, j])
            lp = self.post_logp(deter, tok[:, j])
            if gumbel_fn is not None:
                stoch = self.sample_st(lp, gumbel_fn(j))
            else:
                stoch = self.sample_st(lp, gumbel[:, j]) if sample else self.mode(lp)
            if keep:
                deters.append(deter)
                stochs.append(stoch)
                logps.append(lp)
        if keep:
            return deter, stoch, torch.stack(deters, 1), torch.stack(stochs, 1), torch.stack(logps, 1)
        return deter, stoch

    def imagine_sample(self, deter, stoch, fut_act, gumbel_fn):
        x2 = self.act_branch(fut_act)
        deters, stochs = [], []
        for k in range(fut_act.shape[1]):
            deter = self.core(deter, stoch, x2[:, k])
            stoch = self.sample_st(self.prior_logp(deter), gumbel_fn(k))
            deters.append(deter)
            stochs.append(stoch)
        return torch.stack(deters, 1), torch.stack(stochs, 1)

    def imagine_mode(self, deter, stoch, fut_act):
        x2 = self.act_branch(fut_act)
        deters, stochs = [], []
        for k in range(fut_act.shape[1]):
            deter = self.core(deter, stoch, x2[:, k])
            stoch = self.mode(self.prior_logp(deter))
            deters.append(deter)
            stochs.append(stoch)
        return torch.stack(deters, 1), torch.stack(stochs, 1)


class LaPropAGC(torch.optim.Optimizer):


    def __init__(self, params, lr=4e-5, agc=0.3, pmin=1e-3, beta1=0.9, beta2=0.999, eps=1e-20, warmup=1000):
        defaults = dict(lr=0.0 if warmup else lr, base_lr=lr, agc=agc, pmin=pmin, beta1=beta1, beta2=beta2,
                        eps=eps, warmup=warmup, count=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            agc, pmin = group["agc"], group["pmin"]
            b1, b2, eps, warm = group["beta1"], group["beta2"], group["eps"], group["warmup"]
            count = int(group["count"])
            t = count + 1
            lr_t = group["base_lr"] * (min(count / warm, 1.0) if warm else 1.0)
            group["lr"] = lr_t
            group["count"] = t
            c2 = 1.0 - b2 ** t
            c1 = 1.0 - b1 ** t
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if len(st) == 0:
                    st["nu"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["mu"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                g = p.grad
                if agc:
                    upper = agc * torch.clamp(torch.linalg.vector_norm(p), min=pmin)
                    g = g / torch.clamp(torch.linalg.vector_norm(g) / upper, min=1.0)
                nu, mu = st["nu"], st["mu"]
                nu.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                u = g / (torch.sqrt(nu / c2) + eps)
                mu.mul_(b1).add_(u, alpha=1.0 - b1)
                if lr_t != 0.0:
                    p.add_(mu, alpha=-lr_t / c1)
        return loss


def make_optimizer(model, cfg):
    if cfg.get("weight_decay", 0.0):
        raise ValueError("weight decay is not part of the native DreamerV3 default optimizer")
    return LaPropAGC(model.parameters(), lr=cfg["lr"], agc=cfg["agc"], pmin=cfg["agc_pmin"],
                     beta1=cfg["beta1"], beta2=cfg["beta2"], eps=cfg["opt_eps"], warmup=cfg["warmup"])


def build(cfg: dict, seed: int, device: str, stats=None) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = DreamerMLP(cfg)
    n = sum(p.numel() for p in model.parameters())
    assert n == param_count(cfg), (n, param_count(cfg))
    return model.to(device)


def _prevact(first_like: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros_like(first_like), acts], 1)


def _kl(lp_p, lp_q):
    return (torch.exp(lp_p) * (lp_p - lp_q)).sum((-2, -1))


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
    z = batch["win_obs"]
    a = batch["win_act"]
    T = z.shape[1]
    prev = _prevact(a[:, :1], a[:, : T - 1])
    _, _, deters, stochs, post_lp = model.observe(z, prev, sample=True, keep=True)
    prior_lp = model.prior_logp(deters)
    free = cfg["free_nats"]
    dyn_raw = _kl(post_lp.detach(), prior_lp)
    rep_raw = _kl(post_lp, prior_lp.detach())
    dyn = torch.clamp(dyn_raw, min=free).mean()
    rep = torch.clamp(rep_raw, min=free).mean()
    pred = model.decode(stochs, deters)
    rec = (pred - symlog(z)).square().sum(-1).mean()
    sc = cfg["loss_scales"]
    loss = sc["rec"] * rec + sc["dyn"] * dyn + sc["rep"] * rep
    with torch.no_grad():
        post_ent = -(torch.exp(post_lp) * post_lp).sum((-2, -1)).mean()
        prior_ent = -(torch.exp(prior_lp) * prior_lp).sum((-2, -1)).mean()
    return loss, _floats(loss=loss, rec=rec, dyn=dyn, rep=rep, kl_raw=dyn_raw.mean(),
                         post_ent=post_ent, prior_ent=prior_ent)


class _Forecaster:
    family = FAMILY
    budget_class = "baseline"
    needs_grad_at_eval = False
    stochastic = False

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.eval_mode = KS.eval_mode()
        self.name = NAME if self.eval_mode == "mode" else f"{NAME}[{self.eval_mode}]"
        self.stochastic = self.eval_mode != "mode"

    def _mean_sampled(self, ctx_obs, ctx_act, fut_act, keys):
        m, n = self.model, KS.N_SAMPLES
        Q = ctx_obs.shape[0]
        rep = lambda x: x.repeat_interleave(n, 0)
        prev = _prevact(ctx_act[:, :1], ctx_act)
        so = KS.query_sample_seeds(keys, n, KS.PHASE_OBSERVE)
        si = KS.query_sample_seeds(keys, n, KS.PHASE_IMAGINE)
        deter, stoch = m.observe(rep(ctx_obs), rep(prev), sample=False, keep=False,
                                 gumbel_fn=lambda t: KS.gumbel_step(so, t, m.S, m.K, ctx_obs.dtype))
        deters, stochs = m.imagine_sample(deter, stoch, rep(fut_act),
                                          lambda k: KS.gumbel_step(si, k, m.S, m.K, ctx_obs.dtype))
        pred = symexp(m.decode(stochs, deters))
        return pred.view(Q, n, *pred.shape[1:]).mean(1)

    def _mean(self, ctx_obs, ctx_act, fut_act):
        m = self.model
        prev = _prevact(ctx_act[:, :1], ctx_act)
        deter, stoch = m.observe(ctx_obs, prev, sample=False, keep=False)
        deters, stochs = m.imagine_mode(deter, stoch, fut_act)
        return symexp(m.decode(stochs, deters))

    @torch.no_grad()
    def forecast(self, q: QueryBatch) -> Forecast:
        dev = next(self.model.parameters()).device
        if self.eval_mode == "mode":
            mean = fixed_chunk_forecast(self._mean, q, int(self.cfg.get("eval_chunk", 240)), dev)
        else:
            mean = KS.chunked_keyed(self._mean_sampled, q, KS.KEYED_CHUNK, dev)
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
    groups = {}
    for g, names in DreamerMLP.GROUPS.items():
        groups[g] = sum(p.numel() for n in names for p in getattr(model, n).parameters())
    total = trainable + frozen
    assert sum(groups.values()) == total, (groups, total)
    target = cfg["budget_target"]
    return {"total": total, "trainable": trainable, "frozen": frozen, "stored_nongrad": 0,
            "buffers": buffers, "groups": groups, "budget_class": cfg["budget_class"], "budget_target": target,
            "within_budget": abs(total - target) <= P.BUDGET_TOL * target,
            "state": {"deter": cfg["deter"], "stoch": f"{cfg['stoch']}x{cfg['classes']}"}}
