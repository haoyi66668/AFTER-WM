from __future__ import annotations

import contextlib

import numpy as np
import torch
from torch.utils import _pytree as PYT


def make_ctx_tensors(ctx_obs_n, ctx_act_n, device):
    co = np.ascontiguousarray(ctx_obs_n, np.float32)[None]
    ca = np.ascontiguousarray(ctx_act_n, np.float32)[None]
    return co, ca, torch.from_numpy(co).to(device), torch.from_numpy(ca).to(device)


class PlanningModel:
    split = False
    name = "?"


class ForecasterPlan(PlanningModel):
    split = False

    def __init__(self, module, model, cfg, device):
        self.fm = module.forecaster(model, cfg)
        self.device = device
        self.name = getattr(self.fm, "name", "?")
        self.needs_grad = bool(getattr(self.fm, "needs_grad_at_eval", False))
        self.stochastic = bool(getattr(self.fm, "stochastic", False))

    def encode(self, ctx_obs_n, ctx_act_n):
        from pipeline import data as D
        co, ca, cot, cat = make_ctx_tensors(ctx_obs_n, ctx_act_n, self.device)
        key = int(D.context_keys(co, ca)[0])
        return (cot, cat, key)

    def predict(self, enc, fut_act):
        from pipeline import evaluate as E
        from pipeline.eval_api import QueryBatch
        co, ca, key = enc
        n = int(fut_act.shape[0])
        qb = QueryBatch(ctx_obs=co.expand(n, -1, -1).contiguous(), ctx_act=ca.expand(n, -1, -1).contiguous(),
                        fut_act=fut_act.detach().contiguous(),
                        key=torch.full((n,), key, dtype=torch.int64, device=co.device))
        grad_ctx = contextlib.nullcontext() if self.needs_grad else torch.no_grad()
        with grad_ctx, E.label_guard():
            f = self.fm.forecast(qb)
        m = f.mean.detach()
        if tuple(m.shape) != (n, fut_act.shape[1], 17):
            raise ValueError(f"forecast mean shape {tuple(m.shape)}")
        return m.float()


class _SplitBase(PlanningModel):
    split = True
    stochastic = False
    needs_grad = False
    graph_encode = True
    graph_plan = True

    def __init__(self, model, name, device):
        self.m, self.name, self.device = model, name, device

    def encode(self, ctx_obs_n, ctx_act_n):
        _, _, co, ca = make_ctx_tensors(ctx_obs_n, ctx_act_n, self.device)
        return self.encode_t(co, ca)


def _expand(x, n):
    if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.shape[0] == 1:
        return x.expand(n, *x.shape[1:])
    return x


class AfterWMPlan(_SplitBase):

    @torch.no_grad()
    def encode_t(self, co, ca):
        m = self.m
        zc, ac, _, _ = m.native_inputs(co, ca, ca[:, :1])
        state = m.init_state(zc.shape[0], zc)
        for j in range(ac.shape[1]):
            state = m.observe_fast(state, zc[:, j], ac[:, j], zc[:, j + 1])
        leaves, self.spec = PYT.tree_flatten(state)
        return tuple(leaves) + (zc[:, -1],)

    @torch.no_grad()
    def predict(self, enc, fut_act):
        m = self.m
        n = fut_act.shape[0]
        leaves, z = enc[:-1], enc[-1]
        state = PYT.tree_unflatten([_expand(x, n) for x in leaves], self.spec)
        af = m.fmap.act_to_native(fut_act)
        pred, logp = m.predict_fast(state, z.expand(n, *z.shape[1:]), af)
        logw = torch.log_softmax(logp.double(), -1)
        mean_native = (logw.exp()[:, None, :, None] * pred.double()).sum(2)
        return m.fmap.obs_to_train(mean_native).float()


class ReBALPlan(_SplitBase):
    @torch.no_grad()
    def encode_t(self, co, ca):
        _, h = self.m.teacher_forced(co[:, :-1], ca)
        return (h, co[:, -1])

    @torch.no_grad()
    def predict(self, enc, fut_act):
        m = self.m
        n = fut_act.shape[0]
        h, z = (x.expand(n, *x.shape[1:]) for x in enc)
        wt = m.w_h_t()
        outs = []
        for k in range(fut_act.shape[1]):
            h = m.cell(h, m.proj_za(z, fut_act[:, k]), wt)
            z = z + m.denorm_delta(m.head(h))
            outs.append(z)
        return torch.stack(outs, 1)


class TDMPC2Plan(_SplitBase):
    @torch.no_grad()
    def encode_t(self, co, ca):
        return (self.m.encode(co[:, -1]),)

    @torch.no_grad()
    def predict(self, enc, fut_act):
        m = self.m
        n = fut_act.shape[0]
        z = enc[0].expand(n, *enc[0].shape[1:])
        zs = []
        for k in range(fut_act.shape[1]):
            z = m.next(z, fut_act[:, k])
            zs.append(z)
        return m.decode(torch.stack(zs, 1))


class AdaptOncePlan(_SplitBase):
    graph_encode = False

    def __init__(self, fm, row, device):
        self.fm, self.row, self.device = fm, row, device
        self.m = fm.model
        self.name = getattr(fm, "name", row)

    def encode_t(self, co, ca):
        from baselines import grbal as G
        model, cfg = self.m, self.fm.cfg
        net, dm, ds = model.net, model.delta_mean, model.delta_std
        with torch.inference_mode(False):
            co, ca = co.detach().float().clone(), ca.detach().float().clone()
            p0 = model.theta(detach=True)
            x, y = G.transitions(co, ca, dm, ds)
            with torch.enable_grad():
                if self.row == "grbal":
                    M = int(cfg["adapt_m"])
                    pa = G.adapt_k_shot(net, p0, x[:, -M:], y[:, -M:], float(cfg["inner_lr"]))
                else:
                    from baselines import mole as MO
                    if not bool((model.sigma2 > 0).all()):
                        raise RuntimeError("MOLe sigma^2 is not calibrated")
                    C = max(1, int(cfg.get("eval_chunk", 48)))
                    xc, yc = x.expand(C, *x.shape[1:]).contiguous(), y.expand(C, *y.shape[1:]).contiguous()
                    pa, _ = MO.online_em(net, p0, xc, yc, model.sigma2, cfg)
                    pa = {k: v[:1] for k, v in pa.items()}
        self.keys = sorted(pa)
        return tuple(pa[k].detach() for k in self.keys) + (co[:, -1].detach(),)

    @torch.no_grad()
    def predict(self, enc, fut_act):
        from torch.func import functional_call
        net, dm, ds = self.m.net, self.m.delta_mean, self.m.delta_std
        params = {k: v[0] for k, v in zip(self.keys, enc[:-1])}
        n = fut_act.shape[0]
        z = enc[-1].expand(n, *enc[-1].shape[1:])
        out = []
        for k in range(fut_act.shape[1]):
            z = z + dm + ds * functional_call(net, params, (torch.cat([z, fut_act[:, k]], dim=-1),))
            out.append(z)
        return torch.stack(out, 1)


class RSSMPlan(_SplitBase):
    graph_encode = False

    def __init__(self, fm, device):
        if getattr(fm, "eval_mode", None) != "sample16":
            raise RuntimeError("the RSSM must run in its JSHC evaluation mode sample16")
        self.fm, self.m, self.device = fm, fm.model, device
        self.name = getattr(fm, "name", "rssm")

    @torch.no_grad()
    def encode(self, ctx_obs_n, ctx_act_n):
        from baselines import rssm as DR
        from baselines import keyed_sampling as KS
        from pipeline import data as D
        co_np, ca_np, co, ca = make_ctx_tensors(ctx_obs_n, ctx_act_n, self.device)
        keys = torch.as_tensor(D.context_keys(co_np, ca_np), device=self.device, dtype=torch.long)
        m, n = self.m, KS.N_SAMPLES
        rep = lambda x: x.repeat_interleave(n, 0)
        prev = DR._prevact(ca[:, :1], ca)
        so = KS.query_sample_seeds(keys, n, KS.PHASE_OBSERVE)
        si = KS.query_sample_seeds(keys, n, KS.PHASE_IMAGINE)
        deter, stoch = m.observe(rep(co), rep(prev), sample=False, keep=False,
                                 gumbel_fn=lambda t: KS.gumbel_step(so, t, m.S, m.K, co.dtype))
        return (deter, stoch, si)

    @torch.no_grad()
    def predict(self, enc, fut_act):
        from baselines import rssm as DR
        from baselines import keyed_sampling as KS
        m, n = self.m, KS.N_SAMPLES
        deter1, stoch1, si1 = enc
        N = fut_act.shape[0]
        deter = deter1.repeat(N, *([1] * (deter1.dim() - 1)))
        stoch = stoch1.repeat(N, *([1] * (stoch1.dim() - 1)))
        si = si1.repeat(N)
        deters, stochs = m.imagine_sample(deter, stoch, fut_act.repeat_interleave(n, 0),
                                          lambda k: KS.gumbel_step(si, k, m.S, m.K, fut_act.dtype))
        pred = DR.symexp(m.decode(stochs, deters))
        return pred.view(N, n, *pred.shape[1:]).mean(1)


class GPMMPlan(_SplitBase):
    graph_encode = False
    graph_plan = False

    def __init__(self, fm, device):
        self.fm, self.m, self.device = fm, fm.model, device
        self.name = getattr(fm, "name", "gpmm")

    @torch.no_grad()
    def encode_t(self, co, ca):
        from baselines import gpmm as GP
        fmo, model, cfg, fd = self.fm, self.m, self.fm.cfg, self.fm.fd
        with torch.inference_mode(False):
            co, ca = co.detach().float().clone(), ca.detach().float().clone()
            dev = co.device
            Q, Lc = co.shape[0], co.shape[1] - 1
            Kl = fmo.Kl
            nloc = min(int(cfg["test_local_new"]), int(cfg["K_max"]) - Kl) if Kl > 0 else 1
            nloc = max(0, nloc)
            Kc = Kl + nloc
            grp = int(cfg["out_group"])
            jit = float(cfg["jitter"])
            DIN, DOUT = GP.DIN, GP.DOUT
            x = torch.cat([co[:, :-1], ca], -1)
            y = (co[:, 1:] - co[:, :-1]).double()
            mu = torch.empty(Q, Kc, DOUT, Lc, dtype=torch.float64, device=dev)
            C = torch.empty(Q, Kc, DOUT, Lc, Lc, dtype=torch.float64, device=dev)
            noise = torch.empty(1, Kc, DOUT, dtype=torch.float64, device=dev)
            hyp = []
            for k in range(Kl):
                ls, os_, nz, c = (t.detach() for t in model.hyper(k))
                hyp.append((ls, os_, c))
                f = fmo.fac[k]
                mu[:, k], C[:, k] = GP.ctx_stats(f["W"], f["beta"], f["X"], x, ls, os_, c, grp, fd)
                noise[0, k] = GP.eff_noise(nz, os_, jit)
            pls, pos_, pnz, pc = model.prior_hyper()
            for k in range(Kl, Kc):
                hyp.append((pls, pos_, pc))
                mu[:, k], C[:, k] = GP.ctx_stats(None, None, x.new_zeros(0, DIN), x, pls, pos_, pc, grp, fd)
                noise[0, k] = GP.eff_noise(pnz, pos_, jit)
            pvar = pos_ + (GP.eff_noise(pnz, pos_, jit) if cfg["assign_lik"] == "noisy" else 0.0)
            lpn = GP.log_normal_mean(y, pc, pvar)
            Rm = torch.zeros(Q, Kc, Kc, dtype=torch.float64, device=dev)
            Rm[:, :Kl, :Kl] = model.R[:Kl, :Kl]
            exists = torch.zeros(Q, Kc, dtype=torch.bool, device=dev)
            exists[:, :Kl] = True
            npts = torch.zeros(Q, Kc, dtype=torch.long, device=dev)
            npts[:, :Kl] = model.n_points[:Kl]
            newest = torch.full((Q,), Kl - 1, dtype=torch.long, device=dev)
            prev = torch.full((Q,), -1, dtype=torch.long, device=dev)
            next_slot = torch.full((Q,), Kl if nloc > 0 else Kc, dtype=torch.long, device=dev)
            res = GP.assign_sequence(mu, C, y, noise, lpn, Rm, exists, npts, newest, prev, next_slot, cfg,
                                     prior_noise=pnz.view(1, 1, DOUT).expand(1, Kc, DOUT))
            kstar = int(res["assign"][0, -1])
            gamma = torch.linalg.solve_triangular(res["L"].transpose(-1, -2), res["u"].unsqueeze(-1), upper=True)[..., 0]
            g_k = gamma[0, kstar]
            ls, os_, c = hyp[kstar]
            if kstar < Kl:
                f = fmo.fac[kstar]
                n = f["n"]
                flat = x.reshape(Q * Lc, DIN).to(fd)
                w2 = []
                for g0 in range(0, DOUT, grp):
                    sl = slice(g0, min(DOUT, g0 + grp))
                    gg = sl.stop - sl.start
                    V = torch.matmul(f["W"][sl], GP.kern(f["X"].to(fd), flat, ls[sl].to(fd), os_[sl].to(fd)))
                    psi = torch.einsum("gnqi,qgi->qgn", V.reshape(gg, n, Q, Lc), gamma[:, kstar, sl].to(fd))
                    w2.append(f["beta"][sl].to(fd).unsqueeze(0) - psi)
                w2 = torch.cat(w2, 1)[0]
                X = (f["X"], f["W"])
            else:
                w2, X = None, None
        self.k_info = (kstar, Kl, grp, fd, ls, os_, c, X, w2)
        return (x.to(fd), g_k, co[:, -1])

    @torch.no_grad()
    def predict(self, enc, fut_act):
        from baselines import gpmm as GP
        kstar, Kl, grp, fd, ls, os_, c, X, w2 = self.k_info
        xq, g_k, z1 = enc
        n = fut_act.shape[0]
        z = z1.expand(n, *z1.shape[1:])
        outs = []
        for step in range(fut_act.shape[1]):
            xin = torch.cat([z, fut_act[:, step]], -1)
            kA = GP.kern(xq.unsqueeze(1), xin.to(fd).view(n, 1, 1, GP.DIN), ls.to(fd).unsqueeze(0),
                         os_.to(fd).unsqueeze(0))[..., 0]
            corr = (kA.double() * g_k).sum(-1)
            if kstar < Kl:
                main = []
                Xl, W = X
                for g0 in range(0, GP.DOUT, grp):
                    sl = slice(g0, min(GP.DOUT, g0 + grp))
                    v = torch.matmul(W[sl], GP.kern(Xl.to(fd), xin.to(fd), ls[sl].to(fd), os_[sl].to(fd)))
                    main.append(torch.einsum("gnq,gn->qg", v, w2[sl]))
                dz = c.double() + torch.cat(main, 1).double() + corr
            else:
                dz = c.double() + corr
            z = z + dz.to(z.dtype)
            outs.append(z)
        return torch.stack(outs, 1)


SPLIT = {"afterwm": AfterWMPlan}


def make_plan(row, module, model, cfg, device, generic_only=False):
    if generic_only:
        return ForecasterPlan(module, model, cfg, device)
    if row in SPLIT:
        name = getattr(module.forecaster(model, cfg), "name", row)
        return SPLIT[row](model, name, device)
    if row in ("grbal", "mole"):
        return AdaptOncePlan(module.forecaster(model, cfg), row, device)
    if row == "rssm":
        return RSSMPlan(module.forecaster(model, cfg), device)
    if row == "gpmm":
        return GPMMPlan(module.forecaster(model, cfg), device)
    return ForecasterPlan(module, model, cfg, device)


class GraphedEncode:

    def __init__(self, plan, co, ca, warmup=2):
        self.plan, self.status, self.graph = plan, "eager", None
        if not co.is_cuda:
            return
        try:
            self.s_o, self.s_a = co.detach().clone(), ca.detach().clone()
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(warmup):
                    plan.encode_t(self.s_o, self.s_a)
            torch.cuda.current_stream().wait_stream(st)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.s_out = plan.encode_t(self.s_o, self.s_a)
            torch.cuda.synchronize()
            self.graph, self.status = g, "graph"
        except Exception as exc:
            self.graph = None
            self.status = f"eager (capture failed: {type(exc).__name__}: {str(exc)[:200]})"
            torch.cuda.synchronize()

    def __call__(self, ctx_obs_n, ctx_act_n):
        _, _, co, ca = make_ctx_tensors(ctx_obs_n, ctx_act_n, self.plan.device)
        if self.graph is None:
            return self.plan.encode_t(co, ca)
        self.s_o.copy_(co)
        self.s_a.copy_(ca)
        self.graph.replay()
        return tuple(x.clone() for x in self.s_out)
