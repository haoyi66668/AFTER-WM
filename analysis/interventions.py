from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HS = (1, 5, 10, 20, 40)
PREFIX = [f"nMSE_train__prefix_H{h}" for h in HS]
VARIANTS = ("intact", "ident", "base", "zero", "shuf", "delay4")
PERM_SEED = 20261015
DELAY = 4
S6_TOL = 1e-5


def donor_perm(parent: np.ndarray, seed: int = PERM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(parent)
    perm = rng.permutation(n)
    for _ in range(100):
        bad = np.flatnonzero(parent[perm] == parent)
        if len(bad) == 0:
            break
        for i in bad:
            for _t in range(10000):
                j = int(rng.integers(n))
                if parent[perm[j]] != parent[i] and parent[perm[i]] != parent[j]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    if not (np.array_equal(np.sort(perm), np.arange(n)) and np.all(parent[perm] != parent)):
        raise RuntimeError("donor permutation failed")
    return perm


def fast_update_e(fp, z, a, p, log_p, F, e_raw):
    import torch
    e_safe = fp.clip(e_raw, fp.qcfg.residual_clip_norm, fp.qcfg.epsilon)
    x_with = torch.cat((z, a, p, F.flatten(start_dim=-2), e_safe.flatten(start_dim=-2)), dim=-1)
    x_without = torch.cat((z, a, p, F.flatten(start_dim=-2), torch.zeros_like(e_safe).flatten(start_dim=-2)), dim=-1)
    q_delta = fp.qcfg.logit_step_limit * torch.tanh(fp.uq_raw(x_with) - fp.uq_raw(x_without))
    posterior_logits = log_p.to(dtype=torch.float32) + q_delta
    log_q = torch.log_softmax(posterior_logits, dim=-1)
    q = log_q.exp()
    evidence_safe = fp.clip(e_raw, fp.ufcfg.residual_clip_norm, fp.ufcfg.epsilon)
    uf_with = fp.uf_input(z, a, p, q, F, evidence_safe)
    uf_without = fp.uf_input(z, a, p, q, F, torch.zeros_like(evidence_safe))
    innovation = fp.uf_innovation(uf_with, uf_without)
    F_next = (fp.retention * F) + (fp.write * innovation)
    return q, F_next


def observe_e(model, state, z, a, zn, pick):
    import torch
    q, F, c, P = state
    fp, m = model.fp, model.core
    D = int(z.shape[-1])
    joint = torch.cat((z, a), dim=-1)
    log_p, p = fp.route(joint, q)
    z_hat = fp.experts_zhat(z, a, F, joint)
    err = zn[:, None, :] - z_hat
    eF, eC = pick(err)
    q_new, F_new = fast_update_e(fp, z, a, p, log_p, F, -eF)
    cs, ps = [], []
    for k in range(F.shape[1]):
        Fk = F[:, k]
        out = m.measurement[k](torch.cat((z, a, Fk), dim=-1))
        J = torch.tanh(out[..., :3 * D].reshape(*out.shape[:-1], D, 3))
        bias = out[..., 3 * D:]
        y = eC[:, k] - bias
        r = 0.05 + torch.nn.functional.softplus(m.filter[k](torch.cat((z, a, Fk), dim=-1)))
        Pk = P[:, k].clamp_min(1e-4)
        JPJ = (J.square() * Pk[:, None, :]).sum(dim=1)
        S = JPJ + r + 1e-3
        innov = (J * y[:, :, None]).sum(dim=1)
        gain = Pk / S.clamp_min(1e-3)
        c_next = c[:, k] + gain * innov / float(D)
        c_next = torch.tanh(c_next)
        p_next = ((1.0 - (Pk[:, None, :] * J.square() / S[:, None, :]).mean(dim=1)).clamp(0.02, 1.0) * Pk + 0.01)
        cs.append(c_next)
        ps.append(p_next)
    return (q_new, F_new, torch.stack(cs, dim=1), torch.stack(ps, dim=1)), err, p


def replay_e(model, zc, ac, mode, donor=None, keep_err=False):
    import torch
    state = model.init_state(zc.shape[0], zc)
    errs, buf, last = [], [], None
    for j in range(ac.shape[1]):
        def pick(err, j=j):
            if mode == "ident":
                return err, err
            if mode == "zero":
                z0 = torch.zeros_like(err)
                return z0, z0
            if mode == "shuf":
                return donor[:, j], donor[:, j]
            if mode == "delay4":
                e = buf[j - DELAY] if j >= DELAY else torch.zeros_like(err)
                return e, e
            raise ValueError(mode)
        state, err, p = observe_e(model, state, zc[:, j], ac[:, j], zc[:, j + 1], pick)
        if mode == "delay4":
            buf.append(err)
        if keep_err:
            errs.append(err)
        last = (err, p)
    return state, (torch.stack(errs, 1) if keep_err else None), last


def main(argv=None):
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="Test-time alterations of the context prediction errors of one AFTER-WM run "
                                             "(corrections removed, errors zeroed, errors of another sequence, errors "
                                             "delayed by 4 steps).")
    P.add_cli_options(ap)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    from analysis import statelog as SL
    from pipeline import evaluate as E
    from pipeline import scorer as SC
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    rep_path = os.path.join(a.out, f"afterwm__{a.seed}__REPORT.json")
    if os.path.exists(rep_path):
        raise SystemExit(f"{rep_path} exists; refusing to overwrite")
    R = SL.Row(a.run, a.eval, a.device)
    model, stats, pdir = R.model, R.stats, R.eval_dir
    own, common, raw_scale = SC.frame_scales(stats)
    own_t, common_t = np.asarray(own, np.float64), np.asarray(common, np.float64)
    rep = {"dataset": P.DATASET, "seed": int(a.seed), "best_pt_sha256": R.best_sha, "gamma": float(model.gamma),
           "determinism": R.deterministic, "smoke": a.smoke,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu", "torch": torch.__version__,
           "splits": {}}

    def T(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(a.device)

    def forecast(state, zc, af):
        pred, logp = model.predict_fast(state, zc[:, -1], af)
        return SL.forecast_arrays_from(model, pred, logp)

    def scores(c_, l_, y):
        ps = SC.point_scores(c_, l_, y, own, common, raw_scale)
        return np.stack([np.asarray(ps[k], np.float64) for k in PREFIX], -1)

    ok_all = True
    for split in ("test", "test_b"):
        sq = E.split_queries(split, stats)
        stored = R.stored(split)
        with np.load(os.path.join(pdir, split, "PER_QUERY.npz"), allow_pickle=False) as z:
            pq = {k: np.asarray(z[k]) for k in PREFIX + ["query_id", "parent"]}
        if not (np.array_equal(stored["e"], sq["e"]) and np.array_equal(stored["t"], sq["t"])
                and np.array_equal(stored["query_id"], pq["query_id"])):
            raise SystemExit(f"{split}: query order differs")
        n = len(sq["e"]) if not a.smoke else min(len(sq["e"]), a.smoke)
        parent = pq["parent"].astype(str)
        perm = donor_perm(parent[:n])
        chunk = R.chunk(split)
        out = {v: [] for v in VARIANTS}
        stp = {v: [] for v in VARIANTS}
        errs, d_state, d_pred = [], 0.0, 0.0
        arrs = {"comp": [], "lw": [], "mean": []}
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            zc, ac, af, _ = model.native_inputs(T(sq["ctx_obs"][s0:s1]), T(sq["ctx_act"][s0:s1]), T(sq["fut_act"][s0:s1]))
            y = sq["fut_obs"][s0:s1]
            st_i = SL.replay(model, zc, ac)
            c_, l_, m_ = forecast(st_i, zc, af)
            arrs["comp"].append(c_), arrs["lw"].append(l_), arrs["mean"].append(m_)
            out["intact"].append(scores(c_, l_, y))
            stp["intact"].append(SL.step_errors(c_, l_, y, own_t, common_t))
            st_r, er, _ = replay_e(model, zc, ac, "ident", keep_err=True)
            errs.append(er)
            d_state = max([d_state] + [float((x - y_).abs().max().item()) for x, y_ in zip(st_i, st_r)])
            c2, l2, _ = forecast(st_r, zc, af)
            d_pred = max(d_pred, float(np.max(np.abs(c2 - c_))), float(np.max(np.abs(l2 - l_))))
            out["ident"].append(scores(c2, l2, y))
            stp["ident"].append(SL.step_errors(c2, l2, y, own_t, common_t))
            c3, l3, _ = forecast(SL.variant_states(st_i)["base"], zc, af)
            out["base"].append(scores(c3, l3, y))
            stp["base"].append(SL.step_errors(c3, l3, y, own_t, common_t))
        err_all = torch.cat(errs, 0)
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            zc, ac, af, _ = model.native_inputs(T(sq["ctx_obs"][s0:s1]), T(sq["ctx_act"][s0:s1]), T(sq["fut_act"][s0:s1]))
            y = sq["fut_obs"][s0:s1]
            donor = err_all[torch.as_tensor(perm[s0:s1], device=err_all.device)]
            for v in ("zero", "shuf", "delay4"):
                st_v, _, _ = replay_e(model, zc, ac, v, donor=donor)
                c_, l_, _ = forecast(st_v, zc, af)
                out[v].append(scores(c_, l_, y))
                stp[v].append(SL.step_errors(c_, l_, y, own_t, common_t))
        res = {v: np.concatenate(out[v], 0) for v in VARIANTS}
        comp_i, lw_i, mean_i = (np.concatenate(arrs[k], 0) for k in ("comp", "lw", "mean"))
        d6 = max(float(np.max(np.abs(comp_i.astype(np.float64) - stored["comp_mean"][:n]))),
                 float(np.max(np.abs(lw_i.astype(np.float64) - stored["comp_logw"][:n]))),
                 float(np.max(np.abs(mean_i.astype(np.float64) - stored["mean"][:n]))))
        pq_arr = np.stack([pq[k][:n].astype(np.float64) for k in PREFIX], -1)
        pq_rel = float(np.max(np.abs(res["intact"] - pq_arr) / np.maximum(np.abs(pq_arr), 1e-300)))
        ident_ok = bool(d_state == 0.0 and d_pred == 0.0 and np.array_equal(res["ident"], res["intact"]))
        finite = {v: bool(np.isfinite(res[v]).all()) for v in VARIANTS}
        chk = {"n_queries": int(n), "reproduction_max_abs_diff": d6, "reproduction_passed": bool(d6 <= S6_TOL),
               "per_query_prefix_max_rel_diff": pq_rel, "per_query_ok": bool(pq_rel <= 1e-5),
               "ident_state_max_abs_diff": d_state, "ident_pred_max_abs_diff": d_pred, "ident_bitwise": ident_ok,
               "finite": finite, "perm_sha256": hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()}
        ok_all = ok_all and chk["reproduction_passed"] and chk["per_query_ok"] and ident_ok and all(finite.values())
        save = {"query_id": pq["query_id"][:n], "parent": pq["parent"][:n], "perm": perm, "metrics": np.asarray(PREFIX)}
        for v in VARIANTS:
            save[v] = res[v]
            save[v + "__step"] = np.concatenate(stp[v], 0)
        np.savez_compressed(os.path.join(a.out, f"afterwm__{a.seed}__{split}.npz"), **save)
        rep["splits"][split] = chk
        print(f"{a.seed} {split}: {json.dumps(chk)} ({time.time() - t0:.0f}s)", flush=True)
    rep["passed"] = bool(ok_all)
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=1, default=str)
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
