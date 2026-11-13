from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

S6_TOL = 1e-5


def predict_drop(model, state, z, af, drop_fast: bool, drop_slow: bool):
    import torch
    q, F, c, P = state
    fp, m = model.fp, model.core
    log_p, _ = fp.route(torch.cat((z, af[:, 0]), dim=-1), q)
    paths = []
    for k in range(F.shape[1]):
        st, Fk, ck = z, F[:, k], c[:, k]
        zero_c = torch.zeros_like(ck)
        pp = []
        for h in range(af.shape[1]):
            a_h = af[:, h]
            joint = torch.cat((st, a_h), dim=-1)
            d_base = fp.g0[k](joint)
            if drop_fast:
                d_hat = d_base
            else:
                expert = fp.gf[k]
                d_fast = expert(torch.cat((st, a_h, Fk), dim=-1)) - expert(torch.cat((st, a_h, torch.zeros_like(Fk)), dim=-1))
                d_hat = d_base + d_fast
            z_hat = st + d_hat
            if drop_slow:
                st = z_hat
            else:
                x = torch.cat((st, a_h, Fk, ck), dim=-1)
                x0 = torch.cat((st, a_h, Fk, zero_c), dim=-1)
                st = z_hat + (m.readout[k](x) - m.readout[k](x0))
            pp.append(st)
            Fk = fp.retention * Fk
        paths.append(torch.stack(pp, 1))
    return torch.stack(paths, -2), log_p


def main(argv=None):
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="End-of-context fast and slow states (probes) and forecasts with one "
                                             "correction removed at test time, for one AFTER-WM run.")
    P.add_cli_options(ap)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--run", required=True, help="AFTER-WM run directory (best.pt, PROTOCOL.json, COMPLETE.json)")
    ap.add_argument("--eval", default=None, help="evaluation directory of the run (default <run>/eval)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--state-splits", default="train,val,test,test_b")
    ap.add_argument("--forecast-splits", default="test,test_b")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    from analysis import statelog as SL
    from pipeline import data as D
    from pipeline import evaluate as E
    from pipeline import scorer as SC
    ds = P.DATASET
    os.makedirs(a.out, exist_ok=True)
    tag = f"{ds}__{a.seed}"
    out_npz, rep_path = os.path.join(a.out, tag + ".npz"), os.path.join(a.out, tag + "__REPORT.json")
    for p in (out_npz, rep_path):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite")
    R = SL.Row(a.run, a.eval, a.device)
    model, stats = R.model, R.stats
    own, common, _ = SC.frame_scales(stats)
    own_t, common_t = np.asarray(own, np.float64), np.asarray(common, np.float64)
    rep = {"dataset": ds, "seed": a.seed, "best_pt_sha256": R.best_sha, "gamma": float(model.gamma),
           "determinism": R.deterministic, "smoke": a.smoke,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu", "torch": torch.__version__,
           "splits": {}}

    def T(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(a.device)

    def steps(pred, logp, y):
        c_, l_, _ = SL.forecast_arrays_from(model, pred, logp)
        return SL.step_errors(c_, l_, y, own_t, common_t)

    save = {}
    fsplits = [s for s in a.forecast_splits.split(",") if s]
    ok = True
    for split in [s for s in a.state_splits.split(",") if s]:
        if split == "train":
            obs, act = D.load_visible("train")
            obs_n, act_n = stats.norm_obs(obs), stats.norm_act(act)
            e_, t_ = D.eval_queries(obs.shape[0])
            w = D.windows(obs_n, act_n, e_, t_, P.MAX_H)
            sq = {"ctx_obs": w["ctx_obs"], "ctx_act": w["ctx_act"], "fut_act": w["fut_act"], "fut_obs": w["fut_obs"],
                  "e": e_, "t": t_}
            chunk = 720
        else:
            sq = E.split_queries(split, stats)
            chunk = R.chunk(split)
        n = len(sq["e"]) if not a.smoke else min(len(sq["e"]), a.smoke)
        do_fc = split in fsplits
        rec = {"c": [], "F": []}
        stp = {"intact": [], "zero_fast": [], "zero_slow": []}
        arrs = {"comp": [], "lw": [], "mean": []}
        d_ident, d_slow0 = 0.0, 0.0
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            zc, ac, af, _ = model.native_inputs(T(sq["ctx_obs"][s0:s1]), T(sq["ctx_act"][s0:s1]), T(sq["fut_act"][s0:s1]))
            st = SL.replay(model, zc, ac)
            q, F, c, Pm = st
            rec["c"].append(c.reshape(c.shape[0], -1).float().cpu().numpy())
            rec["F"].append(F.reshape(F.shape[0], -1).float().cpu().numpy())
            if not do_fc:
                continue
            y = sq["fut_obs"][s0:s1]
            pred, logp = model.predict_fast(st, zc[:, -1], af)
            c_, l_, m_ = SL.forecast_arrays_from(model, pred, logp)
            arrs["comp"].append(c_), arrs["lw"].append(l_), arrs["mean"].append(m_)
            stp["intact"].append(SL.step_errors(c_, l_, y, own_t, common_t))
            p2, lp2 = predict_drop(model, st, zc[:, -1], af, False, False)
            d_ident = max(d_ident, float((p2 - pred).abs().max().item()), float((lp2 - logp).abs().max().item()))
            pf, lpf = predict_drop(model, st, zc[:, -1], af, True, False)
            stp["zero_fast"].append(steps(pf, lpf, y))
            ps_, lps = predict_drop(model, st, zc[:, -1], af, False, True)
            stp["zero_slow"].append(steps(ps_, lps, y))
            p0, _ = model.predict_fast(SL.variant_states(st)["slow0"], zc[:, -1], af)
            d_slow0 = max(d_slow0, float((p0 - ps_).abs().max().item()))
        for k, v in rec.items():
            save[f"{split}|{k}"] = np.concatenate(v, 0)
        save[f"{split}|e"] = np.asarray(sq["e"][:n], np.int64)
        save[f"{split}|t"] = np.asarray(sq["t"][:n], np.int64)
        chk = {"n": int(n), "chunk": chunk, "finite_states": bool(all(np.isfinite(save[f"{split}|{k}"]).all() for k in rec))}
        if do_fc:
            stored = R.stored(split)
            if not (np.array_equal(stored["e"][:n], sq["e"][:n]) and np.array_equal(stored["t"][:n], sq["t"][:n])):
                raise SystemExit(f"{split}: query order differs from the stored predictions")
            ci, li, mi = (np.concatenate(arrs[k], 0) for k in ("comp", "lw", "mean"))
            d6 = max(float(np.max(np.abs(ci.astype(np.float64) - stored["comp_mean"][:n]))),
                     float(np.max(np.abs(li.astype(np.float64) - stored["comp_logw"][:n]))),
                     float(np.max(np.abs(mi.astype(np.float64) - stored["mean"][:n]))))
            for v, xs in stp.items():
                save[f"{split}|step_{v}"] = np.concatenate(xs, 0)
            save[f"{split}|query_id"] = stored["query_id"][:n]
            fin = {v: bool(np.isfinite(save[f"{split}|step_{v}"]).all()) for v in stp}
            chk.update({"reproduction_max_abs_diff": d6, "reproduction_passed": bool(d6 <= S6_TOL),
                        "ident_max_abs_diff": d_ident, "ident_bitwise": bool(d_ident == 0.0),
                        "zero_slow_vs_c0_max_abs_diff": d_slow0, "zero_slow_equals_c0": bool(d_slow0 == 0.0), "finite": fin})
            ok = ok and chk["reproduction_passed"] and chk["ident_bitwise"] and chk["zero_slow_equals_c0"] and all(fin.values())
        ok = ok and chk["finite_states"]
        rep["splits"][split] = chk
        print(f"{tag} {split}: {json.dumps(chk)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(out_npz, **save)
    rep["npz_sha256"] = SL.sha256_file(out_npz)
    rep["passed"] = bool(ok)
    with open(rep_path, "x") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"{tag} passed={ok} ({time.time() - t0:.0f}s)", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
