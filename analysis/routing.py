from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

S6_TOL = 1e-5


def main(argv=None):
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="Forecasts with the error term of the routing-memory update removed during the "
                                             "context replay, for one AFTER-WM run.")
    P.add_cli_options(ap)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    from analysis import statelog as SL
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
    fp = model.fp
    K = len(fp.g0)
    if "uq_raw" in fp.__dict__:
        raise SystemExit("uq_raw already patched")
    rep = {"dataset": ds, "seed": a.seed, "best_pt_sha256": R.best_sha, "gamma": float(model.gamma),
           "determinism": R.deterministic, "smoke": a.smoke,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu", "splits": {}}

    def T(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(a.device)

    def steps(pred, logp, y):
        c_, l_, _ = SL.forecast_arrays_from(model, pred, logp)
        return SL.step_errors(c_, l_, y, own_t, common_t), c_, l_

    def zeros_uq(x):
        return torch.zeros(x.shape[:-1] + (K,), dtype=x.dtype, device=x.device)

    save, ok = {}, True
    for split in ("test", "test_b"):
        sq = E.split_queries(split, stats)
        chunk = R.chunk(split)
        n = len(sq["e"]) if not a.smoke else min(len(sq["e"]), a.smoke)
        stp = {"intact": [], "route_noerr": []}
        arrs = {"comp": [], "lw": []}
        dq_max, restore = 0.0, 0.0
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            zc, ac, af, _ = model.native_inputs(T(sq["ctx_obs"][s0:s1]), T(sq["ctx_act"][s0:s1]), T(sq["fut_act"][s0:s1]))
            y = sq["fut_obs"][s0:s1]
            st = SL.replay(model, zc, ac)
            pred, logp = model.predict_fast(st, zc[:, -1], af)
            si, c_, l_ = steps(pred, logp, y)
            stp["intact"].append(si)
            arrs["comp"].append(c_)
            arrs["lw"].append(l_)
            fp.uq_raw = zeros_uq
            try:
                st_r = SL.replay(model, zc, ac)
                pred_r, logp_r = model.predict_fast(st_r, zc[:, -1], af)
            finally:
                del fp.uq_raw
            stp["route_noerr"].append(steps(pred_r, logp_r, y)[0])
            dq_max = max(dq_max, float((st_r[0] - st[0]).abs().max().item()))
            if s0 == 0:
                st2 = SL.replay(model, zc, ac)
                p2, _ = model.predict_fast(st2, zc[:, -1], af)
                restore = float((p2 - pred).abs().max().item())
        stored = R.stored(split, ("comp_mean", "comp_logw", "query_id", "e", "t"))
        if not (np.array_equal(stored["e"][:n], sq["e"][:n]) and np.array_equal(stored["t"][:n], sq["t"][:n])):
            raise SystemExit(f"{split}: query order differs from the stored predictions")
        ci, li = np.concatenate(arrs["comp"], 0), np.concatenate(arrs["lw"], 0)
        d6 = max(float(np.max(np.abs(ci.astype(np.float64) - stored["comp_mean"][:n]))),
                 float(np.max(np.abs(li.astype(np.float64) - stored["comp_logw"][:n]))))
        for v, xs in stp.items():
            save[f"{split}|step_{v}"] = np.concatenate(xs, 0)
        save[f"{split}|query_id"] = stored["query_id"][:n]
        fin = {v: bool(np.isfinite(save[f"{split}|step_{v}"]).all()) for v in stp}
        chk = {"n": int(n), "chunk": chunk, "reproduction_max_abs_diff": d6, "reproduction_passed": bool(d6 <= S6_TOL),
               "q_changed_max": dq_max, "q_changed": bool(dq_max > 0), "restore_max_abs_diff": restore,
               "restore_bitwise": bool(restore == 0.0), "finite": fin}
        ok = ok and chk["reproduction_passed"] and chk["q_changed"] and chk["restore_bitwise"] and all(fin.values())
        rep["splits"][split] = chk
        print(f"{tag} {split}: {json.dumps(chk)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(out_npz, **save)
    rep.update({"npz_sha256": SL.sha256_file(out_npz), "passed": bool(ok)})
    with open(rep_path, "x") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"{tag} passed={ok}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
