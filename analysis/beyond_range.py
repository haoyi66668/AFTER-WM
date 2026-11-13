from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPRO_TOL = 1e-3
HS = (1, 5, 10, 20, 40)
FILES = {"test": "escphc_test.npz", "test_b": "escphc_testB.npz"}


def main(argv=None):
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="One ESCP-HC run (trained with factors 1.5^U(-3,3)) evaluated on the test files "
                                             "with factors 1.5^U(-4,4), after reproducing its own test values.")
    P.add_cli_options(ap)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--model", required=True, help="model id used in the output file name")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--beyond-dir", default=None, help="directory with escphc_test.npz / escphc_testB.npz "
                                                       "(default <data-root>/escphc_beyond)")
    ap.add_argument("--rssm-eval-mode", default=None, help="mode|sample16 (default: the mode of the stored evaluation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    from analysis import statelog as SL
    from pipeline import data as D
    from pipeline import evaluate as E
    from pipeline import scorer as SC
    if P.DATASET != "escphc":
        raise SystemExit("--dataset must be escphc")
    beyond = a.beyond_dir or os.path.join(os.path.dirname(P.DATA_DIR.rstrip("/")), "escphc_beyond")
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.model}__{a.seed}"
    out_npz, rep_path = os.path.join(a.out, tag + ".npz"), os.path.join(a.out, tag + "__REPORT.json")
    for p in (out_npz, rep_path):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; refusing to overwrite")
    rd = a.run
    pdir = a.eval or os.path.join(rd, "eval")
    eprot = json.load(open(os.path.join(pdir, "EVALUATION_PROTOCOL.json")))
    dt = eprot.get("determinism")
    det = bool(dt.get("deterministic")) if isinstance(dt, dict) else False
    E.setup_runtime(a.device, det)
    comp = json.load(open(os.path.join(rd, "COMPLETE.json")))
    best_sha = SL.sha256_file(os.path.join(rd, "best.pt"))
    if comp.get("best_pt_sha256") not in (None, best_sha):
        raise SystemExit("best.pt sha256 differs from COMPLETE.json")
    reg_fc = eprot.get("forecaster") or {}
    reg_name = str(reg_fc.get("name", ""))
    mode = a.rssm_eval_mode
    if mode is None and "[" in reg_name and reg_name.endswith("]"):
        mode = reg_name[reg_name.index("[") + 1:-1]
    if mode is not None:
        from baselines import keyed_sampling as KS
        KS.set_eval_mode(mode)
    mod, model, cfg, prot, stats, ck = E.load_run(rd, device=a.device)
    fm = mod.forecaster(model, cfg)
    own, common, raw_scale = SC.frame_scales(stats)
    own_t, common_t = np.asarray(own, np.float64), np.asarray(common, np.float64)
    rep = {"model": a.model, "seed": a.seed, "best_pt_sha256": best_sha, "module": prot.get("module"),
           "forecaster": str(getattr(fm, "name", "")), "determinism": det, "smoke": a.smoke,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu", "reproduction": {}, "beyond": {}}

    def per_step(out, y):
        cm, lw = SC.as_components(out["mean"], out.get("comp_mean"), out.get("comp_logw"))
        return SL.step_errors(cm, lw, y, own_t, common_t)

    save = {}
    ok = True
    for split in ("test", "test_b"):
        sq = E.split_queries(split, stats)
        n = len(sq["e"]) if not a.smoke else min(len(sq["e"]), a.smoke)
        chunk = int(json.load(open(os.path.join(pdir, split, "RESULT.json"))).get("chunk_used") or 720)
        sub = {k: sq[k][:n] for k in ("ctx_obs", "ctx_act", "fut_act", "keys")}
        st = per_step(E.forecast_arrays(fm, sub, a.device, chunk), sq["fut_obs"][:n])
        with np.load(os.path.join(pdir, split, "PER_QUERY.npz"), allow_pickle=False) as z:
            pq = {h: np.asarray(z[f"nMSE_train__prefix_H{h}"], np.float64)[:n] for h in HS}
        new_f = float(np.mean([np.mean(st[:, :h].mean(1)) for h in HS]))
        reg_f = float(np.mean([np.mean(pq[h]) for h in HS]))
        rel = abs(new_f - reg_f) / abs(reg_f)
        rec = {"n": int(n), "chunk": chunk, "multi_step_new": new_f, "multi_step_stored": reg_f, "rel_diff": rel,
               "passed": bool(rel <= REPRO_TOL), "finite": bool(np.isfinite(st).all())}
        rep["reproduction"][split] = rec
        ok = ok and rec["passed"]
        save[f"own_{split}|step"] = st
        print(f"{tag} reproduction {split}: {json.dumps(rec)} ({time.time() - t0:.0f}s)", flush=True)
    if ok:
        for split, fn in FILES.items():
            path = os.path.join(beyond, fn)
            with np.load(path, allow_pickle=False) as z:
                obs = np.asarray(z["obs"], np.float32)
                act = np.asarray(z["act"], np.float32)
            assert obs.shape[1:] == (P.T_OBS, P.OBS_DIM) and act.shape[1:] == (P.T_ACT, P.ACT_DIM)
            obs_n, act_n = stats.norm_obs(obs), stats.norm_act(act)
            e_, t_ = D.eval_queries(obs.shape[0])
            w = D.windows(obs_n, act_n, e_, t_, P.MAX_H)
            n = len(e_) if not a.smoke else min(len(e_), a.smoke)
            sq = {k: np.ascontiguousarray(w[k][:n]) for k in ("ctx_obs", "ctx_act", "fut_act", "fut_obs")}
            sq["keys"] = D.context_keys(sq["ctx_obs"], sq["ctx_act"])
            out = E.forecast_arrays(fm, sq, a.device, int(rep["reproduction"]["test"]["chunk"]))
            st = per_step(out, sq["fut_obs"])
            save[f"beyond_{split}|step"] = st
            save[f"beyond_{split}|e"] = e_[:n]
            save[f"beyond_{split}|t"] = t_[:n]
            rep["beyond"][split] = {"sha256": SL.sha256_file(path), "n": int(n),
                                    "n_nonfinite_queries": int((~np.isfinite(st).all(1)).sum())}
            print(f"{tag} beyond {split}: {json.dumps(rep['beyond'][split])} ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(out_npz, **save)
    rep.update({"npz_sha256": SL.sha256_file(out_npz), "passed": bool(ok)})
    with open(rep_path, "x") as f:
        json.dump(rep, f, indent=1, default=str)
    print(f"{tag} passed={ok} ({time.time() - t0:.0f}s)", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
