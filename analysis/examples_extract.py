from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SEED = 20260910
KEYS = (1, 5, 10, 20, 40)
HS = (1, 5, 10, 20, 40)


def main(argv=None):
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="Extract the selected robot-arm example (true sequence, predictions of AFTER-WM and "
                                             "the compared model, per-step errors) for rendering.")
    P.add_cli_options(ap)
    ap.add_argument("--examples", required=True, help="examples.json written by analysis.examples_select")
    ap.add_argument("--eval-template", default="{runs_root}/runs/{dataset}/{model}/{seed}/eval",
                    help="evaluation directory of a run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from pipeline import evaluate as E
    from pipeline import scorer as S
    ds = P.DATASET
    sel = json.load(open(a.examples))["selection"][ds]
    M, comp = sel["selected"], sel["compared"]
    stats = E.get_stats()
    store = {"obs_mean": stats.arrays()[0], "obs_std": stats.arrays()[1]}
    s, i = M["split"], M["idx"]
    sq = E.split_queries(s, stats)
    y = np.asarray(sq["fut_obs"], np.float64)
    for mo in ("afterwm", comp):
        ev = a.eval_template.format(runs_root=P.RUNS_ROOT, dataset=ds, model=mo, seed=SEED)
        with np.load(os.path.join(ev, s, "PREDICTIONS.npz"), allow_pickle=False) as z:
            fs = set(z.files)
            cm, lw = S.as_components(np.asarray(z["mean"]), np.asarray(z["comp_mean"]) if "comp_mean" in fs else None,
                                     np.asarray(z["comp_logw"]) if "comp_logw" in fs else None)
            if str(np.asarray(z["query_id"])[i]) != M["query_id"]:
                raise SystemExit(f"{ev}: query order differs")
        mu = S.mixture_mean_np(np.asarray(cm[i:i + 1], np.float64), np.asarray(lw[i:i + 1], np.float64))
        e2 = (mu - y[i:i + 1]) ** 2
        nm = e2.mean(-1)[0]
        tag = "afterwm" if mo == "afterwm" else "compared"
        store[f"{tag}|pred_raw"] = stats.denorm_obs(mu)[0].astype(np.float64)
        store[f"{tag}|nmse_step"] = nm.astype(np.float64)
        store[f"{tag}|multi_step"] = np.float64(np.mean([nm[:h].mean() for h in HS]))
    with np.load(os.path.join(P.DATA_DIR, f"{ds}_{'test' if s == 'test' else 'testB'}.npz"), allow_pickle=False) as z:
        L = {k: np.asarray(z[k]) for k in z.files}
    store["obs_seq"] = L["obs"][M["e"]].astype(np.float64)
    store["t0"] = np.int64(M["t0"])
    for k in ("friction", "gain", "payload", "wrench", "impulse", "seg"):
        if k in L:
            store[f"lab|{k}"] = L[k][M["e"]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, compared=np.asarray(comp), query_id=np.asarray(M["query_id"]), **store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
