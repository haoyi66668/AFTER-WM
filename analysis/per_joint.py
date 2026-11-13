from __future__ import annotations

import argparse

import numpy as np

from analysis import examples_select as ES
from afterwm_stats import load as L

JOINTS = {"panda": 7, "ur5e": 6}


def joints(ex):
    pj = ex["per_joint"]
    n = JOINTS[ex["dataset"]]
    t, c = np.asarray(pj["afterwm_step40"]), np.asarray(pj["compared_step40"])
    idx = [(i, n + i) for i in range(n)]
    tj = [float(t[list(p)].mean()) for p in idx]
    cj = [float(c[list(p)].mean()) for p in idx]
    return {"compared": ex["compared"], "seeds": pj["seeds"], "afterwm": tj, "compared_values": cj,
            "joints": n, "afterwm_lower": int(sum(1 for a, b in zip(tj, cj) if a < b))}


def run(outputs=None):
    ex = ES.load_examples(outputs)
    return {ds: joints(ex[ds]) for ds in ES.ROBOTS}


def from_predictions(compared, eval_template, seeds):
    from pipeline import evaluate as E
    from pipeline import registry as P
    from pipeline import scorer as S
    stats = E.get_stats()
    Y = {s: np.asarray(E.split_queries(s, stats)["fut_obs"], np.float64) for s in L.SPLITS}
    out = {}
    for mo in ("afterwm", compared):
        per = []
        for sd in seeds:
            parts = []
            for s in L.SPLITS:
                ev = eval_template.format(runs_root=P.RUNS_ROOT, dataset=P.DATASET, model=mo, seed=sd)
                with np.load(f"{ev}/{s}/PREDICTIONS.npz", allow_pickle=False) as z:
                    fs = set(z.files)
                    cm, lw = S.as_components(np.asarray(z["mean"]), np.asarray(z["comp_mean"]) if "comp_mean" in fs else None,
                                             np.asarray(z["comp_logw"]) if "comp_logw" in fs else None)
                mu = S.mixture_mean_np(np.asarray(cm, np.float64), np.asarray(lw, np.float64))
                parts.append((mu - Y[s]) ** 2)
            per.append(np.concatenate(parts).mean(0))
        out[mo] = np.mean(per, 0)
    return {"afterwm_step40": out["afterwm"][39].tolist(), "compared_step40": out[compared][39].tolist(), "seeds": list(seeds)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-joint step-40 errors of AFTER-WM and the compared model (all test queries).")
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--from-predictions", action="store_true",
                    help="recompute the per-dimension step-40 errors from the evaluation predictions of one dataset "
                         "(needs --dataset, --runs-root and --compared)")
    ap.add_argument("--compared", default=None)
    ap.add_argument("--eval-template", default="{runs_root}/runs/{dataset}/{model}/{seed}/eval")
    ap.add_argument("--seeds", default=",".join(map(str, L.SEEDS)))
    ap.add_argument("--out", required=True)
    a, _ = ap.parse_known_args(argv)
    if a.from_predictions:
        L.write_json(a.out, from_predictions(a.compared, a.eval_template, [int(x) for x in a.seeds.split(",")]))
    else:
        L.write_json(a.out, run(a.outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
