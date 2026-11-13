from __future__ import annotations

import argparse
import json
import os

import numpy as np

from afterwm_stats import bootstrap as BT

ALPHAS = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4)
CS = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
SHUF_SEED = 20261022
REPS = ("c", "F", "c_shuf")
REP_NAME = {"c": "slow state", "F": "fast state", "c_shuf": "slow state, shuffled labels"}
STARTS = (64, 96, 128, 160, 192)
SPLITS = ("test", "test_b")
ALL = ("train", "val") + SPLITS
SPLIT_FILE = {"train": "train", "val": "val", "test": "test", "test_b": "testB"}
SPLIT_TAG = {"test": "TEST", "test_b": "TEST_B"}
ARMS = ("panda", "ur5e", "hopper")
CONTINUOUS = ("escphc", "panda", "ur5e", "hopper")


def load_data(data_root, ds, split):
    with np.load(os.path.join(data_root, ds, f"{ds}_{SPLIT_FILE[split]}.npz"), allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def grid(n_ep):
    return np.repeat(np.arange(n_ep), len(STARTS)).astype(np.int64), np.tile(np.asarray(STARTS), n_ep).astype(np.int64)


def window_labels(ds, arrays):
    e, t = grid(arrays["obs"].shape[0])
    j = t - 1
    out = {"e": e, "t": t}
    if ds == "jshc":
        out["cls"] = arrays["fail_joint"][e, j].astype(np.int64)
    elif ds == "escphc":
        out["targets"] = {k: np.log(arrays[k][e, j].astype(np.float64)) for k in ("grav_mult", "damp_mult", "mass_mult")}
    else:
        out["targets"] = {"payload": arrays["payload"][e, j].astype(np.float64),
                          "log_friction": np.log(arrays["friction"][e, j].astype(np.float64)),
                          "log_gain": np.log(arrays["gain"][e, j].astype(np.float64))}
    if "parent" in arrays:
        out["parent"] = arrays["parent"].astype(str)[e]
    return out


def standardize(Xtr, *Xs):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(X - mu) / sd for X in (Xtr,) + Xs]


def ridge_fit(X, y, alpha):
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ yc)
    return w, ym - xm @ w


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def bal_acc(y, p, classes):
    return float(np.mean([np.mean(p[y == k] == k) for k in classes if np.any(y == k)]))


def probe_continuous(Xtr, ytr, Xva, yva, Xte):
    Xtr_s, Xva_s, Xte_s = standardize(Xtr, Xva, Xte)
    m, s = ytr.mean(), ytr.std() if ytr.std() > 1e-12 else 1.0
    best = None
    for a in ALPHAS:
        w, b = ridge_fit(Xtr_s, (ytr - m) / s, a)
        sc = r2(yva, (Xva_s @ w + b) * s + m)
        if best is None or sc >= best[0]:
            best = (sc, a)
    w, b = ridge_fit(np.vstack([Xtr_s, Xva_s]), (np.concatenate([ytr, yva]) - m) / s, best[1])
    return (Xte_s @ w + b) * s + m, {"alpha": best[1], "val_score": float(best[0])}


def probe_logistic(Xtr, ytr, Xva, yva, Xte):
    from sklearn.linear_model import LogisticRegression
    Xtr_s, Xva_s, Xte_s = standardize(Xtr, Xva, Xte)
    classes = np.unique(ytr)
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000, tol=1e-6)
        clf.fit(Xtr_s, ytr)
        sc = bal_acc(yva, clf.predict(Xva_s), classes)
        if best is None or sc > best[0]:
            best = (sc, C)
    clf = LogisticRegression(C=best[1], penalty="l2", solver="lbfgs", max_iter=5000, tol=1e-6)
    clf.fit(np.vstack([Xtr_s, Xva_s]), np.concatenate([ytr, yva]))
    return clf.predict(Xte_s), {"C": best[1], "val_score": float(best[0])}


class ProbeBoot:
    def __init__(self, draws):
        self.d = draws

    def _sums(self, s, x):
        P = len(self.d[s]["parents"])
        return np.bincount(self.d[s]["pidx"], weights=x, minlength=P)

    def r2(self, y, p):
        n = sy = syy = sse = 0.0
        nb = syb = syyb = sseb = 0.0
        for s in SPLITS:
            C = self.d[s]["C"]
            parts = [self._sums(s, v) for v in (np.ones_like(y[s]), y[s], y[s] ** 2, (y[s] - p[s]) ** 2)]
            n, sy, syy, sse = n + parts[0].sum(), sy + parts[1].sum(), syy + parts[2].sum(), sse + parts[3].sum()
            nb, syb, syyb, sseb = nb + C @ parts[0], syb + C @ parts[1], syyb + C @ parts[2], sseb + C @ parts[3]
        pt = 1.0 - sse / (syy - sy ** 2 / n)
        bt = 1.0 - sseb / (syyb - syb ** 2 / nb)
        return float(pt), bt

    def bal(self, y, p, classes):
        rp, rb = [], []
        for k in classes:
            n = c = 0.0
            nb = cb = 0.0
            for s in SPLITS:
                C = self.d[s]["C"]
                a = self._sums(s, (y[s] == k).astype(float))
                b = self._sums(s, ((y[s] == k) & (p[s] == k)).astype(float))
                n, c, nb, cb = n + a.sum(), c + b.sum(), nb + C @ a, cb + C @ b
            if n > 0:
                rp.append(c / n)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rb.append(cb / nb)
        rb = np.vstack(rb)
        return float(np.mean(rp)), np.nanmean(rb, 0)


def test_parents(lab):
    return {s: np.asarray([f"{SPLIT_TAG[s]}:{p}" for p in lab[s]["parent"]]) for s in SPLITS}


def run_dataset(ds, rows, data_root):
    lab = {s: window_labels(ds, load_data(data_root, ds, s)) for s in ALL}
    pb = ProbeBoot(BT.make_draws(test_parents(lab)))
    rng = np.random.default_rng(SHUF_SEED)
    perm_tr = rng.permutation(len(lab["train"]["e"]))
    perm_va = rng.permutation(len(lab["val"]["e"]))
    res = {r: {} for r in REPS}
    for sd in sorted(rows):
        z = rows[sd]
        for s in ALL:
            if not (np.array_equal(z[f"{s}|e"], lab[s]["e"]) and np.array_equal(z[f"{s}|t"], lab[s]["t"])):
                raise ValueError(f"{ds}/{sd}/{s}: window grid differs")
        feats = {r: {s: z[f"{s}|{r}"].astype(np.float64) for s in ALL} for r in ("c", "F")}
        feats["c_shuf"] = feats["c"]

        def fit_eval(r, ytr, yva, kind):
            X = feats[r]
            Xte = np.vstack([X[s] for s in SPLITS])
            if r == "c_shuf":
                ytr, yva = ytr[perm_tr], yva[perm_va]
            if kind == "cont":
                p, info = probe_continuous(X["train"], ytr, X["val"], yva, Xte)
            else:
                p, info = probe_logistic(X["train"], ytr, X["val"], yva, Xte)
            out, i0 = {}, 0
            for s in SPLITS:
                n = len(X[s])
                out[s] = p[i0:i0 + n]
                i0 += n
            return out, info

        for r in REPS:
            if ds == "jshc":
                y = {s: lab[s]["cls"] for s in ALL}
                p, info = fit_eval(r, y["train"], y["val"], "multi")
                pt, bt = pb.bal({s: y[s] for s in SPLITS}, p, np.unique(y["train"]))
                res[r][sd] = {"score": pt, "draw": bt, "per_target": {"fail_joint": pt}, "info": {"fail_joint": info}}
            else:
                pts, bts, per, infos = [], [], {}, {}
                for tg in lab["train"]["targets"]:
                    y = {s: lab[s]["targets"][tg] for s in ALL}
                    p, info = fit_eval(r, y["train"], y["val"], "cont")
                    pt, bt = pb.r2({s: y[s] for s in SPLITS}, p)
                    pts.append(pt)
                    bts.append(bt)
                    per[tg], infos[tg] = pt, info
                res[r][sd] = {"score": float(np.mean(pts)), "draw": np.mean(bts, 0), "per_target": per, "info": infos}
    seeds = sorted(rows)
    return {"metric": "balanced accuracy" if ds == "jshc" else "mean R2 over the change parameters", "seeds": seeds,
            "scores": {r: {"mean": float(np.mean([res[r][sd]["score"] for sd in seeds])),
                           "ci95": [float(np.nanquantile(np.mean([res[r][sd]["draw"] for sd in seeds], 0), q)) for q in (0.025, 0.975)],
                           "seed": [res[r][sd]["score"] for sd in seeds],
                           "per_target_seed": {tg: [res[r][sd]["per_target"][tg] for sd in seeds]
                                               for tg in res[r][seeds[0]]["per_target"]},
                           "probe_info": {str(sd): res[r][sd]["info"] for sd in seeds}} for r in REPS}}


def load_rows(rows_dir, ds, seeds):
    out = {}
    for sd in seeds:
        with np.load(os.path.join(rows_dir, f"{ds}__{sd}.npz"), allow_pickle=False) as z:
            out[sd] = {k: np.asarray(z[k]) for k in z.files if k.split("|")[-1] in ("c", "F", "e", "t")}
    return out


def summarize(P):
    mean = {ds: {r: float(np.mean(P[ds]["scores"][r]["seed"])) for r in REPS} for ds in P}
    cont = [ds for ds in CONTINUOUS if ds in P]
    out = {"mean": mean}
    if cont:
        out["fast_r2_range"] = [min(mean[ds]["F"] for ds in cont), max(mean[ds]["F"] for ds in cont)]
        out["slow_r2_range"] = [min(mean[ds]["c"] for ds in cont), max(mean[ds]["c"] for ds in cont)]
        out["shuffled_r2_max"] = max(mean[ds]["c_shuf"] for ds in cont)
    if "jshc" in P:
        out["jshc_balanced_accuracy"] = {r: mean["jshc"][r] for r in REPS}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Linear probes of the fast and slow states for the change parameters.")
    ap.add_argument("--rows", help="directory with <dataset>__<seed>.npz state rows (written by analysis.removal)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--datasets", default="jshc,escphc,panda,ur5e,hopper")
    ap.add_argument("--seeds", default=None, help="dataset:seed,seed;... (default: all files found)")
    ap.add_argument("--scores", help="existing probe scores (outputs/analysis/probes.json): summarize only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.scores:
        with open(a.scores) as f:
            P = json.load(f)
        res = {"summary": summarize(P)}
    else:
        want = {}
        if a.seeds:
            for part in a.seeds.split(";"):
                d, s = part.split(":")
                want[d] = [int(x) for x in s.split(",")]
        P = {}
        for ds in a.datasets.split(","):
            seeds = want.get(ds) or sorted(int(f.split("__")[1].split(".")[0]) for f in os.listdir(a.rows)
                                           if f.startswith(ds + "__") and f.endswith(".npz"))
            P[ds] = run_dataset(ds, load_rows(a.rows, ds, seeds), a.data_root)
        res = P
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
