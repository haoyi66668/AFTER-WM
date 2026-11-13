from __future__ import annotations

import argparse
import json
import os

import numpy as np

from control import spec as SP

B_BOOT = 10000
BOOT_SEED = 20261007


def episode_return(rec):
    r = np.asarray(rec["reward"], np.float64)
    return r.sum(), r


def load_runs(runs, rows, seeds, n_ep):
    vals = np.full((len(rows), len(seeds), n_ep), np.nan)
    present = np.zeros((len(rows), len(seeds), n_ep), bool)
    for i, row in enumerate(rows):
        for j, s in enumerate(seeds):
            for e in range(n_ep):
                fn = os.path.join(runs, row, str(s), f"ep_{e:03d}.npz")
                if not os.path.exists(fn):
                    continue
                rec = np.load(fn)
                if not (bool(rec["warm_match"]) and bool(rec["anchor_match"])) or len(rec["reward"]) != SP.K_MPC:
                    continue
                g, r = episode_return(rec)
                if not np.isfinite(r).all():
                    continue
                vals[i, j, e] = g
                present[i, j, e] = True
    return vals, present


def load_returns(path):
    with np.load(path, allow_pickle=False) as z:
        return ([str(x) for x in z["rows"]], [int(x) for x in z["seeds"]], np.asarray(z["returns"], np.float64),
                np.asarray(z["valid"], bool))


def boot_idx(n_s, n_e, seed=BOOT_SEED, B=B_BOOT):
    rng = np.random.default_rng(seed)
    si = rng.integers(0, n_s, size=(B, n_s))
    ei = rng.integers(0, n_e, size=(B, n_e))
    return si, ei


def boot_mean(X, si, ei):
    out = np.empty(si.shape[0])
    for b0 in range(0, si.shape[0], 500):
        s, e = si[b0:b0 + 500], ei[b0:b0 + 500]
        out[b0:b0 + 500] = X[s[:, :, None], e[:, None, :]].mean((1, 2))
    return out


def ci(dist):
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return [float(lo), float(hi)]


def compare(A, Bm, si, ei):
    D = A - Bm
    point = float(D.mean())
    seed_diff = D.mean(1)
    return {"point": point, "ci95": ci(boot_mean(D, si, ei)), "seed_diffs": seed_diff.tolist(),
            "seeds_better": int((seed_diff > 0).sum()), "n_seeds": int(D.shape[0]), "rival_mean": float(Bm.mean()),
            "afterwm_mean": float(A.mean())}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def prediction_error(outputs=None):
    from afterwm_stats import aggregate as AG
    from afterwm_stats import load as L
    qs = L.per_query(SP.DATASET, outputs)
    return {row: AG.model_table(qs, row)["multi_step"]["mean"] for row in SP.MODEL_ROWS}


def analyse(rows, seeds, vals, present, pred_err=None):
    common = np.all(present.all(1), 0)
    E_idx = np.flatnonzero(common)
    si, ei = boot_idx(len(seeds), len(E_idx))
    X = {r: vals[i][:, E_idx] for i, r in enumerate(rows)}
    res = {"rows": {}, "comparisons": {}, "n_common_episodes": int(len(E_idx)), "n_seeds": len(seeds)}
    for r in rows:
        res["rows"][r] = {"mean": float(X[r].mean()), "ci95": ci(boot_mean(X[r], si, ei)), "seed_means": X[r].mean(1).tolist()}
    for r in SP.COMPARATORS:
        res["comparisons"][r] = compare(X["afterwm"], X[r], si, ei)
    means = {r: res["rows"][r]["mean"] for r in rows}
    rank = sorted(rows, key=lambda r: -means[r])
    gaps = {r: c["point"] for r, c in res["comparisons"].items()}
    second = rank[1]
    res["rank"] = rank
    res["summary"] = {"afterwm_mean_return": means["afterwm"], "afterwm_highest": rank[0] == "afterwm", "second": second,
                      "gap_to_second": means["afterwm"] - means[second],
                      "beaten": int(sum(1 for g in gaps.values() if g > 0)), "of": len(gaps),
                      "smallest_gain": min(gaps.values()), "closest": min(gaps, key=gaps.get)}
    if pred_err is not None:
        p_rows = [r for r in rank if r in pred_err]
        res["prediction_link"] = {"multi_step_error": {r: pred_err[r] for r in p_rows},
                                  "spearman": spearman([-pred_err[r] for r in p_rows], [means[r] for r in p_rows]),
                                  "n_rows": len(p_rows)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Returns of the JSHC control experiment, gaps and the rank correlation with "
                                             "the prediction error.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--runs", help="directory <runs>/<row>/<seed>/ep_###.npz written by control.run")
    g.add_argument("--returns", help="compact returns file (outputs/control/returns.npz)")
    ap.add_argument("--n-episodes", type=int, default=SP.N_EPISODES["test"])
    ap.add_argument("--outputs", default=None, help="outputs directory with per_query/jshc.npz (prediction errors)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.returns:
        rows, seeds, vals, present = load_returns(a.returns)
    else:
        rows, seeds = list(SP.MODEL_ROWS), list(SP.SEEDS)
        vals, present = load_runs(a.runs, rows, seeds, a.n_episodes)
    res = analyse(rows, seeds, vals, present, prediction_error(a.outputs))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
