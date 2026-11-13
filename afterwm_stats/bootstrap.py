from __future__ import annotations

import numpy as np

BOOT_SEED = 20261007
B_DEFAULT = 10000
ORDER = ("val", "test", "test_b")
STREAM_SLOTS = 5
HS = (1, 5, 10, 20, 40)
SPLITS = ("test", "test_b")


def make_draws(parents: dict, B: int = B_DEFAULT, seed: int = BOOT_SEED) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for s in ORDER + (None,) * (STREAM_SLOTS - len(ORDER)):
        if s in parents:
            p = np.unique(parents[s])
            out[s] = {"parents": p, "pidx": np.searchsorted(p, parents[s]),
                      "C": rng.multinomial(len(p), np.full(len(p), 1.0 / len(p)), size=B).astype(np.float64)}
        else:
            rng.multinomial(12, np.full(12, 1.0 / 12), size=B)
    return out


def _parts(vals, weights, members, draws, seeds):
    parts = []
    for s in members:
        P = len(draws[s]["parents"])
        pidx = draws[s]["pidx"]
        Sm = np.zeros((len(seeds), P))
        Wm = np.zeros((len(seeds), P))
        for i, sd in enumerate(seeds):
            x = np.asarray(vals[sd][s], np.float64)
            w = np.ones_like(x) if weights is None else np.asarray(weights[sd][s], np.float64)
            x = np.where(w > 0, x, 0.0)
            Sm[i] = np.bincount(pidx, weights=w * x, minlength=P)
            Wm[i] = np.bincount(pidx, weights=w, minlength=P)
        parts.append((Sm, Wm, draws[s]["C"]))
    return parts


def stat(vals, members, draws, seeds, weights=None):
    parts = _parts(vals, weights, members, draws, seeds)
    num_pt = sum(Sm.sum(1) for Sm, _, _ in parts)
    den_pt = sum(Wm.sum(1) for _, Wm, _ in parts)
    with np.errstate(all="ignore"):
        num_b = sum(C @ Sm.T for Sm, _, C in parts)
        den_b = sum(C @ Wm.T for _, Wm, C in parts)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = num_pt / den_pt
        boot = num_b / den_b
    if weights is None:
        return point, boot.mean(1)
    return point, np.nanmean(boot, 1) if np.isnan(boot).any() else boot.mean(1)


def summarize(point, boot, ref_mean=None) -> dict:
    point = np.asarray(point, np.float64)
    b = boot[np.isfinite(boot)]
    lo, hi = (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) else (float("nan"), float("nan"))
    r = {"mean": float(point.mean()), "seed_values": point.tolist(), "n_seeds": int(len(point)), "ci95": [lo, hi],
         "boot_finite": int(len(b)), "boot_total": int(len(boot)),
         "all_seeds_same_direction": bool((point > 0).all() or (point < 0).all()),
         "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    if ref_mean is not None and ref_mean != 0:
        r["ref_mean"] = float(ref_mean)
        r["pct"] = 100.0 * r["mean"] / ref_mean
        r["pct_ci95"] = [100.0 * lo / ref_mean, 100.0 * hi / ref_mean]
        r["pct_seed_values"] = [100.0 * v / ref_mean for v in point.tolist()]
    return r


def paired(va, vb, members, draws, seeds, weights=None, vref=None):
    vr = vb if vref is None else vref
    d = {sd: {s: np.asarray(va[sd][s], np.float64) - np.asarray(vb[sd][s], np.float64) for s in members} for sd in seeds}
    w = None if weights is None else {sd: {s: weights[s] for s in members} for sd in seeds}
    pt, bt = stat(d, members, draws, seeds, w)
    rpt, _ = stat(vr, members, draws, seeds, w)
    out = summarize(pt, bt, float(np.mean(rpt)))
    out["n_queries"] = int(sum(len(np.asarray(va[seeds[0]][s])) if weights is None else int(np.sum(np.asarray(weights[s]) > 0))
                               for s in members))
    return out


def model_stat(vals, members, draws, seeds):
    pt, bt = stat(vals, members, draws, seeds)
    r = summarize(pt, bt)
    r["sd"] = float(np.std(pt, ddof=1)) if len(pt) > 1 else None
    return r


def finite_pair_weights(va, vb, members, seeds, weights=None):
    out = {}
    for sd in seeds:
        out[sd] = {}
        for s in members:
            base = np.ones(len(np.asarray(va[sd][s]))) if weights is None else np.asarray(weights[s], np.float64)
            out[sd][s] = base * np.isfinite(va[sd][s]) * np.isfinite(vb[sd][s])
    return out


def pct_seedwise(va, vb, members, draws, seeds, weights=None, finite_pairs=False):
    if finite_pairs:
        ww = finite_pair_weights(va, vb, members, seeds, weights)
    else:
        ww = {sd: {s: (np.ones(len(np.asarray(va[sd][s]))) if weights is None else np.asarray(weights[s], np.float64))
                   for s in members} for sd in seeds}
    d = {sd: {s: np.where(ww[sd][s] > 0, va[sd][s] - vb[sd][s], 0.0) for s in members} for sd in seeds}
    r = {sd: {s: np.where(ww[sd][s] > 0, vb[sd][s], 0.0) for s in members} for sd in seeds}
    pt, bt = stat(d, members, draws, seeds, ww)
    rpt, _ = stat(r, members, draws, seeds, ww)
    ref = float(np.mean(rpt))
    nq = int(sum(np.sum(ww[seeds[0]][s] > 0) for s in members))
    return {"pct": 100.0 * float(np.mean(pt)) / ref, "seed": (100.0 * pt / ref).tolist(), "draw": 100.0 * bt / ref,
            "ref": ref, "n_q": nq, "seeds": list(seeds)}


def verdict(point, seed_vals, draws, sign):
    b = np.asarray(draws, np.float64)
    b = b[np.isfinite(b)]
    lo, hi = (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))) if len(b) else (float("nan"), float("nan"))
    sv = [float(x) for x in seed_vals]
    k = sum(1 for x in sv if sign * x > 0)
    return {"value": float(point), "ci95": [lo, hi], "seed_values": sv, "seeds_in_direction": k, "n_seeds": len(sv),
            "strict": bool(len(sv) >= 1 and k == len(sv) and ((lo > 0) if sign > 0 else (hi < 0)))}


def curve(vals, members, draws, seeds, weights=None):
    num_b = den_b = num_pt = den_pt = 0.0
    for s in members:
        Cm, pidx = draws[s]["C"], draws[s]["pidx"]
        P = Cm.shape[1]
        K = np.asarray(vals[seeds[0]][s]).shape[1]
        Sm = np.zeros((len(seeds), P, K))
        Wm = np.zeros((len(seeds), P))
        for i, sd in enumerate(seeds):
            x = np.asarray(vals[sd][s], np.float64)
            w = np.ones(len(x)) if weights is None else np.asarray(weights[sd][s], np.float64)
            np.add.at(Sm[i], pidx, w[:, None] * np.where(w[:, None] > 0, x, 0.0))
            Wm[i] = np.bincount(pidx, weights=w, minlength=P)
        num_b = num_b + np.einsum("bp,spk->bsk", Cm, Sm)
        with np.errstate(all="ignore"):
            den_b = den_b + Cm @ Wm.T
        num_pt = num_pt + Sm.sum(1)
        den_pt = den_pt + Wm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = num_pt / den_pt[:, None]
        boot = (num_b / den_b[:, :, None]).mean(1)
    return point, boot


def curve_ci(boot):
    out = np.full((2, boot.shape[1]), np.nan)
    for k in range(boot.shape[1]):
        b = boot[:, k][np.isfinite(boot[:, k])]
        if len(b):
            out[:, k] = np.quantile(b, [0.025, 0.975])
    return out


def paired_curve(va, vb, members, draws, seeds, vref=None):
    vr = vb if vref is None else vref
    vd = {sd: {s: np.asarray(va[sd][s], np.float64) - np.asarray(vb[sd][s], np.float64) for s in members} for sd in seeds}
    pd_, bd = curve(vd, members, draws, seeds)
    ref = np.array([np.concatenate([np.asarray(vr[sd][s], np.float64) for s in members]).mean(0) for sd in seeds]).mean(0)
    c = curve_ci(bd)
    return {"pct": 100 * pd_.mean(0) / ref, "pct_ci": 100 * c / ref, "ref_mean": ref, "per_seed_pct": 100 * pd_ / ref,
            "seeds": list(seeds)}
