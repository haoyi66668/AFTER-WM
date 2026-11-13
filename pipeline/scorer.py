from __future__ import annotations

import math

import numpy as np
import torch

from .oracle import PARITY_TOL, check_parity, oracle_one
from .registry import HORIZONS
from .registry import OBS_DIM as _OBS_DIM

COVERAGES = tuple(np.arange(1, 10, dtype=float) / 10)
PIT_EDGES = np.linspace(0, 1, 21)
SIGMA_GRID = np.exp(np.linspace(np.log(0.05), np.log(1.5), 61))
SIGMA_FIXED = 0.5
SIGMA_FIT_HORIZON = 5
SIGMA_FIT_SEED = 20261007
SIGMA_FIT_MAX_QUERIES = 8192


def validate_arrays(means, log_weights, targets, own_std, common_std, raw_scale):
    mu, lp, y = means.double(), log_weights.double(), targets.double()
    if mu.ndim != 4 or y.shape != (mu.shape[0], mu.shape[1], mu.shape[-1]):
        raise ValueError("means [N,H,K,D] and targets [N,H,D] required")
    if lp.shape != (mu.shape[0], mu.shape[2]):
        raise ValueError("one fixed expert probability per query required")
    own = torch.as_tensor(own_std, device=mu.device, dtype=torch.float64)
    common = torch.as_tensor(common_std, device=mu.device, dtype=torch.float64)
    if own.shape != (mu.shape[-1],) or common.shape != own.shape:
        raise ValueError("standard-deviation geometry")
    if not (torch.isfinite(own).all() and torch.isfinite(common).all()
            and (own > 0).all() and (common > 0).all()
            and math.isfinite(float(raw_scale)) and raw_scale > 0):
        raise ValueError("invalid physical TRAIN scales")
    finite = (torch.isfinite(mu).flatten(1).all(1) & torch.isfinite(y).flatten(1).all(1)
              & (~torch.isnan(lp)).all(1) & (~torch.isposinf(lp)).all(1)
              & torch.isfinite(lp).any(1))
    lp = torch.log_softmax(lp, -1)
    return mu, lp, y, own, common, finite


def _abs_normal_mean(m, s):
    u = m / s
    return s * math.sqrt(2 / math.pi) * torch.exp(-0.5 * u.square()) + m * torch.erf(u / math.sqrt(2))


def _marginals(mu, lp, y, sigma):
    weights = lp.exp()[:, None, :, None]
    cdf = (weights * torch.special.ndtr((y[:, :, None] - mu) / sigma)).sum(2)
    first = (weights * _abs_normal_mean(y[:, :, None] - mu, sigma)).sum(2)
    pair = mu[:, :, :, None] - mu[:, :, None, :]
    pair_weights = weights[:, :, :, None] * weights[:, :, None, :]
    second = (pair_weights * _abs_normal_mean(pair, sigma * math.sqrt(2))).sum((2, 3))
    return first - 0.5 * second, cdf


def score_queries(means, log_weights, targets, own_std, common_std, raw_scale,
                  horizons=HORIZONS, sigma_fixed=0.5, sigma_cal=None,
                  keep_by_horizon=True):

    mu, lp, y, own, common, finite = validate_arrays(
        means, log_weights, targets, own_std, common_std, raw_scale)
    N, H, K, D = mu.shape
    if max(horizons) > H or min(horizons) < 1:
        raise ValueError("requested horizon does not fit predictions")
    mean = (lp.exp()[:, None, :, None] * mu).sum(2)
    e = mean - y
    raw_e, norm_e = e * own, e * own / common
    basic_h = {
        "MSE_native": e.square().mean(-1), "MSE_raw": raw_e.square().mean(-1),
        "nMSE_train": norm_e.square().mean(-1), "MAE_native": e.abs().mean(-1),
        "MAE_raw": raw_e.abs().mean(-1), "nMAE_train": norm_e.abs().mean(-1),
    }
    basic_h["J"] = 0.5 * (basic_h["nMSE_train"] + basic_h["MSE_raw"] / raw_scale)
    selections = [(f"prefix_H{h}", slice(0, h)) for h in horizons]
    selections += [(f"endpoint_H{h}", slice(h - 1, h)) for h in horizons]
    if H >= 100:
        selections += [("tail_H100_steps51_100", slice(50, 100))]
    out = {"finite_query": finite}
    for name, sl in selections:
        for key, values in basic_h.items():
            out[f"{key}__{name}"] = values[:, sl].mean(1)
    if keep_by_horizon:
        out.update({f"{key}__by_horizon": value for key, value in basic_h.items()})
    squared = (mu - y[:, :, None]).square().sum(-1)
    sigmas = [("fixed", float(sigma_fixed))]
    if sigma_cal is not None:
        sigmas.append(("cal", float(sigma_cal)))
    for tag, sigma in sigmas:
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive finite sigma required")
        crps, pit = _marginals(mu, lp, y, sigma)
        for name, sl in selections:
            nsteps = mu[:, sl].shape[1]
            cost = squared[:, sl].sum(1) / (2 * sigma ** 2)
            native = -torch.logsumexp(lp - cost, -1) / (nsteps * D) + 0.5 * math.log(2 * math.pi * sigma ** 2)
            values = {
                "NLL_native": native,
                "NLL_raw": native + own.log().mean(),
                "NLL_common": native + (own / common).log().mean(),
                "CRPS_native": crps[:, sl].mean((1, 2)),
                "CRPS_raw": (crps[:, sl] * own).mean((1, 2)),
                "CRPS_common": (crps[:, sl] * own / common).mean((1, 2)),
                "PIT_mean": pit[:, sl].mean((1, 2)),
                "PIT_second_moment": pit[:, sl].square().mean((1, 2)),
            }
            for alpha in COVERAGES:
                lo = (1 - alpha) / 2
                values[f"coverage{int(round(alpha * 100)):02d}"] = ((pit[:, sl] >= lo) & (pit[:, sl] <= 1 - lo)).double().mean((1, 2))
            out.update({f"{key}_{tag}__{name}": value for key, value in values.items()})
            idx = (pit[:, sl].clamp(0, 1) * (len(PIT_EDGES) - 1)).long().clamp(0, len(PIT_EDGES) - 2)
            hist = torch.zeros((N, len(PIT_EDGES) - 1), device=mu.device, dtype=torch.int64)
            hist.scatter_add_(1, idx.reshape(N, -1), torch.ones_like(idx).reshape(N, -1))
            out[f"PIT_hist_{tag}__{name}"] = hist
    numeric_keys = [k for k, value in out.items() if value.is_floating_point()]
    if numeric_keys:
        metric_finite = torch.stack([torch.isfinite(out[k]).flatten(1).all(1)
                                    if out[k].ndim > 1 else torch.isfinite(out[k])
                                    for k in numeric_keys], 1).all(1)
        finite = finite & metric_finite
        out["finite_query"] = finite
    for key in numeric_keys:
        mask = finite if out[key].ndim == 1 else finite[:, None]
        out[key] = torch.where(mask, out[key], torch.full_like(out[key], float("nan")))
    return out


def calibration_curve(means, log_weights, targets, own_std, common_std, sigma_grid=SIGMA_GRID):
    mu, lp, y, own, common, finite = validate_arrays(
        means, log_weights, targets, own_std, common_std,
        float(np.mean(np.asarray(common_std, dtype=float) ** 2)))
    if not bool(finite.all()):
        raise FloatingPointError("nonfinite calibration prediction; no query dropped")
    cost = (mu - y[:, :, None]).square().sum((1, 3))
    H, D = mu.shape[1], mu.shape[-1]
    sig = torch.as_tensor(sigma_grid, device=mu.device, dtype=torch.float64)
    energy = lp[:, :, None] - cost[:, :, None] / (2 * sig[None, None].square())
    return (-torch.logsumexp(energy, 1) / (H * D)
            + 0.5 * torch.log(2 * math.pi * sig.square())[None]
            + (own / common).log().mean())


def summarize(arrays, parents):
    flags = np.asarray(arrays["finite_query"], bool)
    parent = np.asarray(parents).astype(str)
    if len(flags) != len(parent):
        raise ValueError("parent identity length")
    result = {"queries": len(flags), "parents": len(np.unique(parent)),
              "finite_queries": int(flags.sum()), "failed_queries": int((~flags).sum()),
              "ranking_eligible": bool(flags.all()), "metrics": {}, "calibration": {}}
    if not flags.all():
        result["failure_policy"] = "Retained in NPZ; entire split is not ranking eligible; no finite-only aggregate."
        return result
    ix = {p: np.flatnonzero(parent == p) for p in np.unique(parent)}
    for key, data in arrays.items():
        x = np.asarray(data)
        if x.ndim != 1 or key == "finite_query":
            continue
        pm = np.asarray([x[v].mean() for v in ix.values()])
        result["metrics"][key] = {
            "micro_mean": float(x.mean()), "parent_macro_mean": float(pm.mean()),
            "query_std": float(x.std()), "median": float(np.median(x)),
            "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)),
            "max": float(x.max()), "worst_parent_mean": float(pm.max()),
        }
    for key, hist in arrays.items():
        if not key.startswith("PIT_hist_"):
            continue
        tag, segment = key.removeprefix("PIT_hist_").split("__", 1)
        total = np.asarray(hist).sum(0, dtype=np.int64)
        freq = total / total.sum()
        observed = np.asarray([arrays[f"coverage{int(round(c * 100)):02d}_{tag}__{segment}"].mean()
                               for c in COVERAGES])
        parent_cov = np.asarray([[arrays[f"coverage{int(round(c * 100)):02d}_{tag}__{segment}"][v].mean()
                                  for c in COVERAGES] for v in ix.values()]).mean(0)
        result["calibration"][f"{tag}__{segment}"] = {
            "central_nominal": list(COVERAGES), "central_observed_micro": observed.tolist(),
            "central_observed_parent_macro": parent_cov.tolist(),
            "central_ECE_micro": float(np.mean(abs(observed - COVERAGES))),
            "central_ECE_parent_macro": float(np.mean(abs(parent_cov - COVERAGES))),
            "PIT_bin_edges": PIT_EDGES.tolist(), "PIT_counts": total.tolist(),
            "PIT_KS_at_bin_edges": float(np.max(abs(np.cumsum(freq) - PIT_EDGES[1:]))),
            "PIT_KS_resolution": .05,
        }
    return result


def frame_scales(stats):
    common = np.asarray(stats.obs_std, dtype=np.float64)
    if common.shape != (_OBS_DIM,) or not (np.isfinite(common).all() and (common > 0).all()):
        raise ValueError("invalid TRAIN obs std")
    own = common.copy()
    raw_scale = float(np.mean(common ** 2))
    return own, common, raw_scale


def as_components(mean, comp_mean=None, comp_logw=None):
    mean = np.asarray(mean)
    if comp_mean is None:
        return mean[:, :, None, :], np.zeros((mean.shape[0], 1), dtype=np.float32)
    comp_mean = np.asarray(comp_mean)
    if comp_logw is None:
        comp_logw = np.zeros(comp_mean.shape[:1] + comp_mean.shape[2:3], dtype=np.float32)
    return comp_mean, np.asarray(comp_logw)


def mixture_mean_np(comp_mean, comp_logw):
    mu = torch.from_numpy(np.asarray(comp_mean)).double()
    lp = torch.log_softmax(torch.from_numpy(np.asarray(comp_logw)).double(), -1)
    return (lp.exp()[:, None, :, None] * mu).sum(2).numpy()


def score_split(comp_mean, comp_logw, targets, own, common, raw_scale, sigma_fixed=SIGMA_FIXED,
                sigma_cal=None, horizons=HORIZONS, chunk=None, device="cpu", keep_by_horizon=False):
    comp_mean = np.asarray(comp_mean)
    comp_logw = np.asarray(comp_logw)
    targets = np.asarray(targets)
    n, _, k, _ = comp_mean.shape
    if chunk is None:
        chunk = max(1, min(n, int(2_000_000 // max(1, k * k * comp_mean.shape[1] * comp_mean.shape[3]))))
    parts = {}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        out = score_queries(torch.from_numpy(comp_mean[s:e]).to(device),
                            torch.from_numpy(comp_logw[s:e]).to(device),
                            torch.from_numpy(targets[s:e]).to(device),
                            own, common, raw_scale, horizons=horizons, sigma_fixed=sigma_fixed,
                            sigma_cal=sigma_cal, keep_by_horizon=keep_by_horizon)
        for key, val in out.items():
            parts.setdefault(key, []).append(val.detach().cpu().numpy())
    return {key: np.concatenate(v, 0) for key, v in parts.items()}


def point_scores(comp_mean, comp_logw, targets, own, common, raw_scale, horizons=HORIZONS):
    mu, lp, y, own_t, common_t, finite = validate_arrays(
        torch.from_numpy(np.asarray(comp_mean)), torch.from_numpy(np.asarray(comp_logw)),
        torch.from_numpy(np.asarray(targets)), own, common, raw_scale)
    mean = (lp.exp()[:, None, :, None] * mu).sum(2)
    e = mean - y
    raw_e, norm_e = e * own_t, e * own_t / common_t
    basic_h = {
        "MSE_native": e.square().mean(-1), "MSE_raw": raw_e.square().mean(-1),
        "nMSE_train": norm_e.square().mean(-1), "MAE_native": e.abs().mean(-1),
        "MAE_raw": raw_e.abs().mean(-1), "nMAE_train": norm_e.abs().mean(-1),
    }
    basic_h["J"] = 0.5 * (basic_h["nMSE_train"] + basic_h["MSE_raw"] / raw_scale)
    selections = [(f"prefix_H{h}", slice(0, h)) for h in horizons]
    selections += [(f"endpoint_H{h}", slice(h - 1, h)) for h in horizons]
    out = {}
    for name, sl in selections:
        for key, values in basic_h.items():
            out[f"{key}__{name}"] = values[:, sl].mean(1)
    metric_finite = torch.stack([torch.isfinite(v) for v in out.values()], 1).all(1)
    finite = finite & metric_finite
    res = {"finite_query": finite.numpy()}
    for key, v in out.items():
        res[key] = torch.where(finite, v, torch.full_like(v, float("nan"))).numpy()
    return res


def selection_metric(comp_mean, comp_logw, targets, own, common, raw_scale, horizons=HORIZONS):
    ps = point_scores(comp_mean, comp_logw, targets, own, common, raw_scale, horizons)
    finite = bool(np.asarray(ps["finite_query"]).all())
    per_h = {f"nMSE_train__prefix_H{h}": (float(np.asarray(ps[f"nMSE_train__prefix_H{h}"]).mean())
                                          if finite else float("inf")) for h in horizons}
    per_h.update({f"J__prefix_H{h}": (float(np.asarray(ps[f"J__prefix_H{h}"]).mean()) if finite
                                      else float("inf")) for h in horizons})
    value = float(np.mean([per_h[f"nMSE_train__prefix_H{h}"] for h in horizons])) if finite else float("inf")
    return value, per_h, finite


def fit_sigma_on_val(comp_mean, comp_logw, targets, own, common, fit_horizon=SIGMA_FIT_HORIZON,
                     max_queries=SIGMA_FIT_MAX_QUERIES, sampling_seed=SIGMA_FIT_SEED, device="cpu"):
    n = len(targets)
    selected = np.random.default_rng(sampling_seed).choice(n, min(max_queries, n), replace=False)
    selected.sort()
    cm = np.asarray(comp_mean)[selected, :fit_horizon]
    lw = np.asarray(comp_logw)[selected]
    yy = np.asarray(targets)[selected, :fit_horizon]
    curve = calibration_curve(torch.from_numpy(cm).to(device), torch.from_numpy(lw).to(device),
                              torch.from_numpy(yy).to(device), own, common).cpu().numpy()
    average = curve.sum(0) / len(selected)
    choice = int(average.argmin())
    return {"fit_split": "val", "fit_horizon": fit_horizon, "queries": int(len(selected)),
            "query_sampling_seed": sampling_seed, "sigma_grid": SIGMA_GRID.tolist(),
            "mean_NLL_common": average.tolist(), "sigma_cal": float(SIGMA_GRID[choice]),
            "grid_choice": choice, "boundary_choice": choice in (0, len(SIGMA_GRID) - 1),
            "main_sigma_unchanged": SIGMA_FIXED, "frame": "TRAIN-normalized (own=common=TRAIN std)"}


def weighted_means(arrays, weights_by_h, horizons=HORIZONS):
    out = {}
    for key, x in arrays.items():
        x = np.asarray(x)
        if x.ndim != 1 or key == "finite_query" or not np.issubdtype(x.dtype, np.floating):
            continue
        h = horizon_of_key(key)
        if h is None or h not in weights_by_h:
            continue
        w = np.asarray(weights_by_h[h], dtype=np.float64)
        out[key] = float(np.sum(w * x) / np.sum(w))
    return out


def horizon_of_key(key: str):
    if "__prefix_H" in key:
        return int(key.rsplit("__prefix_H", 1)[1])
    if "__endpoint_H" in key:
        return int(key.rsplit("__endpoint_H", 1)[1])
    return None


def per_start_means(arrays, t, keys):
    t = np.asarray(t)
    out = {}
    for key in keys:
        if key not in arrays:
            continue
        x = np.asarray(arrays[key])
        out[key] = {int(s): float(x[t == s].mean()) for s in np.unique(t)}
    return out


def parity_selftest(seed=0, n=6, horizon=40, ks=(1, 3), dims=17, tol=PARITY_TOL, sigma_cal=0.3123,
                    distinct_frames=True):

    rng = np.random.default_rng(seed)
    report = {"cases": []}
    for k in ks:
        for frame in ((True, False) if distinct_frames else (False,)):
            comp_mean = rng.normal(size=(n, horizon, k, dims))
            targets = comp_mean.mean(2) + rng.normal(scale=0.7, size=(n, horizon, dims))
            comp_logw = rng.normal(size=(n, k))
            if k > 1:
                comp_logw[0, 0] = -np.inf
            common = rng.uniform(0.2, 3.5, size=dims)
            own = rng.uniform(0.05, 2.0, size=dims) if frame else common.copy()
            raw_scale = float(np.mean(common ** 2))
            scored = score_split(comp_mean, comp_logw, targets, own, common, raw_scale,
                                 sigma_cal=sigma_cal, keep_by_horizon=False, chunk=2)
            checks = check_parity(comp_mean, comp_logw, targets, scored, own, common, raw_scale,
                                  indices=range(n), sigma_cal=sigma_cal, tol=tol)
            case = {"K": k, "own_ne_common": bool(frame), "queries": n,
                    "max_abs_error": max(c["scalar_metric_max_abs_error"] for c in checks),
                    "checks": sum(c["scalar_checks"] for c in checks)}
            if k == 1 and not frame:
                nll = scored["NLL_native_fixed__prefix_H40"]
                ident = 2 * scored["MSE_native__prefix_H40"] + 0.5 * math.log(2 * math.pi * 0.25)
                assert np.allclose(nll, ident, rtol=1e-12, atol=1e-12), "K=1 NLL identity"
                z = (targets - comp_mean[:, :, 0]) / SIGMA_FIXED
                from math import erf, exp, pi, sqrt
                phi = np.vectorize(lambda v: exp(-0.5 * v * v) / sqrt(2 * pi))(z)
                Phi = np.vectorize(lambda v: 0.5 * (1 + erf(v / sqrt(2))))(z)
                crps = SIGMA_FIXED * (z * (2 * Phi - 1) + 2 * phi - 1 / sqrt(pi))
                assert np.allclose(scored["CRPS_native_fixed__prefix_H40"], crps.mean((1, 2)),
                                   rtol=1e-10, atol=1e-12), "K=1 CRPS closed form"
                assert np.allclose(scored["MSE_native__prefix_H40"], scored["nMSE_train__prefix_H40"],
                                   rtol=1e-12, atol=1e-15), "own==common frame identity"
                case["analytic_identities"] = "passed"
            report["cases"].append(case)
    cm = rng.normal(size=(3, 5, 2, dims))
    cm[1, 2, 0, 3] = np.nan
    y = rng.normal(size=(3, 5, dims))
    lw = np.zeros((3, 2))
    sc = score_split(cm, lw, y, np.ones(dims), np.ones(dims), 1.0, horizons=(1, 5))
    assert sc["finite_query"].tolist() == [True, False, True], "NaN policy"
    assert np.isnan(sc["J__prefix_H5"][1]) and np.isfinite(sc["J__prefix_H5"][[0, 2]]).all()
    report["nan_policy"] = "passed"
    report["passed"] = True
    return report
