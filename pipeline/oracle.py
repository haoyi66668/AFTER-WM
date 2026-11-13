from __future__ import annotations

import math

import numpy as np

from .registry import HORIZONS

COVERAGES = tuple(np.arange(1, 10, dtype=float) / 10)
PIT_EDGES = np.linspace(0.0, 1.0, 21)
PARITY_TOL = 2e-10
PIT_CLAMP = True


def logsumexp_scalar(values):
    x = np.asarray(values, dtype=np.float64)
    top = float(np.max(x))
    if not math.isfinite(top):
        raise ValueError("log density has no finite component")
    return top + math.log(sum(math.exp(float(v) - top) for v in x))


def normal_cdf_scalar(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def absolute_normal_moment_scalar(mean, sigma):
    if not sigma > 0:
        raise ValueError("positive standard deviation required")
    u = float(mean) / float(sigma)
    return (2.0 * sigma * math.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)
            + mean * (2.0 * normal_cdf_scalar(u) - 1.0))


def marginal_crps_pit_scalar(means, weights, truth, sigma):
    first = sum(float(w) * absolute_normal_moment_scalar(float(truth) - float(m), sigma)
                for m, w in zip(means, weights))
    second = 0.0
    for i in range(len(means)):
        for j in range(len(means)):
            second += float(weights[i] * weights[j]) * absolute_normal_moment_scalar(
                float(means[i] - means[j]), math.sqrt(2.0) * sigma)
    pit = sum(float(w) * normal_cdf_scalar((float(truth) - float(m)) / sigma)
              for m, w in zip(means, weights))
    return first - 0.5 * second, pit


def oracle_one(means, log_weights, targets, own_std, common_std, raw_scale,
               horizons=HORIZONS, sigma=0.5, density_tag="fixed"):

    mu = np.asarray(means, np.float64)
    y = np.asarray(targets, np.float64)
    lw = np.asarray(log_weights, np.float64)
    own = np.asarray(own_std, np.float64)
    common = np.asarray(common_std, np.float64)
    if mu.ndim != 3 or y.shape != (mu.shape[0], mu.shape[2]):
        raise ValueError("oracle tensor geometry")
    H, K, D = mu.shape
    if lw.shape != (K,) or own.shape != (D,) or common.shape != (D,):
        raise ValueError("oracle weights/scales geometry")
    if not (np.isfinite(mu).all() and np.isfinite(y).all()
            and np.isfinite(own).all() and np.isfinite(common).all()
            and (own > 0).all() and (common > 0).all()
            and np.isfinite(lw).any() and not np.isnan(lw).any()
            and not np.isposinf(lw).any() and raw_scale > 0 and sigma > 0):
        raise ValueError("nonfinite/invalid oracle input")
    lw = lw - logsumexp_scalar(lw)
    weights = np.exp(lw)
    point = np.einsum("k,hkd->hd", weights, mu)
    err = point - y
    squared_component = ((mu - y[:, None, :]) ** 2).sum(-1)
    cdf = np.empty((H, D), np.float64)
    crps_native = np.empty((H, D), np.float64)
    for h in range(H):
        for d in range(D):
            crps_native[h, d], cdf[h, d] = marginal_crps_pit_scalar(
                mu[h, :, d], weights, y[h, d], sigma)
    out = {}
    selections = [(f"prefix_H{h}", slice(0, h)) for h in horizons]
    selections += [(f"endpoint_H{h}", slice(h - 1, h)) for h in horizons]
    if H >= 100:
        selections += [("tail_H100_steps51_100", slice(50, 100))]
    gaussian_const = 0.5 * math.log(2.0 * math.pi * sigma ** 2)
    for name, selection in selections:
        ee = err[selection]
        length = len(ee)
        if not length:
            raise ValueError("horizon outside query")
        raw = ee * own
        standardized = raw / common
        if density_tag == "fixed":
            basic = {
                "MSE_native": float(np.mean(ee ** 2)),
                "MSE_raw": float(np.mean(raw ** 2)),
                "nMSE_train": float(np.mean(standardized ** 2)),
                "MAE_native": float(np.mean(abs(ee))),
                "MAE_raw": float(np.mean(abs(raw))),
                "nMAE_train": float(np.mean(abs(standardized))),
            }
            basic["J"] = 0.5 * (basic["nMSE_train"] + basic["MSE_raw"] / raw_scale)
            out.update({f"{key}__{name}": value for key, value in basic.items()})
        costs = squared_component[selection].sum(0) / (2.0 * sigma ** 2)
        nll_native = -logsumexp_scalar(lw - costs) / (length * D) + gaussian_const
        full = {
            "NLL_native": nll_native,
            "NLL_raw": nll_native + float(np.log(own).mean()),
            "NLL_common": nll_native + float(np.log(own / common).mean()),
            "CRPS_native": float(crps_native[selection].mean()),
            "CRPS_raw": float((crps_native[selection] * own).mean()),
            "CRPS_common": float((crps_native[selection] * own / common).mean()),
            "PIT_mean": float(cdf[selection].mean()),
            "PIT_second_moment": float((cdf[selection] ** 2).mean()),
        }
        for alpha in COVERAGES:
            lo = (1.0 - alpha) / 2.0
            full[f"coverage{int(round(alpha * 100)):02d}"] = float(
                ((cdf[selection] >= lo) & (cdf[selection] <= 1.0 - lo)).mean())
        out.update({f"{key}_{density_tag}__{name}": value for key, value in full.items()})
        pit_values = np.clip(cdf[selection], 0.0, 1.0) if PIT_CLAMP else cdf[selection]
        out[f"PIT_hist_{density_tag}__{name}"] = np.histogram(pit_values, PIT_EDGES)[0]
    return out


def calibration_nll_curve_one(means, log_weights, targets, own_std, common_std, sigma_grid):
    return np.asarray([oracle_one(means, log_weights, targets, own_std, common_std,
                                  float(np.mean(np.asarray(common_std) ** 2)),
                                  horizons=(len(targets),), sigma=float(s),
                                  density_tag="cal")[f"NLL_common_cal__prefix_H{len(targets)}"]
                       for s in sigma_grid], np.float64)


def check_parity(comp_mean, comp_logw, targets, scored, own, common, raw_scale, indices,
                 sigma_fixed=0.5, sigma_cal=None, tol=PARITY_TOL, horizons=HORIZONS):

    results = []
    for i in indices:
        i = int(i)
        expected = oracle_one(comp_mean[i], comp_logw[i], targets[i], own, common, raw_scale,
                              horizons=horizons, sigma=float(sigma_fixed))
        if sigma_cal is not None:
            expected.update(oracle_one(comp_mean[i], comp_logw[i], targets[i], own, common, raw_scale,
                                       horizons=horizons, sigma=float(sigma_cal), density_tag="cal"))
        maximum = 0.0
        for key, value in expected.items():
            if key not in scored:
                raise AssertionError(f"scorer is missing oracle key {key}")
            observed = np.asarray(scored[key])[i]
            if key.startswith("PIT_hist_"):
                if not np.array_equal(observed, value):
                    raise AssertionError(f"independent NumPy histogram mismatch {key}")
            elif not np.isclose(observed, value, rtol=tol, atol=tol):
                raise AssertionError(f"independent NumPy oracle mismatch {key}: {observed} != {value}")
            else:
                maximum = max(maximum, float(abs(observed - value)))
        results.append({"query_index": i, "scalar_metric_max_abs_error": maximum,
                        "scalar_checks": len(expected), "passed": True})
    return results
