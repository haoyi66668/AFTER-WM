from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import contextlib
import hashlib
import json
import random
import sys
import time
import traceback

import numpy as np
import torch

from . import data as D
from . import registry as P
from . import run_io as R
from . import scorer as S
from .eval_api import QueryBatch
from .oracle import check_parity
from afterwm import legacy as LEGACY

if __name__ == "__main__":
    sys.modules.setdefault("pipeline.evaluate", sys.modules["__main__"])

SCHEMA = "afterwm_eval_v1"
SPLITS = ("val", "test", "test_b")
SPLIT_TAG = {"train": "TRAIN", "val": "VAL", "test": "TEST", "test_b": "TEST_B"}
EXTRA_SPLITS = tuple(s for s in getattr(P, "ACTIVE_EXTRA_SPLITS", {}) if s not in SPLITS)
SPLIT_TAG.update({s: P.ACTIVE_EXTRA_SPLITS[s]["tag"] for s in EXTRA_SPLITS})
ALL_SPLITS = SPLITS + EXTRA_SPLITS
MIX_ATOL, MIX_RTOL = 1e-4, 1e-5
KEY_METRICS = tuple([f"J__prefix_H{h}" for h in P.HORIZONS] + [f"nMSE_train__prefix_H{h}" for h in P.HORIZONS]
                    + [f"J__endpoint_H{h}" for h in P.HORIZONS] + [f"nMSE_train__endpoint_H{h}" for h in P.HORIZONS])


class AlreadyComplete(RuntimeError):
    pass


_STATS = {}


def get_stats(data_dir: str = P.DATA_DIR):
    if data_dir not in _STATS:
        st = D.train_stats(data_dir)
        prep = os.path.join(P.PREP_DIR, "TRAIN_STATS.json")
        if data_dir == P.DATA_DIR and os.path.exists(prep):
            pst = D.TrainStats.load(prep)
            if pst.sha256() != st.sha256():
                raise RuntimeError(f"prepared TRAIN_STATS.json sha {pst.sha256()} != recomputed {st.sha256()}")
        _STATS[data_dir] = st
    return _STATS[data_dir]


_SPLIT_CACHE = {}


def split_queries(split: str, stats=None, data_dir: str = P.DATA_DIR) -> dict:
    stats = stats if stats is not None else get_stats(data_dir)
    ck = (split, stats.sha256(), data_dir)
    if ck not in _SPLIT_CACHE:
        obs, act = D.load_visible(split, data_dir)
        obs_n, act_n = stats.norm_obs(obs), stats.norm_act(act)
        e, t = D.eval_queries(obs.shape[0])
        w = D.windows(obs_n, act_n, e, t, P.MAX_H)
        c = {k: np.ascontiguousarray(w[k]) for k in ("ctx_obs", "ctx_act", "fut_act", "fut_obs")}
        c.update({"split": split, "e": e, "t": t, "keys": D.context_keys(c["ctx_obs"], c["ctx_act"]),
                  "obs_n": obs_n, "act_n": act_n, "n_episodes": int(obs.shape[0]),
                  "data_sha256": R.sha256_file(D.split_path(split, data_dir)), "stats_sha256": stats.sha256()})
        _SPLIT_CACHE[ck] = c
    return _SPLIT_CACHE[ck]


_MANIFEST_CACHE = {}


def query_manifest(split: str, data_dir: str = P.DATA_DIR) -> dict:

    ck = (split, data_dir)
    if ck in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[ck]
    obs, act = D.load_visible(split, data_dir)
    labels = D.load_analysis_labels(split, data_dir)
    e, t = D.eval_queries(obs.shape[0])
    tag = SPLIT_TAG[split]
    out = {"e": e, "t": t,
           "query_id": np.asarray([f"{tag}:{int(a)}:{int(b)}" for a, b in zip(e, t)]),
           "parent": np.asarray([f"{tag}:{p}" for p in labels["parent"][e]]) if "parent" in labels
           else np.asarray([f"{tag}:unknown"] * len(e))}
    for k in ("cohort", "sequence"):
        if k in labels:
            out[k] = np.asarray(labels[k][e]).astype(str)
    for h in P.HORIZONS:
        kin, kfull = [], []
        for ei, ti in zip(e, t):
            hi = hashlib.sha256(np.ascontiguousarray(obs[ei, ti - P.CONTEXT: ti + 1]).tobytes()
                                + np.ascontiguousarray(act[ei, ti - P.CONTEXT: ti + h]).tobytes())
            hf = hashlib.sha256(hi.digest() + np.ascontiguousarray(obs[ei, ti + 1: ti + h + 1]).tobytes())
            kin.append(hi.hexdigest()[:16])
            kfull.append(hf.hexdigest()[:16])
        for name, keys in (("in", kin), ("full", kfull)):
            first, gid = {}, np.empty(len(keys), np.int64)
            for i, k in enumerate(keys):
                gid[i] = first.setdefault(k, i)
            mult = np.bincount(gid, minlength=len(keys))[gid]
            out[f"key_{name}_H{h}"] = np.asarray(keys)
            out[f"gid_{name}_H{h}"] = gid
            out[f"mult_{name}_H{h}"] = mult.astype(np.int64)
    out["manifest_sha256"] = np.asarray(manifest_sha(out))
    _MANIFEST_CACHE[ck] = out
    return out


def manifest_sha(m: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(m):
        if k == "manifest_sha256":
            continue
        h.update(k.encode())
        h.update(R.array_sha256(m[k]).encode())
    return h.hexdigest()


def query_manifest_checked(split: str, data_dir: str = P.DATA_DIR) -> dict:
    m = query_manifest(split, data_dir)
    prep = P.PREP_DIR if split not in EXTRA_SPLITS else (getattr(P, "EXTRA_PREP_DIR", None) or P.PREP_DIR)
    path = os.path.join(prep, f"QUERY_MANIFEST_{split}.npz")
    if data_dir == P.DATA_DIR and os.path.exists(path):
        disk = R.load_npz(path)
        for k, v in m.items():
            if k not in disk or not np.array_equal(np.asarray(disk[k]), np.asarray(v)):
                raise RuntimeError(f"prepared {path} differs from recomputed manifest at key {k}")
    return m


def label_guard():
    @contextlib.contextmanager
    def _guard():
        orig = D.load_analysis_labels

        def _blocked(*a, **k):
            raise RuntimeError("hidden labels are not available to models/trainers (label guard)")
        D.load_analysis_labels = _blocked
        try:
            yield
        finally:
            D.load_analysis_labels = orig
    return _guard()


def _np(x):
    if x is None:
        return None
    x = x.detach()
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return x.cpu().numpy()


def _is_oom(exc: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def forecast_chunk(fm, ctx_obs, ctx_act, fut_act, keys, device) -> dict:
    qb = QueryBatch(ctx_obs=torch.from_numpy(np.ascontiguousarray(ctx_obs)).to(device),
                    ctx_act=torch.from_numpy(np.ascontiguousarray(ctx_act)).to(device),
                    fut_act=torch.from_numpy(np.ascontiguousarray(fut_act)).to(device),
                    key=torch.from_numpy(np.ascontiguousarray(keys).astype(np.int64)).to(device))
    grad_ctx = contextlib.nullcontext() if getattr(fm, "needs_grad_at_eval", False) else torch.no_grad()
    with grad_ctx, label_guard():
        f = fm.forecast(qb)
    b, h = fut_act.shape[0], fut_act.shape[1]
    out = {"mean": _np(f.mean), "comp_mean": _np(getattr(f, "comp_mean", None)),
           "comp_logw": _np(getattr(f, "comp_logw", None)), "comp_var": _np(getattr(f, "comp_var", None))}
    if out["mean"] is None or out["mean"].shape != (b, h, P.OBS_DIM):
        raise ValueError(f"Forecast.mean must be [B,H,17]=({b},{h},17), got "
                         f"{None if out['mean'] is None else out['mean'].shape}")
    if out["comp_mean"] is not None:
        cm = out["comp_mean"]
        if cm.ndim != 4 or cm.shape[:2] != (b, h) or cm.shape[3] != P.OBS_DIM:
            raise ValueError(f"comp_mean must be [B,H,K,17], got {cm.shape}")
        if out["comp_logw"] is not None and out["comp_logw"].shape != (b, cm.shape[2]):
            raise ValueError(f"comp_logw must be [B,K], got {out['comp_logw'].shape}")
        if out["comp_var"] is not None and out["comp_var"].shape != cm.shape:
            raise ValueError(f"comp_var must match comp_mean {cm.shape}, got {out['comp_var'].shape}")
    elif out["comp_logw"] is not None:
        raise ValueError("comp_logw given without comp_mean")
    elif out["comp_var"] is not None and out["comp_var"].shape not in ((b, h, 1, P.OBS_DIM),):
        raise ValueError("comp_var without comp_mean must be [B,H,1,17]")
    return out


def forecast_arrays(fm, sq: dict, device, chunk: int = 720, fut_act=None, order=None,
                    progress: str | None = None) -> dict:
    ctx_obs, ctx_act, keys = sq["ctx_obs"], sq["ctx_act"], sq["keys"]
    fa = sq["fut_act"] if fut_act is None else fut_act
    if order is not None:
        ctx_obs, ctx_act, keys, fa = ctx_obs[order], ctx_act[order], keys[order], fa[order]
    n = len(keys)
    parts, s, cur = [], 0, max(1, int(chunk))
    while s < n:
        e = min(n, s + cur)
        try:
            parts.append(forecast_chunk(fm, ctx_obs[s:e], ctx_act[s:e], fa[s:e], keys[s:e], device))
        except Exception as exc:
            if _is_oom(exc) and cur > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cur = max(1, cur // 2)
                print(json.dumps({"stage": "forecast_oom_fallback", "chunk": cur}), flush=True)
                continue
            raise
        s = e
        if progress:
            print(json.dumps({"stage": progress, "queries_done": s, "queries_total": n}), flush=True)
    out = {}
    for k in ("mean", "comp_mean", "comp_logw", "comp_var"):
        vals = [p[k] for p in parts]
        if all(v is None for v in vals):
            out[k] = None
        elif any(v is None for v in vals):
            raise ValueError(f"Forecast field {k} present in some chunks only")
        else:
            out[k] = np.concatenate(vals, 0)
    out["chunk_used"] = cur
    return out


def components(out: dict):
    return S.as_components(out["mean"], out.get("comp_mean"), out.get("comp_logw"))


def mixture_consistency(out: dict) -> dict:
    if out.get("comp_mean") is None:
        return {"has_components": False, "max_abs_dev": 0.0, "consistent": True, "K": 1}
    cm, lw = components(out)
    mm = S.mixture_mean_np(cm, lw)
    mean = np.asarray(out["mean"], np.float64)
    with np.errstate(invalid="ignore"):
        dev = np.abs(mean - mm)
        ok = bool(np.all((dev <= MIX_ATOL + MIX_RTOL * np.abs(mean)) | (np.isnan(mean) & np.isnan(mm))))
    md = float(np.nanmax(dev)) if np.isfinite(dev).any() else float("nan")
    return {"has_components": True, "max_abs_dev": md, "consistent": ok, "K": int(cm.shape[2]),
            "tolerance": f"{MIX_ATOL} + {MIX_RTOL}*|mean|"}


def state_hash(model) -> dict:
    hh = hashlib.sha256()
    rows = sorted(model.state_dict().items())
    for name, value in rows:
        x = value.detach().cpu().contiguous()
        hh.update(name.encode())
        hh.update(str(tuple(x.shape)).encode())
        hh.update(str(x.dtype).encode())
        hh.update(x.numpy().tobytes() if x.dtype != torch.bfloat16 else x.float().numpy().tobytes())
    return {"sha256": hh.hexdigest(), "tensor_count": len(rows)}


def save_rng() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def quick_val(module, model, cfg, device, return_details: bool = False):
    t0 = time.time()
    stats = get_stats()
    sq = split_queries("val", stats)
    own, common, raw_scale = S.frame_scales(stats)
    rng = save_rng()
    was_training = model.training
    model.eval()
    try:
        fm = module.forecaster(model, cfg)
        out = forecast_arrays(fm, sq, device)
        cm, lw = components(out)
        value, per_h, finite = S.selection_metric(cm, lw, sq["fut_obs"], own, common, raw_scale)
        mix = mixture_consistency(out)
    finally:
        model.train(was_training)
        restore_rng(rng)
    if not return_details:
        return value
    return value, {"per_h": per_h, "finite": finite, "mixture": mix, "seconds": time.time() - t0,
                   "chunk_used": out["chunk_used"], "metric": P.VAL_SELECTION_METRIC}


def resolve_module(name: str):
    from .trainer import resolve_module as _rm
    return _rm(name)


def load_run(run_dir: str, module_name: str | None = None, device: str = "cuda", ckpt_name: str = "best.pt"):
    prot = R.read_json(os.path.join(run_dir, "PROTOCOL.json"))
    if prot is None:
        raise FileNotFoundError(f"{run_dir}/PROTOCOL.json missing")
    prot = LEGACY.upgrade_protocol(prot)
    mod_name = module_name or prot["module"]
    if module_name and module_name != prot["module"]:
        raise ValueError(f"--module {module_name} != PROTOCOL module {prot['module']}")
    mod = resolve_module(mod_name)
    cfg = prot["cfg"]
    stats = get_stats()
    if stats.sha256() != prot["stats_sha256"]:
        raise RuntimeError("TRAIN stats sha differs from the run's PROTOCOL.json")
    ck_path = os.path.join(run_dir, ckpt_name)
    with label_guard():
        model = mod.build(cfg, int(prot["seed"]), device, stats)
        ck = R.torch_load(ck_path, map_location=device)
        if hasattr(mod, "load_checkpoint"):
            mod.load_checkpoint(model, ck, cfg)
        else:
            model.load_state_dict(ck["model"])
    model.eval()
    return mod, model, cfg, prot, stats, ck


def _parity_indices(finite: np.ndarray) -> list:
    idx = np.flatnonzero(finite)
    if len(idx) == 0:
        return []
    return sorted({int(idx[0]), int(idx[len(idx) // 2]), int(idx[-1])})


def score_forecasts(split: str, out: dict, sq: dict, man: dict, stats, sigma_cal=None) -> tuple:
    own, common, raw_scale = S.frame_scales(stats)
    cm, lw = components(out)
    targets = sq["fut_obs"]
    arrays = S.score_split(cm, lw, targets, own, common, raw_scale, sigma_fixed=S.SIGMA_FIXED,
                           sigma_cal=sigma_cal, keep_by_horizon=False)
    finite = np.asarray(arrays["finite_query"], bool)
    parity, parity_error = [], None
    try:
        parity = check_parity(cm, lw, targets, arrays, own, common, raw_scale, _parity_indices(finite),
                              sigma_fixed=S.SIGMA_FIXED, sigma_cal=sigma_cal)
    except AssertionError as exc:
        parity_error = str(exc)
    result = S.summarize(arrays, man["parent"])
    mix = mixture_consistency(out)
    sel, per_h, _ = S.selection_metric(cm, lw, targets, own, common, raw_scale)
    finite_ok = bool(finite.all())
    result.update({
        "schema": SCHEMA, "split": split, "frame": "TRAIN-normalized; own_std = common_std = TRAIN obs std (301 steps, ddof 0)",
        "own_std": own.tolist(), "common_std": common.tolist(), "raw_scale": raw_scale,
        "sigma_fixed": S.SIGMA_FIXED, "sigma_cal": sigma_cal, "K": int(cm.shape[2]),
        "has_comp_var": out.get("comp_var") is not None,
        "numpy_oracle_checks": parity, "numpy_oracle_error": parity_error,
        "mixture_consistency": mix,
        "selection_metric": {"name": P.VAL_SELECTION_METRIC, "value": sel, "per_h": per_h},
        "data_sha256": sq["data_sha256"], "stats_sha256": sq["stats_sha256"],
        "query_manifest_sha256": str(man["manifest_sha256"]),
        "query_ids_sha256": R.sha256_bytes("\n".join(man["query_id"].tolist()).encode()),
    })
    if finite_ok:
        weights = {h: 1.0 / man[f"mult_full_H{h}"] for h in P.HORIZONS}
        result["dedup_full_mean"] = S.weighted_means(arrays, weights)
        weights_in = {h: 1.0 / man[f"mult_in_H{h}"] for h in P.HORIZONS}
        result["dedup_in_mean"] = S.weighted_means({k: arrays[k] for k in KEY_METRICS if k in arrays}, weights_in)
        result["per_start_mean"] = S.per_start_means(arrays, man["t"], KEY_METRICS)
    reasons = []
    if not finite_ok:
        reasons.append(f"{int((~finite).sum())} nonfinite queries")
    if parity_error:
        reasons.append("numpy oracle parity failed")
    if not mix["consistent"]:
        reasons.append("Forecast.mean != mixture mean of components")
    result["ranking_eligible"] = not reasons
    result["not_eligible_reasons"] = reasons
    return arrays, result


def write_split(eval_dir: str, split: str, out: dict, arrays: dict, result: dict, sq: dict, man: dict):
    d = os.path.join(eval_dir, split)
    os.makedirs(d, exist_ok=True)
    pred = {"e": sq["e"], "t": sq["t"], "query_id": man["query_id"], "key": sq["keys"],
            "mean": out["mean"], "finite_query": np.asarray(arrays["finite_query"], bool)}
    for k in ("comp_mean", "comp_logw", "comp_var"):
        if out.get(k) is not None:
            pred[k] = out[k]
    R.atomic_npz(os.path.join(d, "PREDICTIONS.npz"), compressed=True, **pred)
    pq = {k: v for k, v in arrays.items()}
    pq.update({"e": sq["e"], "t": sq["t"], "parent": man["parent"], "query_id": man["query_id"]})
    R.atomic_npz(os.path.join(d, "PER_QUERY.npz"), compressed=True, **pq)
    result["files"] = {"PREDICTIONS.npz": R.sha256_file(os.path.join(d, "PREDICTIONS.npz")),
                       "PER_QUERY.npz": R.sha256_file(os.path.join(d, "PER_QUERY.npz"))}
    R.atomic_json(os.path.join(d, "RESULT.json"), result)


def _afterwm_identity(model, module_name: str) -> dict | None:
    if not str(module_name).startswith("afterwm."):
        return None
    ident = getattr(model, "parent_identity", None)
    if ident is None:
        ident = getattr(model, "identity", None)
    gamma = getattr(model, "gamma", None)
    out = {"gamma": None if gamma is None else float(gamma)}
    if isinstance(ident, dict):
        keep = ("variant", "sha256", "pinned", "stage1a_state_sha256", "state_sha256", "gamma", "router_gamma",
                "gamma_source", "base_dir", "path", "files", "c_e")
        out["parent_identity"] = R.to_jsonable({k: ident[k] for k in keep if k in ident})
    return out


def _ckpt_check(run_dir: str, ckpt_name: str, train_complete) -> dict:
    got = R.sha256_file(os.path.join(run_dir, ckpt_name))
    want = (train_complete or {}).get("best_pt_sha256") if ckpt_name == "best.pt" else None
    return {"checkpoint": ckpt_name, "sha256": got, "train_complete_best_pt_sha256": want,
            "matches": (None if want is None else got == want)}


def evaluate_run(run_dir: str, module_name: str | None = None, splits=SPLITS, device: str = "cuda",
                 chunk: int = 720, ckpt_name: str = "best.pt", allow_incomplete_train: bool = False,
                 eval_subdir: str = "eval", out_dir: str | None = None, sigma_cal_from: str | None = None,
                 verify_ckpt_sha: bool = False, provenance: dict | None = None) -> dict:
    run_dir = os.path.abspath(run_dir)
    eval_dir = os.path.abspath(out_dir) if out_dir else os.path.join(run_dir, eval_subdir)
    os.makedirs(eval_dir, exist_ok=True)
    with R.FileLock(os.path.join(eval_dir, ".lock")):
        if os.path.exists(os.path.join(eval_dir, "COMPLETE.json")):
            raise AlreadyComplete(f"{eval_dir}/COMPLETE.json exists; refusing to overwrite")
        train_complete = R.read_json(os.path.join(run_dir, "COMPLETE.json"))
        if train_complete is None and not allow_incomplete_train:
            raise RuntimeError(f"{run_dir}/COMPLETE.json missing (training not complete)")
        for name in list(ALL_SPLITS) + ["VAL_SIGMA_CALIBRATION.json", "EVALUATION_PROTOCOL.json", "FAILED.json"]:
            moved = R.move_aside(os.path.join(eval_dir, name), "prev")
            if moved:
                print(json.dumps({"stage": "moved_aside", "from": name, "to": moved}), flush=True)
        t0 = time.time()
        try:
            ckc = _ckpt_check(run_dir, ckpt_name, train_complete)
            if verify_ckpt_sha and ckc["matches"] is not True:
                raise RuntimeError(f"checkpoint sha256 check failed: {ckc}")
            arch_sigma = None
            if sigma_cal_from:
                arch = R.read_json(sigma_cal_from)
                if not arch or arch.get("sigma_cal") is None:
                    raise RuntimeError(f"--sigma-cal-from {sigma_cal_from}: no archived sigma_cal")
                arch_sigma = float(arch["sigma_cal"])
            mod, model, cfg, prot, stats, ck = load_run(run_dir, module_name, device, ckpt_name)
            fm = mod.forecaster(model, cfg)
            before = state_hash(model)
            eprot = {"schema": SCHEMA, "run_dir": run_dir, "run_dir_realpath": os.path.realpath(run_dir),
                     "determinism": dict(_DETERMINISM), "module": prot["module"], "model_id": prot.get("model_id"),
                     "seed": prot.get("seed"), "budget_class": prot.get("budget_class"),
                     "checkpoint": ckpt_name, "checkpoint_sha256": R.sha256_file(os.path.join(run_dir, ckpt_name)),
                     "checkpoint_step": ck.get("step"), "checkpoint_val": ck.get("val"),
                     "train_complete": train_complete is not None, "splits": list(splits),
                     "horizons": list(P.HORIZONS), "context": P.CONTEXT, "eval_starts": list(P.EVAL_STARTS),
                     "sigma_fixed": S.SIGMA_FIXED, "sigma_cal_fit": "VAL, H5 prefix, 61-point grid",
                     "stats_sha256": stats.sha256(), "model_state_before": before,
                     "parameter_report": fm.parameter_report(),
                     "forecaster": {"name": getattr(fm, "name", None), "family": getattr(fm, "family", None),
                                    "budget_class": getattr(fm, "budget_class", None),
                                    "needs_grad_at_eval": bool(getattr(fm, "needs_grad_at_eval", False)),
                                    "stochastic": bool(getattr(fm, "stochastic", False))},
                     "code_files": R.code_manifest(), "env": R.env_info(device)}
            from . import oracle as _O
            eprot["run"] = {"out_dir": eval_dir, "eval_dir_is_run_subdir": not bool(out_dir),
                           "checkpoint_check": ckc, "verify_ckpt_sha": bool(verify_ckpt_sha),
                           "sigma_cal_from": sigma_cal_from, "archived_sigma_cal": arch_sigma,
                           "oracle_pit_clamp": bool(getattr(_O, "PIT_CLAMP", False)),
                           "afterwm": _afterwm_identity(model, prot["module"]), "provenance": dict(provenance or {}),
                           "extra_splits_registry": {s: dict(P.ACTIVE_EXTRA_SPLITS[s]) for s in EXTRA_SPLITS},
                           "code_root": R.CODE_ROOT}
            R.atomic_json(os.path.join(eval_dir, "EVALUATION_PROTOCOL.json"), eprot)
            forecasts, timings = {}, {}
            need = ["val"] + [s for s in splits if s != "val"]
            for split in need:
                sq = split_queries(split, stats)
                ts = time.time()
                forecasts[split] = forecast_arrays(fm, sq, device, chunk, progress=f"forecast_{split}")
                timings[split] = time.time() - ts
            sq_val = split_queries("val", stats)
            cm, lw = components(forecasts["val"])
            own, common, _ = S.frame_scales(stats)
            try:
                fit = S.fit_sigma_on_val(cm, lw, sq_val["fut_obs"], own, common)
                sigma_cal = fit["sigma_cal"]
            except FloatingPointError as exc:
                fit, sigma_cal = {"error": str(exc), "sigma_cal": None}, None
            fit.update({"val_data_sha256": sq_val["data_sha256"], "stats_sha256": stats.sha256()})
            if arch_sigma is not None:
                fit.update({"refit_sigma_cal": sigma_cal, "archived_sigma_cal": arch_sigma,
                            "archived_from": sigma_cal_from, "refit_equals_archived": sigma_cal == arch_sigma,
                            "sigma_cal_source": "archived"})
                sigma_cal = arch_sigma
                fit["sigma_cal"] = arch_sigma
            R.atomic_json(os.path.join(eval_dir, "VAL_SIGMA_CALIBRATION.json"), fit)
            summary = {}
            for split in splits:
                sq = split_queries(split, stats)
                man = query_manifest_checked(split)
                arrays, result = score_forecasts(split, forecasts[split], sq, man, stats, sigma_cal)
                result.update({"forecast_seconds": timings[split], "chunk_used": forecasts[split]["chunk_used"],
                               "checkpoint_sha256": eprot["checkpoint_sha256"], "model_id": prot.get("model_id"),
                               "seed": prot.get("seed"), "module": prot["module"]})
                write_split(eval_dir, split, forecasts[split], arrays, result, sq, man)
                m = result.get("metrics", {})
                summary[split] = {"ranking_eligible": result["ranking_eligible"],
                                  "not_eligible_reasons": result["not_eligible_reasons"],
                                  "selection_metric": result["selection_metric"]["value"],
                                  **{k: m[k]["micro_mean"] for k in ("J__prefix_H1", "J__prefix_H40",
                                                                      "nMSE_train__prefix_H40") if k in m}}
                print(json.dumps({"stage": "scored", "split": split, **summary[split]}), flush=True)
            after = state_hash(model)
            unchanged = after == before
            complete = {"status": "complete", "schema": SCHEMA, "splits": summary, "sigma_cal": sigma_cal,
                        "run_dir": run_dir, "out_dir": eval_dir, "checkpoint_check": ckc,
                        "sigma_cal_refit_equals_archived": fit.get("refit_equals_archived"),
                        "deterministic": _DETERMINISM.get("deterministic"),
                        "model_state_unchanged": unchanged,
                        "ranking_eligible": unchanged and all(v["ranking_eligible"] for v in summary.values()),
                        "checkpoint_sha256": eprot["checkpoint_sha256"], "checkpoint_step": ck.get("step"),
                        "elapsed_s": time.time() - t0, "forecast_seconds": timings}
            R.atomic_json(os.path.join(eval_dir, "COMPLETE.json"), complete)
            return complete
        except Exception as exc:
            R.atomic_json(os.path.join(eval_dir, "FAILED.json"),
                          {"status": "exception", "exception": repr(exc), "traceback": traceback.format_exc(),
                           "elapsed_s": time.time() - t0})
            raise


DETERMINISTIC_CUBLAS = ":4096:8"


def configure_determinism(deterministic: bool) -> dict:

    inherited = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if deterministic:
        if not inherited:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    else:
        if inherited is not None:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    try:
        cuda_init = bool(torch.cuda.is_initialized())
    except Exception:
        cuda_init = None
    return {"deterministic": bool(deterministic),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "inherited_CUBLAS_WORKSPACE_CONFIG": inherited,
            "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_initialized_before": cuda_init}


_DETERMINISM = {"deterministic": None}


def setup_runtime(device: str, deterministic: bool = False) -> dict:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    info = configure_determinism(deterministic)
    _DETERMINISM.clear()
    _DETERMINISM.update(info)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description='Evaluate a trained run on VAL, TEST and TEST_B (writes PER_QUERY.npz, PREDICTIONS.npz, RESULT.json).', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", default=None, help="model module (default: from PROTOCOL.json)")
    ap.add_argument("--run", required=True, help="run dir <runs-root>/runs/<dataset>/<model_id>/<seed>")
    ap.add_argument("--splits", default="val,test,test_b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=720)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--allow-incomplete-train", action="store_true", help="smoke only")
    ap.add_argument("--deterministic", action="store_true",
                    help="CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms (default: off)")
    ap.add_argument("--out-dir", default=None, help="write outputs here instead of <run>/eval (run dir read-only)")
    ap.add_argument("--sigma-cal-from", default=None, help="archived VAL_SIGMA_CALIBRATION.json to score with")
    ap.add_argument("--verify-ckpt-sha", action="store_true", help="require ckpt sha == COMPLETE.json best_pt_sha256")
    ap.add_argument("--provenance", action="append", default=[], help="KEY=VALUE recorded in EVALUATION_PROTOCOL")
    ap.add_argument("--rssm-eval-mode", default=None, choices=["mode", "sample16"],
                    help="evaluation mode of the RSSM (latent mode or mean of 16 keyed samples)")
    P.add_cli_options(ap)
    args = ap.parse_args(argv)
    if args.rssm_eval_mode is not None:
        from baselines import keyed_sampling
        keyed_sampling.set_eval_mode(args.rssm_eval_mode)
        args.provenance.append(f"eval_mode={args.rssm_eval_mode}")
    setup_runtime(args.device, args.deterministic)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    bad = [s for s in splits if s not in ALL_SPLITS]
    if bad:
        ap.error(f"unknown splits {bad}")
    try:
        res = evaluate_run(args.run, args.module, splits, args.device, args.chunk, args.ckpt,
                           args.allow_incomplete_train, out_dir=args.out_dir, sigma_cal_from=args.sigma_cal_from,
                           verify_ckpt_sha=args.verify_ckpt_sha, provenance=R.parse_variant(args.provenance))
    except AlreadyComplete as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}), flush=True)
        return 3
    print(json.dumps({"status": res["status"], "ranking_eligible": res["ranking_eligible"],
                      "elapsed_s": round(res["elapsed_s"], 2)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
