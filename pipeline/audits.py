from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import dataclasses
import json
import math
import re
import sys
import time
import traceback

import numpy as np
import torch

from . import data as D
from . import evaluate as E
from . import paramcount as PC
from . import registry as P
from . import run_io as R
from .eval_api import QueryBatch

if __name__ == "__main__":
    sys.modules.setdefault("pipeline.audits", sys.modules["__main__"])

SCHEMA = "afterwm_audit_v1"
KSET = (1, 2, 3, 5, 10, 20, 39)
NOISE_SCALE = 37.0
NOISE_SEED = 20261012
PC_DELTA, PC_FRAC = 1e-4, 0.95
TOL_NUMERIC = 1e-2
FIELDS = ("mean", "comp_mean", "comp_logw", "comp_var")
QUERYBATCH_FIELDS = ("ctx_obs", "ctx_act", "fut_act", "key")
INFRA_PREFIXES = ("pipeline/", "selection/", "experiments/", "tests/")
A5_PATTERNS = [
    ("hidden_key_literal", re.compile(r"""(['"])(friction|gain|friction_eff|gain_eff|impulse|seg|parent|cohort|sequence)\1""")),
    ("load_analysis_labels", re.compile(r"\bload_analysis_labels\b")),
    ("ANALYSIS_KEYS", re.compile(r"\bANALYSIS_KEYS\b")),
    ("data_file_name", re.compile(r"[a-z0-9]+_(train|val|test|testB)\.npz")),
    ("data_dir_access", re.compile(r"\b(DATA_DIR|SPLIT_FILES|split_path|_checked_npz)\b")),
    ("query_manifest", re.compile(r"\bquery_manifest\b|QUERY_MANIFEST")),
]
A5_INFO = [("np_load", re.compile(r"\b(np|numpy)\.load\s*\(")), ("torch_load", re.compile(r"\btorch\.load\s*\("))]


class AlreadyComplete(RuntimeError):
    pass


_T0 = [None]


def _progress(audit, **kw):
    now = time.time()
    if _T0[0] is None:
        _T0[0] = now
    print(json.dumps({"stage": "audit_progress", "audit": audit, **kw, "t_s": round(now - _T0[0], 1)}), flush=True)


def _diff(a, b) -> float:
    if a is None and b is None:
        return 0.0
    if (a is None) != (b is None):
        return math.inf
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return math.inf
    if a.size == 0:
        return 0.0
    with np.errstate(invalid="ignore"):
        d = np.abs(a - b)
    both_nan = np.isnan(a) & np.isnan(b)
    d[both_nan] = 0.0
    d[np.isnan(d)] = math.inf
    return float(d.max())


def fdiff(f1: dict, f2: dict, k: int | None = None, fields=FIELDS) -> dict:
    out = {}
    for f in fields:
        a, b = f1.get(f), f2.get(f)
        if k is not None and f != "comp_logw" and a is not None and b is not None:
            a, b = a[:, :k], b[:, :k]
        out[f] = _diff(a, b)
    return out


def _max(d: dict) -> float:
    return max(d.values()) if d else 0.0


def _finite(f: dict) -> bool:
    return all(np.isfinite(f[k]).all() for k in FIELDS if f.get(k) is not None and k != "comp_logw") and \
        (f.get("comp_logw") is None or not np.isnan(f["comp_logw"]).any())


def choose_perm(fut: np.ndarray, k_min: int = 1):
    n = fut.shape[0]
    best = None
    for shift in (30, 37, 61, 97, 151, 211, 293, 1, 2, 3, 5, 7):
        if shift >= n:
            continue
        perm = (np.arange(n) + shift) % n
        changed = np.any(fut[perm, k_min:] != fut[:, k_min:], axis=(1, 2))
        if best is None or changed.sum() > best[2]:
            best = (perm, shift, int(changed.sum()))
        if changed.all():
            break
    return best


def audit_a1(fm, sq, device, f0, tol, ks=KSET, chunk=720) -> dict:
    fut = sq["fut_act"]
    rng = np.random.default_rng(NOISE_SEED)
    noise = (rng.standard_normal(fut.shape) * NOISE_SCALE).astype(fut.dtype)
    perm, shift, n_changed = choose_perm(fut)
    causality, ok_c = [], True
    for k in ks:
        for kind in ("permuted", "noise"):
            f2 = fut.copy()
            f2[:, k:] = fut[perm, k:] if kind == "permuted" else fut[:, k:] + noise[:, k:]
            perturbed = int(np.any(f2[:, k:] != fut[:, k:], axis=(1, 2)).sum())
            F = E.forecast_arrays(fm, sq, device, chunk, fut_act=f2)
            d = fdiff(F, f0, k)
            passed = _max(d) <= tol
            _progress("A1", part="causality", k=int(k), kind=kind, passed=bool(passed))
            ok_c &= passed
            causality.append({"k": int(k), "kind": kind, "queries_perturbed": perturbed,
                              "max_diff_before_k": d, "max_diff_from_k_on": _diff(F["mean"][:, k:], f0["mean"][:, k:]),
                              "passed": passed})
    pcs, ok_p = [], True
    for k in (0,) + tuple(ks):
        f2 = fut.copy()
        f2[:, k] = fut[:, k] + noise[:, k]
        F = E.forecast_arrays(fm, sq, device, chunk, fut_act=f2)
        with np.errstate(invalid="ignore"):
            delta = np.abs(F["mean"][:, k].astype(np.float64) - f0["mean"][:, k]).max(-1)
        frac = float(np.mean(delta > PC_DELTA))
        before = fdiff(F, f0, k) if k > 0 else {}
        passed = frac >= PC_FRAC and _max(before) <= tol
        _progress("A1", part="positive_control", k=int(k), passed=bool(passed))
        ok_p &= passed
        pcs.append({"k": int(k), "frac_changed": frac, "median_delta": float(np.nanmedian(delta)),
                    "max_diff_before_k": before, "passed": passed})
    return {"passed": bool(ok_c and ok_p), "causality_passed": bool(ok_c), "positive_controls_passed": bool(ok_p),
            "k_set": list(ks), "perm_shift": int(shift), "perm_queries_changed_k1": n_changed,
            "noise": f"fut_act + {NOISE_SCALE}*N(0,1), seed {NOISE_SEED}",
            "positive_control_rule": f"|delta pred[:,k]| > {PC_DELTA} for >= {PC_FRAC:.0%} of queries",
            "causality": causality, "positive_controls": pcs}


def audit_a2(fm, sq, device, f0, tol, chunk=720) -> dict:
    static = tuple(f.name for f in dataclasses.fields(QueryBatch))
    obs_n, act_n, e, t = sq["obs_n"], sq["act_n"], sq["e"], sq["t"]
    cases, ok = [], static == QUERYBATCH_FIELDS
    for name, val in (("nan", np.nan), ("1e6", 1e6)):
        ctx_obs = np.empty_like(sq["ctx_obs"])
        ctx_act = np.empty_like(sq["ctx_act"])
        fut_act = np.empty_like(sq["fut_act"])
        poisoned_targets = 0
        for s in sorted(set(t.tolist())):
            idx = np.flatnonzero(t == s)
            o, a = obs_n.copy(), act_n.copy()
            o[:, s + 1:] = val
            o[:, : s - P.CONTEXT] = val
            a[:, : s - P.CONTEXT] = val
            a[:, s + P.MAX_H:] = val
            w = D.windows(o, a, e[idx], t[idx], P.MAX_H)
            ctx_obs[idx], ctx_act[idx], fut_act[idx] = w["ctx_obs"], w["ctx_act"], w["fut_act"]
            fo = w["fut_obs"]
            poisoned_targets += int(np.all(np.isnan(fo) if np.isnan(val) else fo == val, axis=(1, 2)).sum())
        inputs_equal = (np.array_equal(ctx_obs, sq["ctx_obs"]) and np.array_equal(ctx_act, sq["ctx_act"])
                        and np.array_equal(fut_act, sq["fut_act"]))
        keys = D.context_keys(ctx_obs, ctx_act)
        sq2 = dict(sq, ctx_obs=ctx_obs, ctx_act=ctx_act, keys=keys)
        F = E.forecast_arrays(fm, sq2, device, chunk, fut_act=fut_act)
        _progress("A2", poison=name)
        d = fdiff(F, f0)
        fin = _finite(F)
        passed = bool(inputs_equal and fin and _max(d) <= tol and np.array_equal(keys, sq["keys"])
                      and poisoned_targets == len(t))
        ok &= passed
        cases.append({"poison": name, "query_inputs_identical": bool(inputs_equal),
                      "keys_identical": bool(np.array_equal(keys, sq["keys"])),
                      "targets_poisoned_queries": poisoned_targets, "forecast_finite": fin, "max_diff": d,
                      "passed": passed})
    return {"passed": bool(ok), "querybatch_fields": list(static), "querybatch_fields_ok": static == QUERYBATCH_FIELDS,
            "cases": cases}


def dup_spread(F: dict, man: dict) -> dict:
    out = {}
    for h in P.HORIZONS:
        gid = np.asarray(man[f"gid_in_H{h}"])
        groups = int(np.sum(np.bincount(gid, minlength=len(gid)) > 1))
        spread = 0.0
        for f in FIELDS:
            x = F.get(f)
            if x is None:
                continue
            xs = x if f == "comp_logw" else x[:, :h]
            spread = max(spread, _diff(xs, xs[gid]))
        out[str(h)] = {"groups_with_duplicates": groups, "queries_in_groups": int(np.sum(man[f"mult_in_H{h}"] > 1)),
                       "max_spread": spread}
    return out


def audit_a3(f0, man, tol, eval_dir=None, split="val") -> dict:
    res = {"audit_split": {split: dup_spread(f0, man)}, "eval_predictions": {}}
    ok = all(v["max_spread"] <= tol for v in res["audit_split"][split].values())
    if eval_dir and os.path.isdir(eval_dir):
        for s in getattr(E, "ALL_SPLITS", E.SPLITS):
            path = os.path.join(eval_dir, s, "PREDICTIONS.npz")
            if not os.path.exists(path):
                continue
            z = R.load_npz(path)
            m = E.query_manifest(s)
            if not (np.array_equal(z["e"], m["e"]) and np.array_equal(z["t"], m["t"])):
                res["eval_predictions"][s] = {"error": "query order mismatch"}
                ok = False
                continue
            F = {f: z.get(f) for f in FIELDS}
            sp = dup_spread(F, m)
            res["eval_predictions"][s] = sp
            ok &= all(v["max_spread"] <= tol for v in sp.values())
    res["passed"] = bool(ok)
    return res


def gradient_audit(module, cfg, seed, device, stats, steps=3) -> dict:
    from .trainer import TrainData
    rng = E.save_rng()
    try:
        torch.manual_seed(int(seed))
        with E.label_guard():
            m = module.build(cfg, int(seed), device, stats)
        m.train()
        opt = module.make_optimizer(m, cfg)
        cov = PC.optimizer_coverage(m, opt)
        td = TrainData(stats, int(seed), device)
        got = {n: False for n, p in m.named_parameters() if p.requires_grad}
        frozen_grad = set()
        clip = cfg.get("grad_clip")
        trainable = [p for p in m.parameters() if p.requires_grad]
        losses = []
        for i in range(steps):
            with E.label_guard():
                loss, _ = module.train_step(m, td.batch(i), i, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            losses.append(float(loss.detach()))
            for n, p in m.named_parameters():
                if p.grad is None:
                    continue
                if not p.requires_grad:
                    frozen_grad.add(n)
                elif bool((p.grad != 0).any()):
                    got[n] = True
            if clip:
                torch.nn.utils.clip_grad_norm_(trainable, float(clip))
            opt.step()
            if hasattr(module, "after_step"):
                module.after_step(m, i, cfg)
        params = dict(m.named_parameters())
        never = sorted(n for n, v in got.items() if not v)
        res = {"steps": steps, "losses": losses, "optimizer_coverage": cov,
               "frozen_with_grad": sorted(frozen_grad), "never_nonzero_grad": never,
               "never_nonzero_grad_numel": int(sum(params[n].numel() for n in never)),
               "trainable_tensors": len(got)}
        del m, opt, td
        return res
    finally:
        E.restore_rng(rng)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def audit_a4(module, model, fm, cfg, budget_class, seed, device, stats) -> dict:
    rep = fm.parameter_report()
    chk = PC.check_report(rep, model, budget_class)
    out = {"parameter_check": chk}
    ok = chk["passed"]
    if hasattr(module, "custom_train") or not hasattr(module, "train_step"):
        out["gradient_audit"] = {"skipped": "module uses custom_train (no generic train_step)"}
    else:
        ga = gradient_audit(module, cfg, seed, device, stats)
        out["gradient_audit"] = ga
        ok = ok and ga["optimizer_coverage"]["passed"] and not ga["frozen_with_grad"]
        out["warnings"] = ([f"{len(ga['never_nonzero_grad'])} trainable tensors "
                            f"({ga['never_nonzero_grad_numel']} params) got no nonzero gradient in "
                            f"{ga['steps']} updates"] if ga["never_nonzero_grad"] else [])
    out["passed"] = bool(ok)
    return out


def model_code_files(module) -> list:
    files = {os.path.relpath(os.path.abspath(module.__file__), R.CODE_ROOT)}
    for rel in R.imported_code_files():
        if not rel.startswith(INFRA_PREFIXES):
            files.add(rel)
    return sorted(files)


def audit_a5(module, protocol=None) -> dict:
    files = model_code_files(module)
    hits, exempt, info = [], [], []
    for rel in files:
        path = os.path.join(R.CODE_ROOT, rel)
        try:
            lines = open(path, encoding="utf-8").read().splitlines()
        except OSError as exc:
            hits.append({"file": rel, "line": 0, "rule": "unreadable", "text": repr(exc)})
            continue
        for i, line in enumerate(lines, 1):
            for rule, rx in A5_PATTERNS:
                if rx.search(line):
                    rec = {"file": rel, "line": i, "rule": rule, "text": line.strip()[:200]}
                    (exempt if "# A5-ok" in line else hits).append(rec)
            for rule, rx in A5_INFO:
                if rx.search(line):
                    info.append({"file": rel, "line": i, "rule": rule, "text": line.strip()[:200]})
    guard = None if protocol is None else protocol.get("label_guard")
    return {"passed": not hits, "files_scanned": files, "hits": hits, "exempt_lines": exempt,
            "info_np_or_torch_load": info, "trainer_runtime_label_guard": guard,
            "evaluator_runtime_label_guard": True, "behavioural_check": "A3 (duplicates differ only in hidden labels)"}


def _unpermute(F, order):
    inv = np.argsort(order)
    return {f: (F[f][inv] if F.get(f) is not None else None) for f in FIELDS}


def audit_a6(fm, model, sq, device, f0, f0b, tol, chunk, hash_before) -> dict:
    n = len(sq["keys"])
    rep = fdiff(f0b, f0)
    rev = np.arange(n)[::-1].copy()
    Frev = _unpermute(E.forecast_arrays(fm, sq, device, chunk, order=rev), rev)
    _progress("A6", part="reversed_order")
    half = max(1, (n + 1) // 2)
    Fcont = E.forecast_arrays(fm, sq, device, half)
    _progress("A6", part="contiguous_halves")
    inter = np.concatenate([np.arange(0, n, 2), np.arange(1, n, 2)])
    Finter_raw = E.forecast_arrays(fm, sq, device, half, order=inter)
    Finter = _unpermute(Finter_raw, inter)
    _progress("A6", part="interleaved_halves")
    Fb = E.forecast_arrays(fm, sq, device, min(97, chunk))
    _progress("A6", part="chunk_97")
    Fc = E.forecast_arrays(fm, sq, device, chunk)
    _progress("A6", part="persistence")
    d_order, d_batch, d_persist = fdiff(Frev, f0), fdiff(Fb, f0), fdiff(Fc, f0)
    d_comp = fdiff(Finter, Fcont)
    same_chunks = Fcont["chunk_used"] == Finter_raw["chunk_used"] == half
    hash_after = E.state_hash(model)
    checks = {
        "repeatability": {"max_diff": rep, "passed": _max(rep) <= tol},
        "persistence_after_other_batches": {"max_diff": d_persist, "passed": _max(d_persist) <= tol},
        "reversed_order": {"max_diff": d_order, "bitwise": _max(d_order) == 0.0, "passed": _max(d_order) <= tol},
        "batch_composition": {"chunk": half, "max_diff": d_comp, "bitwise": _max(d_comp) == 0.0,
                              "same_chunking": bool(same_chunks),
                              "passed": _max(d_comp) <= (tol if same_chunks else max(tol, TOL_NUMERIC))},
        "chunk_97": {"max_diff": d_batch, "bitwise": _max(d_batch) == 0.0,
                     "passed": _max(d_batch) <= max(tol, TOL_NUMERIC), "note": "different batch shape; gross check"},
        "weights_unchanged": {"before": hash_before, "after": hash_after, "passed": hash_before == hash_after},
    }
    return {"passed": all(c["passed"] for c in checks.values()), "tol_numeric": TOL_NUMERIC, "checks": checks}


def run_audits(module, model, cfg, *, budget_class, seed, device="cuda", split="val", stats=None, chunk=720,
               ks=KSET, full_k=False, protocol=None, eval_dir=None, fm=None) -> dict:
    t0 = time.time()
    stats = stats if stats is not None else E.get_stats()
    sq = E.split_queries(split, stats)
    man = E.query_manifest(split)
    fm = fm if fm is not None else module.forecaster(model, cfg)
    model.eval()
    hash_before = E.state_hash(model)
    f0 = E.forecast_arrays(fm, sq, device, chunk)
    _progress("A0", part="reference")
    f0b = E.forecast_arrays(fm, sq, device, chunk)
    floor = _max(fdiff(f0b, f0))
    _progress("A0", part="repeat", noise_floor=floor)
    tol = 0.0 if floor == 0.0 else max(1e-6, 10.0 * floor)
    ks = tuple(range(1, P.MAX_H)) if full_k else tuple(ks)
    audits = {}

    def _run(name, fn):
        ts = time.time()
        try:
            res = fn()
        except Exception as exc:
            res = {"passed": False, "exception": repr(exc), "traceback": traceback.format_exc()}
        res["seconds"] = round(time.time() - ts, 3)
        audits[name] = res
        print(json.dumps({"stage": "audit", "audit": name, "passed": res["passed"], "seconds": res["seconds"]}),
              flush=True)

    _run("A1", lambda: audit_a1(fm, sq, device, f0, tol, ks, chunk))
    _run("A2", lambda: audit_a2(fm, sq, device, f0, tol, chunk))
    _run("A3", lambda: audit_a3(f0, man, tol, eval_dir, split))
    _run("A4", lambda: audit_a4(module, model, fm, cfg, budget_class, seed, device, stats))
    _run("A5", lambda: audit_a5(module, protocol))
    _run("A6", lambda: audit_a6(fm, model, sq, device, f0, f0b, tol, chunk, hash_before))
    return {"schema": SCHEMA, "split": split, "queries": int(len(sq["keys"])), "noise_floor": floor,
            "tol_exact": tol, "tol_numeric": TOL_NUMERIC, "k_set": list(ks),
            "all_passed": all(a["passed"] for a in audits.values()),
            "failed": [k for k, a in audits.items() if not a["passed"]],
            "audits": audits, "elapsed_s": round(time.time() - t0, 2),
            "forecaster": {"name": getattr(fm, "name", None), "stochastic": bool(getattr(fm, "stochastic", False)),
                           "needs_grad_at_eval": bool(getattr(fm, "needs_grad_at_eval", False))}}


def audit_run(run_dir: str, module_name=None, split="val", device="cuda", chunk=720, ckpt="best.pt",
              full_k=False, out_name="AUDIT.json") -> dict:
    run_dir = os.path.abspath(run_dir)
    out = os.path.join(run_dir, out_name)
    with R.FileLock(os.path.join(run_dir, ".audit.lock")):
        if os.path.exists(out):
            raise AlreadyComplete(f"{out} exists; refusing to overwrite")
        mod, model, cfg, prot, stats, ck = E.load_run(run_dir, module_name, device, ckpt)
        res = run_audits(mod, model, cfg, budget_class=prot["budget_class"], seed=int(prot["seed"]), device=device,
                         split=split, stats=stats, chunk=chunk, full_k=full_k, protocol=prot,
                         eval_dir=os.path.join(run_dir, "eval"))
        res.update({"run_dir": run_dir, "run_dir_realpath": os.path.realpath(run_dir),
                    "determinism": dict(E._DETERMINISM), "module": prot["module"], "model_id": prot.get("model_id"),
                    "seed": prot.get("seed"), "budget_class": prot["budget_class"], "checkpoint": ckpt,
                    "checkpoint_sha256": R.sha256_file(os.path.join(run_dir, ckpt)), "checkpoint_step": ck.get("step"),
                    "env": R.env_info(device), "code_files": R.code_manifest(), "time": R.now_iso()})
        R.atomic_json(out, res)
        return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Post-training audits of a run (label guard, query-batch invariance, gradient coverage).', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--module", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test", "test_b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=720)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--full-k", action="store_true", help="k = 1..39 instead of the 7-k set")
    ap.add_argument("--out-name", default="AUDIT.json", help="output file name in the run dir (never overwritten)")
    P.add_cli_options(ap)
    args = ap.parse_args(argv)
    E.setup_runtime(args.device, deterministic=True)
    try:
        res = audit_run(args.run, args.module, args.split, args.device, args.chunk, args.ckpt, args.full_k,
                        args.out_name)
    except AlreadyComplete as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}), flush=True)
        return 3
    print(json.dumps({"status": "complete", "all_passed": res["all_passed"], "failed": res["failed"],
                      "elapsed_s": res["elapsed_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
