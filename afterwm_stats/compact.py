from __future__ import annotations

import argparse
import os

import numpy as np

HS = (1, 5, 10, 20, 40)
SPLITS = ("test", "test_b")
SEEDS = (20260910, 20270913, 20280916, 20290919, 20300922)
MODELS = ("afterwm", "afterwm_fast_only", "afterwm_slow_only", "afterwm_no_error_input", "afterwm_gamma1", "rssm", "tdmpc2", "tdmpc2_h40",
          "mlp", "grbal", "rebal", "mole", "mbcd", "gpmm", "mts3")
STEP_MODELS = ("afterwm", "rssm", "tdmpc2", "tdmpc2_h40", "mlp", "grbal", "rebal", "mole", "mbcd", "gpmm", "mts3",
               "afterwm_fast_only", "afterwm_slow_only")


def five(step):
    return np.mean([step[:, :h].mean(1) for h in HS], 0)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def manifests(prep):
    return {s: load(os.path.join(prep, f"QUERY_MANIFEST_{s}.npz")) for s in ("val",) + SPLITS}


def manifest_keys(man):
    d = {"val/parent": man["val"]["parent"].astype(str)}
    for s in SPLITS:
        d[f"{s}/parent"] = man[s]["parent"].astype(str)
        d[f"{s}/query_id"] = man[s]["query_id"].astype(str)
        if "gid_full_H40" in man[s]:
            d[f"{s}/canonical"] = np.asarray(man[s]["gid_full_H40"]) == np.arange(len(man[s]["query_id"]))
    return d


def per_query_run(eval_dir, man):
    out = {}
    for s in SPLITS:
        z = load(os.path.join(eval_dir, s, "PER_QUERY.npz"))
        if not np.array_equal(z["query_id"].astype(str), man[s]["query_id"].astype(str)):
            raise ValueError(f"{eval_dir} {s}: query order differs from the manifest")
        if not bool(np.asarray(z["finite_query"]).all()):
            raise ValueError(f"{eval_dir} {s}: non-finite queries")
        p = np.stack([np.asarray(z[f"nMSE_train__prefix_H{h}"], np.float64) for h in HS], 0)
        out[s] = {"ms": p.mean(0), "h1": np.asarray(z["nMSE_train__prefix_H1"], np.float64),
                  "h40": np.asarray(z["nMSE_train__endpoint_H40"], np.float64)}
    return out


def per_step_run(eval_dir, Y):
    from pipeline import scorer as S
    out = {}
    for s in SPLITS:
        with np.load(os.path.join(eval_dir, s, "PREDICTIONS.npz"), allow_pickle=False) as z:
            fs = set(z.files)
            cm, lw = S.as_components(np.asarray(z["mean"]), np.asarray(z["comp_mean"]) if "comp_mean" in fs else None,
                                     np.asarray(z["comp_logw"]) if "comp_logw" in fs else None)
        mu = S.mixture_mean_np(np.asarray(cm, np.float64), np.asarray(lw, np.float64))
        out[s] = ((mu - Y[s]) ** 2).mean(-1)
    return out


def eval_dir(template, runs_root, dataset, model, seed):
    return template.format(runs_root=runs_root, dataset=dataset, model=model, seed=seed)


def build_per_query(prep, template, runs_root, dataset, models, seeds, extra=()):
    man = manifests(prep)
    d = manifest_keys(man)
    for name, model in [(m, m) for m in models] + list(extra):
        for sd in seeds:
            ev = eval_dir(template, runs_root, dataset, model, sd)
            if not os.path.exists(os.path.join(ev, "test", "PER_QUERY.npz")):
                continue
            r = per_query_run(ev, man)
            for s in SPLITS:
                for k, v in r[s].items():
                    d[f"{name}/{sd}/{s}/{k}"] = v
    return d


def build_per_step(template, runs_root, dataset, models, seeds):
    from pipeline import evaluate as E
    stats = E.get_stats()
    Y = {s: np.asarray(E.split_queries(s, stats)["fut_obs"], np.float64) for s in SPLITS}
    d = {}
    for m in models:
        for sd in seeds:
            ev = eval_dir(template, runs_root, dataset, m, sd)
            if not os.path.exists(os.path.join(ev, "test", "PREDICTIONS.npz")):
                continue
            for s, x in per_step_run(ev, Y).items():
                d[f"{m}/{sd}/{s}"] = x.astype(np.float32)
    return d


def rows_interventions(rows_dir, dataset, seeds):
    d = {}
    names = {"intact": "intact", "base": "no_correction", "zero": "zero_errors", "delay4": "delay_4", "shuf": "other_sequence"}
    for sd in seeds:
        for s in SPLITS:
            p = os.path.join(rows_dir, f"afterwm__{sd}__{s}.npz")
            if os.path.exists(p):
                z = load(p)
                for v, name in names.items():
                    d[f"{dataset}/{sd}/{s}/{name}"] = five(np.asarray(z[f"{v}__step"], np.float64))
    return d


def rows_removal(rows_dir, dataset, seeds):
    d = {}
    for sd in seeds:
        p = os.path.join(rows_dir, f"{dataset}__{sd}.npz")
        if os.path.exists(p):
            z = load(p)
            for s in SPLITS:
                for v, name in (("intact", "intact"), ("zero_fast", "no_fast"), ("zero_slow", "no_slow")):
                    st = np.asarray(z[f"{s}|step_{v}"], np.float64)
                    d[f"{dataset}/{sd}/{s}/{name}/ms"] = five(st)
                    d[f"{dataset}/{sd}/{s}/{name}/h1"] = st[:, 0].copy()
                    d[f"{dataset}/{sd}/{s}/{name}/h40"] = st[:, 39].copy()
    return d


def rows_routing(rows_dir, dataset, seeds):
    d = {}
    for sd in seeds:
        p = os.path.join(rows_dir, f"{dataset}__{sd}.npz")
        if os.path.exists(p):
            z = load(p)
            for s in SPLITS:
                d[f"{dataset}/{sd}/{s}/intact"] = five(np.asarray(z[f"{s}|step_intact"], np.float64))
                d[f"{dataset}/{sd}/{s}/frozen_routing"] = five(np.asarray(z[f"{s}|step_route_noerr"], np.float64))
    return d


def rows_beyond(rows_dir, data_root, models, seeds):
    d = {}
    for s, fn in (("test", "escphc_test.npz"), ("test_b", "escphc_testB.npz")):
        L = load(os.path.join(data_root, "escphc_beyond", fn))
        e = np.repeat(np.arange(L["obs"].shape[0]), 5)
        d[f"{s}/parent"] = np.asarray([f"{'TEST' if s == 'test' else 'TEST_B'}:{p}" for p in L["parent"].astype(str)[e]])
    for m in models:
        for sd in seeds:
            p = os.path.join(rows_dir, f"{m}__{sd}.npz")
            if os.path.exists(p):
                z = load(p)
                for s in SPLITS:
                    st = np.asarray(z[f"beyond_{s}|step"], np.float64)
                    d[f"{m}/{sd}/{s}/ms"] = five(np.where(np.isfinite(st), st, np.nan))
    return d


def merge_save(path, d):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            old = {k: np.asarray(z[k]) for k in z.files}
        old.update(d)
        d = old
    np.savez_compressed(path, **d)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the compact outputs (outputs/...) from evaluation directories and "
                                             "analysis rows.")
    ap.add_argument("what", choices=("per-query", "per-step", "interventions", "removal", "routing", "beyond-range", "control"))
    ap.add_argument("--set", default=None, help="name of the query set (per-query): dataset name or escphc_ctx32/escphc_ctx128")
    ap.add_argument("--prepared", default=None, help="prepared directory with QUERY_MANIFEST_<split>.npz (per-query)")
    ap.add_argument("--eval-template", default="{runs_root}/runs/{dataset}/{model}/{seed}/eval")
    ap.add_argument("--models", default=None)
    ap.add_argument("--as", dest="as_name", default=None, help="store the (single) model under this name (per-query)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--rows", default=None, help="analysis rows directory (interventions, removal, routing, beyond-range)")
    ap.add_argument("--outputs", default="outputs")
    a, _ = ap.parse_known_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.what == "control":
        from control import analysis as CA
        from control import spec as SP
        rows = list(SP.MODEL_ROWS)
        vals, present = CA.load_runs(a.rows, rows, list(SP.SEEDS), SP.N_EPISODES["test"])
        path = os.path.join(a.outputs, "control", "returns.npz")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, rows=np.asarray(rows), seeds=np.asarray(SP.SEEDS), returns=vals, valid=present)
        return 0
    from pipeline import registry as P
    ds = P.DATASET
    if a.what == "per-query":
        models = a.models.split(",") if a.models else list(MODELS)
        name = a.set or ds
        prep = a.prepared or os.path.join(P.DATA_ROOT, "prepared", ds)
        if a.as_name:
            d = build_per_query(prep, a.eval_template, P.RUNS_ROOT, ds, [], seeds, extra=[(a.as_name, models[0])])
        else:
            d = build_per_query(prep, a.eval_template, P.RUNS_ROOT, ds, models, seeds)
        merge_save(os.path.join(a.outputs, "per_query", f"{name}.npz"), d)
    elif a.what == "per-step":
        models = a.models.split(",") if a.models else list(STEP_MODELS)
        merge_save(os.path.join(a.outputs, "per_step", f"{ds}.npz"), build_per_step(a.eval_template, P.RUNS_ROOT, ds, models, seeds))
    elif a.what == "interventions":
        merge_save(os.path.join(a.outputs, "analysis", "interventions.npz"), rows_interventions(a.rows, ds, seeds))
    elif a.what == "removal":
        merge_save(os.path.join(a.outputs, "analysis", "removal.npz"), rows_removal(a.rows, ds, seeds))
    elif a.what == "routing":
        merge_save(os.path.join(a.outputs, "analysis", "routing.npz"), rows_routing(a.rows, ds, seeds))
    else:
        models = a.models.split(",") if a.models else ["afterwm", "tdmpc2", "tdmpc2_h40", "rebal", "grbal", "mole", "mbcd",
                                                         "mts3", "rssm", "gpmm", "mlp"]
        merge_save(os.path.join(a.outputs, "analysis", "beyond_range.npz"), rows_beyond(a.rows, P.DATA_ROOT, models, seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
