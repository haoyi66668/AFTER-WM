from __future__ import annotations

import argparse
import json
import math
import os

MODES = ("mode", "sample16")
NATIVE = "mode"
GRID_SEEDS = (20260910, 20270913)
RULE = ("RSSM evaluation mode in {mode, sample16 (mean of 16 samples keyed by the query)}; VAL selection metric of "
        "the trained runs of seeds 20260910 and 20270913, mean over the seeds; lowest finite score wins; exact tie -> "
        "mode; a diverged seed is left out; no finite score -> mode; training is unchanged")


def choose(scores: dict) -> dict:
    finite = {m: v for m, v in scores.items() if v is not None and math.isfinite(v)}
    if not finite:
        return {"eval_mode": NATIVE, "tie": False, "reason": "no finite score"}
    best = min(finite, key=lambda m: (finite[m], 0 if m == NATIVE else 1))
    tie = len(finite) == len(MODES) and finite["mode"] == finite["sample16"]
    return {"eval_mode": best, "tie": tie}


def main(argv=None) -> int:
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="Select the RSSM evaluation mode on VAL. Inputs: VAL evaluations written by "
                                             "python -m pipeline.evaluate --splits val --rssm-eval-mode <mode> "
                                             "--out-dir <runs-root>/runs/<dataset>/rssm_mode/<mode>/<seed>.")
    ap.add_argument("--model-dir", default=None, help="default <runs-root>/runs/<dataset>/rssm")
    ap.add_argument("--out", default=None, help="default <runs-root>/runs/<dataset>/rssm_mode/CHOICE.json")
    P.add_cli_options(ap)
    a = ap.parse_args(argv)
    model_dir = a.model_dir or os.path.join(P.RUNS_DIR, "rssm")
    base = os.path.join(P.RUNS_DIR, "rssm_mode")
    included, excluded = [], {}
    for s in GRID_SEEDS:
        run = os.path.join(model_dir, str(s))
        if os.path.exists(os.path.join(run, "COMPLETE.json")):
            included.append(s)
        elif os.path.exists(os.path.join(run, "DIVERGED.json")):
            excluded[str(s)] = "diverged"
        else:
            raise SystemExit(f"{run}: training not finished")
    scores, per = {}, {}
    for mode in MODES:
        vals = []
        for s in included:
            res = json.load(open(os.path.join(base, mode, str(s), "val", "RESULT.json")))
            v = (res.get("selection_metric") or {}).get("value")
            per[f"{mode}/{s}"] = v
            vals.append(v)
        ok = vals and all(v is not None and math.isfinite(v) for v in vals)
        scores[mode] = sum(vals) / len(vals) if ok else None
    rec = {"dataset": P.DATASET, "rule": RULE, "included_seeds": included, "excluded": excluded, "per_seed": per,
           "scores": scores, **choose(scores)}
    out = a.out or os.path.join(base, "CHOICE.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({"scores": scores, "eval_mode": rec["eval_mode"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
