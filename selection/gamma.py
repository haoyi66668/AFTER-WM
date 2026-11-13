from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from afterwm import core as C

C.configure_runtime(1, deterministic=True)

from pipeline import evaluate as E
from pipeline import registry as P
from afterwm import base as S
from afterwm.variants import fast_only as fast

GRID = [0.05, 0.08, 0.1, 0.127, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
H = (1, 5, 10, 20, 40)
EXT_LOW, EXT_HIGH = 0.03, 1.4
RULE = ("argmin VAL five-step mean prefix nMSE of the stage-1 base alone along the deployment path (fast-only model "
        "at initialisation); ties -> larger gamma")
RULE_EXT = ("if the grid choice is the lower boundary 0.05, 0.03 joins the choice; if it is the upper boundary 1.0, "
            "1.4 joins; final = argmin over the grid plus that value (ties -> larger gamma); at most one extension")


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def prefix_nmse(mean, targets):
    se = ((mean - targets) ** 2).mean(-1)
    pref = np.cumsum(se, axis=1) / np.arange(1, se.shape[1] + 1)
    return np.stack([pref[:, h - 1] for h in H], 1)


def score_values(seed: int, sq: dict, targets: np.ndarray, stats, device: str, batch: int, gammas) -> dict:
    cfg = fast.default_config("fast_native")
    m = fast.build(cfg, int(seed), device, stats)
    m.eval()
    scores, per_h = {}, {}
    for g in gammas:
        m.__dict__["_fp"] = None
        m.gamma = float(g)
        means = []
        with torch.no_grad():
            for s in range(0, targets.shape[0], batch):
                sl = slice(s, s + batch)
                t = lambda k: torch.from_numpy(np.ascontiguousarray(sq[k][sl], dtype=np.float32)).to(device)
                zc, ac, af, _ = m.native_inputs(t("ctx_obs"), t("ctx_act"), t("fut_act"))
                pred, logp = m.rollout_fast(zc, ac, af)
                logw = torch.log_softmax(logp.double(), -1)
                mean_native = (logw.exp()[:, None, :, None] * pred.double()).sum(2)
                means.append(m.fmap.obs_to_train(mean_native).cpu().numpy())
        pn = prefix_nmse(np.concatenate(means, 0), targets)
        scores[g] = float(pn.mean())
        per_h[g] = [float(x) for x in pn.mean(0)]
    ident = getattr(m, "parent_identity", None) or {}
    return {"scores": scores, "per_horizon": per_h, "base_sha256": ident.get("sha256")}


def choose(grid_scores: dict, ext_scores: dict) -> dict:
    grid = list(GRID)
    g0 = min(grid, key=lambda g: (grid_scores[g], -g))
    ext = EXT_LOW if g0 == min(grid) else (EXT_HIGH if g0 == max(grid) else None)
    cand = dict((g, grid_scores[g]) for g in grid)
    if ext is not None:
        cand[ext] = ext_scores[ext]
    final = min(cand, key=lambda g: (cand[g], -g))
    return {"grid_choice": g0, "extension_needed": ext is not None, "extension_value": ext,
            "final": final, "extension_chosen": ext is not None and final == ext}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select the inverse routing temperature gamma of each seed on VAL "
                                             "(writes <runs-root>/gamma/<dataset>/GAMMA.json).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--batch", type=int, default=240)
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="default: the pinned seeds")
    ap.add_argument("--check-out", default=None, help="also write the result JSON here")
    P.add_cli_options(ap)
    a = ap.parse_args(argv)
    if P.ARM is not None:
        raise SystemExit("gamma is fixed by the sensitivity settings; no selection for an --arm")
    ds = P.DATASET
    out_path = S.gamma_file(ds)
    if a.write and os.path.exists(out_path):
        raise SystemExit(f"refusing to overwrite {out_path}")
    seeds = [int(s) for s in (a.seeds or sorted(S.BASE_SHA256))]
    stats = E.get_stats()
    sq = E.split_queries("val", stats)
    targets = sq["fut_obs"].astype(np.float64)
    res = {"dataset": ds, "frozen": bool(a.write), "grid": GRID, "rule": RULE, "extension_rule": RULE_EXT,
           "extension_values": [EXT_LOW, EXT_HIGH], "split": "val", "val_sha256": P.SPLIT_SHA256["val"],
           "stats_sha256": stats.sha256(), "seeds": {}}
    t0 = time.time()
    for seed in seeds:
        want = S.pinned_base_sha256(seed)
        sv = score_values(seed, sq, targets, stats, a.device, a.batch, list(GRID) + [EXT_LOW, EXT_HIGH])
        if sv["base_sha256"] != want:
            raise RuntimeError(f"seed {seed}: scored base {sv['base_sha256']} is not the pinned base {want}")
        grid_scores = {g: sv["scores"][g] for g in GRID}
        ext_scores = {EXT_LOW: sv["scores"][EXT_LOW], EXT_HIGH: sv["scores"][EXT_HIGH]}
        c = choose(grid_scores, ext_scores)
        res["seeds"][str(seed)] = {
            "scores": {str(g): v for g, v in grid_scores.items()},
            "per_horizon": {str(g): sv["per_horizon"][g] for g in GRID},
            "extension_scores": {str(k): v for k, v in ext_scores.items()},
            "grid_choice": c["grid_choice"], "extension_needed": c["extension_needed"],
            "extension_value": c["extension_value"], "extension_chosen": c["extension_chosen"], "best": c["final"],
            "base_sha256": want}
        print(json.dumps({"seed": seed, "grid_choice": c["grid_choice"], "extension": c["extension_value"],
                          "final": c["final"]}), flush=True)
    res["gamma_by_seed"] = {s: v["best"] for s, v in res["seeds"].items()}
    res["base_sha256_by_seed"] = {s: v["base_sha256"] for s, v in res["seeds"].items()}
    res["wall_seconds"] = time.time() - t0
    if a.check_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.check_out)), exist_ok=True)
        with open(a.check_out, "w") as f:
            json.dump(res, f, indent=1)
    if a.write:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, out_path)
        print(json.dumps({"written": out_path, "sha256": _sha(out_path), "gamma_by_seed": res["gamma_by_seed"]}),
              flush=True)
    else:
        print(json.dumps(res["gamma_by_seed"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
