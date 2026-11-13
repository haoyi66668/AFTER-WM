from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from control import plant as PT
from control import spec as SP


def designs(protocol, n, seed_offset):
    from jshc import gen as JG
    return JG.design_parents(protocol, {"test": n}, seed_offset)


def joint_draws(protocol, index, seed_offset):
    from simcore import gen as G
    seed = G.derive_seed(protocol["namespace"], "mpc_joints", seed_offset, index)
    rng = np.random.default_rng(seed)
    cand = list(SP.CANDIDATE_JOINTS)
    j0 = int(cand[int(rng.integers(0, len(cand)))])
    o1 = [c for c in cand if c != j0]
    j1 = int(o1[int(rng.integers(0, len(o1)))])
    o2 = [c for c in cand if c != j1]
    j2 = int(o2[int(rng.integers(0, len(o2)))])
    return (j0, j1, j2), seed


def fail_array(draws):
    T = SP.T_WARM + SP.K_MPC
    f = np.full(T, draws[0], np.int8)
    f[SP.T_WARM + SP.K_SWITCH[0]:] = draws[1]
    f[SP.T_WARM + SP.K_SWITCH[1]:] = draws[2]
    return f


def start_episode(plant, protocol, design, seed_offset, fail, warm_act=None):
    from simcore import gen as G
    snap, anchor = G.discover_anchor(plant.ad, protocol, design, seed_offset)
    plan_seed = None
    if warm_act is None:
        acts, plan_seed = G.action_plan(protocol, design["cohort_kinds"][0], design["index"], 0, seed_offset)
        warm_act = np.asarray(acts[:SP.T_WARM], np.float32).copy()
    plant.restore_anchor(snap)
    obs = [np.asarray(snap.flat_observation, np.float32).copy()]
    for t in range(SP.T_WARM):
        res, _ = plant.step(warm_act[t], fail[t])
        if res.episode_end:
            raise RuntimeError("episode ended during warm-up")
        obs.append(res.flat_observation.copy())
    anchor = dict(anchor)
    anchor["plan_seed"] = plan_seed
    return anchor, np.asarray(warm_act, np.float32), np.asarray(obs, np.float32)


def generate(split, n):
    so = SP.SEED_OFFSET[split]
    pl = PT.Plant(SP.NS[split])
    prot = pl.protocol
    ds = designs(prot, n, so)
    warm_act, warm_obs, fails, eps = [], [], [], []
    for d in ds:
        draws, jseed = joint_draws(prot, d["index"], so)
        f = fail_array(draws)
        anchor, wa, wo = start_episode(pl, prot, d, so, f)
        warm_act.append(wa)
        warm_obs.append(wo)
        fails.append(f)
        eps.append({"index": d["index"], "parent_id": d["parent_id"], "contact_regime": d["contact_regime"],
                    "kind": d["cohort_kinds"][0], "joints": list(draws), "joint_seed": jseed,
                    "anchor": {k: anchor[k] for k in ("parent_seed", "attempt", "anchor_step", "state_sha256",
                                                      "plan_seed")}})
    import gymnasium
    import mujoco
    meta = {"schema": "mpcjshc_episodes_v1", "namespace": prot["namespace"], "split": split, "seed_offset": so, "n": n,
            "T_warm": SP.T_WARM, "K_mpc": SP.K_MPC, "K_switch": list(SP.K_SWITCH),
            "versions": {"gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__},
            "episodes": eps}
    return {"warm_act": np.stack(warm_act), "warm_obs": np.stack(warm_obs), "fail": np.stack(fails)}, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(SP.SEED_OFFSET))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    arrays, meta = generate(a.split, a.n)
    arrays["meta_json"] = np.asarray(json.dumps(meta, default=float))
    tmp = a.out + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, a.out)
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(json.dumps({"out": a.out, "sha256": h, "n": a.n, "split": a.split,
                      "joints": [e["joints"] for e in meta["episodes"]][:8]}))


if __name__ == "__main__":
    main()
