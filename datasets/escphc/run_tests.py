from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from . import escp as E
from . import gen as G

NS = G.NAMESPACE + "-unittest"
RESULTS = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name, flush=True)


def main():
    from simcore import gen as mgen
    check("01 derive_seed identical to simcore", all(G.derive_seed(NS, "x", i, "y") == mgen.derive_seed(NS, "x", i, "y")
                                                   for i in range(50)))
    s = E.Schedule()
    cur, ok = None, True
    for j in range(2000):
        ind = s.before_transition()
        if ind is not None:
            cur = ind
        ok &= (cur if cur is not None else 0) == E.list_index_of_transition(j)
    check("03 literal ESCP counter == (j+1)//100", ok and E.list_index_of_transition(98) == 0
          and E.list_index_of_transition(99) == 1 and E.list_index_of_transition(198) == 1 and E.list_index_of_transition(199) == 2)
    tr, te = G.pools(NS)
    lo, hi = 1.5 ** -3, 1.5 ** 3
    mults = np.array([[t["body_mass"][0], t["dof_damping_1_dim"][0], t["gravity"][2] / -9.81] for t in tr + te])
    check("04 pools: 40 + 40, multipliers in [1.5^-3, 1.5^3], gravity x=y=0",
          len(tr) == 40 and len(te) == 40 and (mults >= lo).all() and (mults <= hi).all()
          and all(t["gravity"][0] == 0 and t["gravity"][1] == 0 for t in tr + te))
    tr2, te2 = G.pools(NS)
    check("05 pools deterministic", all(np.array_equal(a[k], b[k]) for a, b in zip(tr + te, tr2 + te2) for k in a))
    rs = np.random.RandomState(G.derive_seed(NS, "pools"))
    u = rs.uniform(-3, 3, size=(1,)), rs.uniform(-3, 3, size=(1,)), rs.uniform(-3, 3, size=(3,))
    check("06 sample order body_mass -> dof_damping_1_dim -> gravity(3)",
          tr[0]["body_mass"][0] == 1.5 ** u[0][0] and tr[0]["dof_damping_1_dim"][0] == 1.5 ** u[1][0]
          and tr[0]["gravity"][2] == -9.81 * 1.5 ** u[2][2])
    env, ap = G.make_env()
    m = env.model
    base = (m.body_mass.copy(), m.dof_damping.copy(), m.opt.gravity.copy())
    for _ in range(100):
        ap.set_task(tr[3])
    check("07 set_task absolute (no compounding after 100 writes)",
          m.body_mass[1] == base[0][1] * tr[3]["body_mass"][0] and np.array_equal(m.body_mass[2:], base[0][2:])
          and np.array_equal(m.dof_damping, base[1] * tr[3]["dof_damping_1_dim"][0]) and np.array_equal(m.opt.gravity, tr[3]["gravity"]))
    ap.set_task({"body_mass": np.array([1.0]), "dof_damping_1_dim": np.array([1.0]), "gravity": np.array([0, 0, -9.81])})
    check("08 restore to originals bitwise", np.array_equal(m.body_mass, base[0]) and np.array_equal(m.dof_damping, base[1])
          and np.array_equal(m.opt.gravity, base[2]))
    d = G.episode_design(NS, "train", 5)
    acts = G.episode_actions(d)
    lab = G.labels(d, tr)
    s0 = d["offset"]
    check("09 3 scheduled boundaries at 100-s0, 200-s0, 300-s0",
          list(lab["chg_state"]) == [100 - s0, 200 - s0, 300 - s0])
    obs, applied, w = G.simulate(env, ap, d, tr, acts)
    obs2, _, _ = G.simulate(env, ap, d, tr, acts)
    check("10 simulate deterministic, shape, finite, no warnings", np.array_equal(obs, obs2) and obs.shape == (301, 17)
          and np.isfinite(obs).all() and not any(w.values()))
    q = np.concatenate([env.data.qpos[1:], env.data.qvel]).astype(np.float32)
    check("11 last obs = qpos[1:] + qvel", np.array_equal(obs[-1], q))
    check("12 actions within 0.95 and low-amplitude windows applied",
          np.abs(acts).max() <= 0.95 and np.abs(acts[s0 + 110: s0 + 140]).max() <= 0.95 * 0.15 + 1e-7
          and np.abs(acts[s0 + 230: s0 + 255]).max() <= 0.95 * 0.15 + 1e-7)
    li = lab["list_index"]
    check("13 applied gravity per transition = scheduled task", np.array_equal(
        applied[:, 2], np.array([tr[d["task_list"][k]]["gravity"][2] for k in li])))
    d2 = dict(d)
    d2["task_list"] = [d["task_list"][0]] * 16
    obs_cf, _, _ = G.simulate(env, ap, d2, tr, acts)
    c = int(lab["chg_state"][0])
    eff = bool(lab["chg_eff"][0])
    check("14 trajectory identical before the first change state, different after (if effective)",
          np.array_equal(obs[:c], obs_cf[:c]) and ((not eff) or not np.array_equal(obs[c], obs_cf[c])))
    with tempfile.TemporaryDirectory() as td:
        arr = {"obs": obs[None], "act": acts[None, s0:s0 + 300], "parent": np.array(["a"])}
        G.save_npz_det(Path(td) / "a.npz", arr)
        G.save_npz_det(Path(td) / "b.npz", arr)
        check("15 deterministic npz bytes", (Path(td) / "a.npz").read_bytes() == (Path(td) / "b.npz").read_bytes())
    ok = True
    for i in range(40):
        dd = G.episode_design(NS, "test", i)
        lb = G.labels(dd, te)
        for t, g, k, sv in zip(G.EVAL_STARTS, lb["since_group"], lb["chg_k"], lb["since_steps"]):
            if g in (1, 2):
                ok &= (k >= 1) and ((k <= 20) == (g == 1))
            elif g in (3, 4):
                ok &= k == -1 and 0 <= sv < 64 and ((sv < 16) == (g == 3))
            else:
                ok &= k == -1 and (sv < 0 or sv >= 64)
    check("16 since-change groups consistent", ok)
    g0 = (0.0, 0.0, -9.81)
    sd = G.derive_seed(NS, "pools")
    p3 = E.make_pools(sd, g0, 3.0)
    check("17 make_pools(L=3) == default pools bitwise (and G.pools at the default L)",
          G.log_scale_limit() == 3.0 and G.protocol(NS)["escp"]["log_scale_limit"] == 3.0
          and all(np.array_equal(a[k], b[k]) for a, b in zip(p3[0] + p3[1], tr + te) for k in a))
    ok18 = True
    lm = lambda P: np.log(np.array([[t["body_mass"][0], t["dof_damping_1_dim"][0], t["gravity"][2] / -9.81]
                                    for t in P[0] + P[1]])) / np.log(1.5)
    for L in (1.0, 2.0, 4.0):
        pL = E.make_pools(sd, g0, L)
        ok18 &= bool(np.allclose(lm(pL), lm(p3) * (L / 3.0), rtol=0, atol=1e-9)) and bool((np.abs(lm(pL)) <= L + 1e-9).all())
        ok18 &= all(t["gravity"][0] == 0 and t["gravity"][1] == 0 for t in pL[0] + pL[1]) and len(pL[0]) == len(pL[1]) == 40
    check("18 L = 1, 2, 4: same uniform draws u (log1.5 multipliers = L/3 x the L = 3 values), |log1.5 m| <= L", ok18)
    G.set_log_scale_limit(2.0)
    tr2b, te2b = G.pools(NS)
    p2 = E.make_pools(sd, g0, 2.0)
    ok19 = G.protocol(NS)["escp"]["log_scale_limit"] == 2.0 and all(
        np.array_equal(a[k], b[k]) for a, b in zip(tr2b + te2b, p2[0] + p2[1]) for k in a)
    ok19 &= G.episode_design(NS, "train", 5) == d
    G.set_log_scale_limit(3.0)
    check("19 set_log_scale_limit(2) -> pools / protocol use L = 2; episode designs unchanged", ok19)
    bad = False
    try:
        G.set_log_scale_limit(5.0)
    except ValueError:
        bad = True
    check("20 L outside (0, 4] refused; default restored", bad and G.log_scale_limit() == 3.0)
    n_fail = sum(1 for _, c in RESULTS if not c)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
