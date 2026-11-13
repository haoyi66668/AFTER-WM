import json, os, sys, tempfile
import numpy as np
from jshc import config as C, gen as G
from simcore import gen as mgen

R = []


def check(name, ok, detail=""):
    R.append({"test": name, "passed": bool(ok), "detail": str(detail)[:300]})
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""), flush=True)


def main():
    tmp = sys.argv[sys.argv.index("--tmp") + 1] if "--tmp" in sys.argv else tempfile.mkdtemp(prefix="jshc_ut_")
    ns = C.NAMESPACE + "-unittest"
    P = C.make_protocol(ns)
    check("01_shared_base_keys", C.shared_keys_ok(P))
    check("02_protocol_values", P["joint_switch"]["candidate_joints"] == [1, 2, 3, 4, 5] and P["joint_switch"]["switch_period"] == 100
          and P["joint_switch"]["phase_range"] == [0, 100] and P["T"] == 300 and P["env_id"] == "HalfCheetah-v5"
          and G.switch_times(P, 0) == [0, 100, 200] and G.switch_times(P, 50) == [0, 50, 150, 250] and G.switch_times(P, 99) == [0, 99, 199, 299])
    f, d, ph, s = G.joint_schedule(P, 3, 1, 2, 0)
    f2, d2, ph2, s2 = G.joint_schedule(P, 3, 1, 2, 0)
    check("03_schedule_deterministic", np.array_equal(f, f2) and d == d2 and s == s2 and ph == ph2)
    sch = [G.joint_schedule(P, i, c, q, o) for i in range(40) for c in (0, 1) for q in range(6) for o in (0, 1)]
    ok4 = True
    for fa, dr, pha, _ in sch:
        b = G.switch_times(P, pha) + [300]
        ok4 &= len(dr) == len(b) - 1 and all((fa[x:y] == dr[k]).all() for k, (x, y) in enumerate(zip(b[:-1], b[1:])))
        ok4 &= all(b2 - b1 == 100 for b1, b2 in zip(b[1:-2], b[2:-1]))
    check("04_schedule_range_and_pieces", ok4 and np.isin(np.stack([x[0] for x in sch]), [1, 2, 3, 4, 5]).all())
    alld = [j for x in sch for j in x[1]]
    rep = [x[1][k] == x[1][k - 1] for x in sch for k in range(1, len(x[1]))]
    vals, cnt = np.unique(alld, return_counts=True)
    phs = [x[2] for x in sch]
    check("05_all_five_joints_drawn_repeats_possible_phase_spread", list(vals) == [1, 2, 3, 4, 5] and any(rep) and not all(rep)
          and min(phs) >= 0 and max(phs) <= 99 and len(set(phs)) > 50, dict(zip(vals.tolist(), cnt.tolist())))
    ch = G.changed_flags(f)
    bt = G.switch_times(P, ph)
    check("06_changed_flags", ch[0] == 0 and set(np.nonzero(ch)[0].tolist()) <= set(bt[1:])
          and all(ch[t] == int(f[t] != f[t - 1]) for t in bt[1:]))
    seeds_differ = len({G.joint_schedule(P, 0, 0, q, 0)[3] for q in range(6)} | {G.joint_schedule(P, 0, 1, q, 0)[3] for q in range(6)}
                       | {G.joint_schedule(P, 0, 0, q, 1)[3] for q in range(6)}) == 18
    check("07_schedule_seeds_distinct", seeds_differ)
    adapter, mujoco = mgen.make_adapter(P)
    designs = G.design_parents(P, {"train": 2, "val": 1, "test": 1}, 0)
    check("08_design_ids", [x["parent_id"] for x in designs] == ["jshc-train-0000", "jshc-train-0001", "jshc-val-0000", "jshc-test-0000"]
          and G.design_parents(P, {"test": 1}, 1)[0]["parent_id"] == "jshc-testB-0000")
    snap, anc = mgen.discover_anchor(adapter, P, designs[0], 0)
    acts, _ = mgen.action_plan(P, designs[0]["cohort_kinds"][0], 0, 0, 0)
    nom = G.rollout(adapter, snap, acts, np.full(300, -1, np.int8))
    ref = mgen.rollout(adapter, snap, acts, np.ones(300), np.ones(300), np.zeros((300, 3)))
    check("09_nominal_equals_base_rollout", np.array_equal(nom, ref))
    adapter.restore_integration_snapshot(snap, parameters={"friction_scale": 1.0, "gain_scale": 1.0})
    a = acts[5].copy(); a[3] = 0.0
    adapter.apply_absolute_dynamics({"friction_scale": 1.0, "gain_scale": 1.0})
    adapter.step_with_impulse(a, None)
    data = adapter._data
    check("10_mask_zero_actuator_force", float(data.actuator_force[3]) == 0.0 and float(data.ctrl[3]) == 0.0
          and float(np.abs(data.actuator_force).sum()) > 0, data.actuator_force.tolist())
    fj = np.full(300, 2, np.int8)
    o1 = G.rollout(adapter, snap, acts, fj)
    o2 = G.rollout(adapter, snap, acts, fj)
    check("11_replay_exact", np.array_equal(o1, o2))
    check("12_mask_changes_trajectory", not np.array_equal(o1, nom) and np.array_equal(o1[0], nom[0]))
    fj2 = np.full(300, -1, np.int8); fj2[150:] = 4
    o3 = G.rollout(adapter, snap, acts, fj2)
    check("13_mask_causal", np.array_equal(o3[:151], nom[:151]) and not np.array_equal(o3[151:], nom[151:]))
    m1 = G.collect(os.path.join(tmp, "a"), P, {"train": 2, "test": 1}, 0, workers=1, log=lambda x: None)
    m2 = G.collect(os.path.join(tmp, "b"), P, {"train": 2, "test": 1}, 0, workers=2, log=lambda x: None)
    same = all(m1["files"][s]["array_sha256"] == m2["files"][s]["array_sha256"] for s in m1["files"])
    check("14_end_to_end_deterministic_any_worker_count", same and all(m1["checks"].values()))
    z = G.load_npz(os.path.join(tmp, "a", "jshc_train.npz"))
    check("15_arrays_and_order", z["obs"].shape == (24, 301, 17) and z["act"].shape == (24, 300, 6)
          and z["sequence"].tolist()[:6] == list(C.SEQUENCE_IDS) and z["cohort"].tolist()[:12] == [0] * 6 + [1] * 6
          and set(z) == set(G.ARRAY_KEYS))
    check("16_obs_has_no_label_columns", z["obs"].shape[-1] == 17)
    try:
        G.collect(os.path.join(tmp, "a"), P, {"train": 1}, 0)
        check("17_never_overwrite", False)
    except FileExistsError:
        check("17_never_overwrite", True)
    out = {"passed": all(r["passed"] for r in R), "n": len(R), "results": R}
    print(json.dumps({"passed": out["passed"], "n": len(R), "n_passed": sum(r["passed"] for r in R)}))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
