from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from panda import env as E
from panda import gen

UNIT_NS = gen.PROTOCOL_PANDA["namespace"].rsplit("-", 1)[0] + "-unittest"
PANDA_XML_SHA = "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273"
SHA256SUMS_SHA = "4a4250e18dd7ce93b6adc1cefe42d383250f0f901545b1bc8975739c5e0a052d"


def unit_protocol():
    return gen.default_config("unittest")["protocol"]


_AD = {}


def adapter():
    if "a" not in _AD:
        _AD["a"] = gen.make_adapter()
    return _AD["a"]


def quiet(*a, **k):
    return None


def test_designs():
    assert gen.DESIGNS == ("panda", "unittest")
    f, ut = (gen.default_config(x)["protocol"] for x in gen.DESIGNS)
    assert f["namespace"] == "Code5.TSD.MTX-Panda-v1" and ut["namespace"] == UNIT_NS
    assert f["plan"] == {"posture_rad": 0.3, "amp_rad": 0.3}
    y = copy.deepcopy(ut)
    y["namespace"], y["parent_prefix"] = f["namespace"], f["parent_prefix"]
    assert y == f, "the unit-test design differs in more than the namespace and the parent prefix"


def test_model_pinned_and_identity():
    import mujoco
    info = E.verify_model_files()
    assert info["panda_xml_sha256"] == PANDA_XML_SHA and info["sha256sums_sha256"] == SHA256SUMS_SHA, info
    assert info["files_checked"] == 82, info
    ad = adapter()
    m = ad.model
    assert (m.nq, m.nv, m.nu) == (7, 7, 7) and abs(ad.dt - 0.05) < 1e-12 and ad.frame_skip == 25
    assert abs(m.opt.timestep - 0.002) < 1e-15 and int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert names == list(E.ARM_JOINTS), names
    assert m.neq == 0 and m.ntendon == 0 and m.nkey == 0
    assert np.array_equal(ad.lo, m.jnt_range[:, 0]) and np.array_equal(ad.hi, m.jnt_range[:, 1])
    assert np.allclose(m.actuator_gainprm[:, 0], E.PLANT["K"]) and np.allclose(-m.actuator_biasprm[:, 1], E.PLANT["K"])
    assert np.allclose(-m.actuator_biasprm[:, 2], E.PLANT["D"]) and not m.actuator_forcelimited.any()
    assert np.allclose(m.jnt_actfrcrange[:, 1], E.PLANT["torque_limit"]) and m.jnt_actfrclimited.all() and m.jnt_actgravcomp.all()
    assert np.allclose(m.dof_frictionloss, 0.5) and np.allclose(m.dof_damping, 1.0) and np.allclose(m.dof_armature, 0.1)
    gc = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): float(m.body_gravcomp[i]) for i in range(m.nbody)}
    assert all(gc[b] == 1.0 for b in E.GRAVCOMP_BODIES) and gc["payload"] == 0.0 and gc["link0"] == 0.0
    assert m.body_mass[ad.payload_id] == 0.0 and np.allclose(m.body_pos[ad.payload_id], [0, 0, 0.1034])
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    assert abs(ad.identity()["total_mass_nominal"] - 17.451901) < 1e-6
    ad.reset_to(ad.mid + 0.1)
    o = ad.step(np.zeros(7, np.float32))
    assert o.shape == (14,) and np.array_equal(o, np.concatenate([ad.data.qpos, ad.data.qvel]).astype(np.float32))
    assert gen.make_adapter().model_sha256 == ad.model_sha256


def test_condition_draws():
    p = unit_protocol()
    ds = gen.design_parents(p, {"train": 24, "val": 12, "test": 12})
    assert len(ds) == 48 and [d["index"] for d in ds] == list(range(48)) and "contact_regime" not in ds[0]
    md = p["min_condition_distance"]
    for d in ds:
        assert d["cohort_kinds"] == [p["action_kinds"][d["index"] % 4], p["action_kinds"][(d["index"] + 1) % 4]]
        for f, g, m in d["values"].values():
            assert 0.1 <= f <= 3.0 and (g == 1.0 or 0.8 <= g <= 1.2) and (m == 0.0 or 0.5 <= m <= 2.5)
            assert abs(math.log(f)) >= md["log_friction"] or abs(g - 1) >= md["gain"] or m >= md["payload"]
    assert ds == gen.design_parents(p, {"train": 24, "val": 12, "test": 12})
    assert ds != gen.design_parents(p, {"train": 24, "val": 12, "test": 12}, seed_offset=1)


def test_schedules_match_rules():
    p = unit_protocol()
    vals = {"A": (0.2, 1.15, 2.0), "B": (2.5, 0.85, 0.0), "C": (1.0, 1.0, 1.5)}
    seqs = gen.condition_sequences(vals, p)
    segs = p["segments"]
    N = (1.0, 1.0, 0.0)
    A = vals["A"]
    want = {"null": [N] * 5, "persistent": [N] + [A] * 4, "switch": [N, A, vals["B"], vals["C"], vals["C"]],
            "recur": [N, A, vals["B"], A, vals["B"]], "recur_nominal": [N, A, N, A, N],
            "drift": [N, A] + [(min(max(A[0] * math.exp(0.15 * k), 0.1), 3.0), min(max(A[1] + 0.02 * k, 0.7), 1.3),
                                min(max(A[2] + 0.25 * k, 0.0), 3.0)) for k in (1, 2, 3)]}
    for name, plan in want.items():
        fr, gn, pl, seg = seqs[name]
        for i, (lo, hi) in enumerate(zip(segs[:-1], segs[1:])):
            assert np.all(seg[lo:hi] == i)
            for arr, j in ((fr, 0), (gn, 1), (pl, 2)):
                assert np.allclose(arr[lo:hi], plan[i][j], rtol=0, atol=1e-15), (name, i, j)
    s2 = gen.condition_sequences({"A": (2.8, 1.2, 2.5), "B": N, "C": N}, p)
    assert s2["drift"][0].max() == 3.0 and s2["drift"][1].max() == 1.26 and s2["drift"][2].max() == 3.0
    d = gen.design_parents(p, {"train": 3})[2]
    for att in (0, 1, 2):
        lab, wrench, imps = gen.disturbance_labels(p, d, 1, att)
        for name in p["sequences"]:
            L = lab[name]
            assert np.array_equal(L["payload"], gen.condition_sequences(d["values"], p)[name][2])
            r = L["friction_eff"] / L["friction"]
            assert np.all((r >= math.exp(-1) - 1e-12) & (r <= math.e + 1e-12))
            g = L["gain_eff"] / L["gain"]
            assert np.all((g == 1.0) | ((g >= 0.875 - 1e-12) & (g <= 1.125 + 1e-12)))
        mag = np.linalg.norm(wrench, axis=1)
        assert np.all((mag == 0) | ((mag >= 5 - 1e-9) & (mag <= 25 + 1e-9))) and np.all(mag[:20] == 0)
        assert imps.shape == (p["T"], 3) and np.all(imps[:45] == 0)
        act = (mag > 0).astype(int) + (np.abs(lab["null"]["gain_eff"] - 1) > 0) + (np.abs(lab["null"]["friction_eff"] - 1) > 0)
        assert act.max() <= 1
    l0, w0, i0 = gen.disturbance_labels(p, d, 1, 0)
    l1, w1, i1 = gen.disturbance_labels(p, d, 1, 1)
    assert not (np.array_equal(i0, i1) and np.array_equal(w0, w1) and np.array_equal(l0["null"]["gain_eff"], l1["null"]["gain_eff"]))
    for name in p["sequences"]:
        for k in ("friction", "gain", "payload", "seg"):
            assert np.array_equal(l0[name][k], l1[name][k])


def test_action_plans():
    p = unit_protocol()
    half = adapter().half
    for kind in p["action_kinds"]:
        a, info = gen.action_plan(p, kind, 5, 0, 0, half)
        b, _ = gen.action_plan(p, kind, 5, 0, 0, half)
        c, _ = gen.action_plan(p, kind, 5, 0, 1, half)
        assert a.dtype == np.float32 and a.shape == (p["T"], 7) and np.array_equal(a, b) and not np.array_equal(a, c)
        q = a.astype(np.float64) * half[None, :]
        assert np.abs(q).max() <= 0.6 + 1e-6 and np.abs(a).max() <= 0.6 / half.min() + 1e-6 <= p["action_limit"]
        post = np.asarray(info["posture_rad"])
        for lo, hi in p["low_amplitude_windows"]:
            assert np.abs(q[lo:hi] - post[None, :]).max() <= 0.3 * 0.15 + 1e-6


def test_health_detection():
    ad = adapter()
    p = unit_protocol()
    ad.reset_to(ad.mid)
    a = np.zeros(7, np.float32)
    a[0] = 1.0
    for t in range(40):
        ad.step(a, None, t)
    assert ad.violation is not None and ad.violation["reason"] == "joint_limit", ad.violation
    ad.reset_to(ad.mid)
    a = np.zeros(7, np.float32)
    a[1] = 0.6
    for t in range(60):
        ad.step(a, np.array([0.0, 0.0, -400.0]), t)
    assert ad.violation is not None and ad.violation["reason"] in ("contact", "joint_limit"), ad.violation
    d = gen.design_parents(p, {"train": 1})[0]
    snap, _ = gen.discover_anchor(ad, p, d)
    lab, w, im = gen.disturbance_labels(p, d, 0, 0)
    acts, _ = gen.action_plan(p, d["cohort_kinds"][0], d["index"], 0, 0, ad.half)
    o, v = gen.rollout(ad, snap, acts, lab["null"]["friction_eff"], lab["null"]["gain_eff"], lab["null"]["payload"], w, im)
    assert v is None and o.shape == (301, 14)


def test_payload_edit_exact_and_reversible():
    import mujoco
    ad = adapter()
    m = ad.model
    fields = ("body_mass", "body_inertia", "body_subtreemass", "body_invweight0", "dof_invweight0", "actuator_acc0",
              "dof_M0", "actuator_length0")

    def snap():
        out = {f: getattr(m, f).copy() for f in fields if hasattr(m, f)}
        out["meaninertia"] = float(m.stat.meaninertia)
        out["meanmass"] = float(m.stat.meanmass)
        return out
    ad.reset_to(ad.mid + 0.05)
    ad.set_dynamics(1.0, 1.0, 0.0)
    base = snap()
    st0 = np.empty(ad.state_size)
    mujoco.mj_getState(m, ad.data, st0, ad.state_spec)
    ad.set_dynamics(1.0, 1.0, 2.0)
    st1 = np.empty_like(st0)
    mujoco.mj_getState(m, ad.data, st1, ad.state_spec)
    assert np.array_equal(st0, st1), "payload edit changed the integration state"
    assert m.body_mass[ad.payload_id] == 2.0 and np.allclose(m.body_inertia[ad.payload_id], 2.0 * 0.4 * 0.03 ** 2)
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    assert m.body_subtreemass[hand] > base["body_subtreemass"][hand] + 1.99
    ad.set_dynamics(1.0, 1.0, 0.0)
    after = snap()
    for k, v in base.items():
        assert np.array_equal(np.asarray(v), np.asarray(after[k])), f"{k} not restored bit-exactly"


def test_dynamics_edits_absolute():
    ad = adapter()
    m = ad.model
    ad.set_dynamics(1.0, 1.0, 0.0)
    base = {k: getattr(m, k).copy() for k in ad.base}
    ad.set_dynamics(2.5, 1.1, 1.0)
    assert np.allclose(m.actuator_gainprm[:, 0], 1.1 * np.asarray(E.PLANT["K"]), rtol=1e-15)
    assert np.allclose(-m.actuator_biasprm[:, 2], 1.1 * np.asarray(E.PLANT["D"]), rtol=1e-15)
    assert np.allclose(m.jnt_actfrcrange[:, 1], 1.1 * np.asarray(E.PLANT["torque_limit"]), rtol=1e-15)
    assert np.allclose(m.body_gravcomp[ad.gc_ids], 1.1) and m.body_gravcomp[ad.payload_id] == 0.0
    assert np.allclose(m.dof_frictionloss, 0.5 * 2.5) and np.allclose(m.dof_damping, 1.0 * 2.5)
    ad.set_dynamics(0.3, 0.9, 0.5)
    ad.set_dynamics(1.0, 1.0, 0.0)
    for k, v in base.items():
        assert np.array_equal(v, getattr(m, k)), f"{k} not restored exactly"
    for bad in ((0.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, -0.1), (float("nan"), 1.0, 0.0)):
        try:
            ad.set_dynamics(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad}")
    ad.set_dynamics(1.0, 1.0, 0.0)


def _one_cohort(p, ad, idx=1, cohort=0, att=0):
    d = gen.design_parents(p, {"train": idx + 1})[idx]
    snap, anc = gen.discover_anchor(ad, p, d)
    labels, wrench, imps = gen.disturbance_labels(p, d, cohort, att)
    acts, _ = gen.action_plan(p, d["cohort_kinds"][cohort], d["index"], cohort, att, ad.half)
    return d, snap, anc, labels, wrench, imps, acts


def _roll(ad, snap, acts, L, w, im):
    return gen.rollout(ad, snap, acts, L["friction_eff"], L["gain_eff"], L["payload"], w, im)


def test_replay_and_order_independence():
    p = unit_protocol()
    ad = adapter()
    d, snap, anc, labels, w, im, acts = _one_cohort(p, ad)
    a1, v1 = _roll(ad, snap, acts, labels["drift"], w, im)
    for name in ("switch", "recur", "null", "recur_nominal"):
        _roll(ad, snap, acts, labels[name], w, im)
    a2, _ = _roll(ad, snap, acts, labels["drift"], w, im)
    assert v1 is None and np.array_equal(a1, a2), "rollout depends on what ran before"
    fresh = gen.make_adapter()
    snap2, anc2 = gen.discover_anchor(fresh, p, d)
    assert snap2["state_sha256"] == snap["state_sha256"] and anc2 == anc
    a3, _ = _roll(fresh, snap2, acts, labels["drift"], w, im)
    assert np.array_equal(a1, a3), "fresh adapter differs"
    n, _ = _roll(ad, snap, acts, labels["null"], w, im)
    assert np.array_equal(n[:41], a1[:41]) and not np.array_equal(n[41:], a1[41:])


def test_no_label_leakage_in_obs():
    p = unit_protocol()
    ad = adapter()
    d, snap, anc, labels, w, im, acts = _one_cohort(p, ad, idx=2)
    L = labels["persistent"]
    o1, _ = _roll(ad, snap, acts, L, w, im)
    L2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in L.items()}
    o2, _ = _roll(ad, snap, acts, L2, w.copy(), im.copy())
    assert np.array_equal(o1, o2) and o1.shape == (301, 14) and np.array_equal(o1[0], snap["obs"])


def test_collect_end_to_end_and_determinism():
    p = unit_protocol()
    cfg = {"design": "unittest", "protocol": p}
    with tempfile.TemporaryDirectory() as tmp:
        m1 = gen.collect(Path(tmp) / "a", copy.deepcopy(cfg), counts={"train": 2, "test": 1}, log=quiet)
        m2 = gen.collect(Path(tmp) / "b", copy.deepcopy(cfg), counts={"train": 2, "test": 1}, log=quiet)
        for s in ("train", "test"):
            assert m1["files"][s]["array_sha256"] == m2["files"][s]["array_sha256"], s
        z = np.load(Path(tmp) / "a" / "panda_train.npz")
        assert set(z.files) == set(gen.ARRAY_KEYS)
        assert z["obs"].shape == (24, 301, 14) and z["act"].shape == (24, 300, 7) and z["wrench"].shape == (24, 300, 3)
        assert all(r["exact"] for r in m1["replay_audit"])
        from panda.stats import audit_dir
        res = audit_dir(Path(tmp) / "a", replay_all=True)
        tr = res["splits"]["train"]
        for k in ("shapes_ok", "pairing_ok", "pre_switch_identical_ok", "finite_ok", "action_range_ok", "joint_range_ok",
                  "health_ok", "labels_recomputed_ok"):
            assert tr[k], (k, tr)
        assert tr["full_replay"]["all_ok"] and tr["full_replay"]["ok"] == 24
        assert res["replay_audit_ok"] and res["disjoint_ok"]


def _always_bad(orig):
    def f(adapter_, snap, actions, friction, gain, payload, wrench, impulses):
        o, _ = orig(adapter_, snap, actions, friction, gain, payload, wrench, impulses)
        return o, {"t": 0, "reason": "contact", "bodies": ["test", "test"]}
    return f


def test_rejection_path():
    p = unit_protocol()
    p["max_plan_attempts"] = 2
    p["max_anchor_redraws"] = 0
    orig = gen.rollout
    gen.rollout = _always_bad(orig)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                gen.collect(Path(tmp) / "a", {"design": "unittest", "protocol": p}, counts={"train": 1}, log=quiet)
            except RuntimeError as e:
                assert "plan attempts exhausted" in str(e) and "under 1 anchor(s)" in str(e), str(e)
            else:
                raise AssertionError("expected exhaustion")
    finally:
        gen.rollout = orig


def test_anchor_redraw_exhaustion():
    p = unit_protocol()
    p["max_plan_attempts"] = 1
    p["max_anchor_redraws"] = 2
    logged = []
    orig = gen.rollout
    gen.rollout = _always_bad(orig)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                gen.collect(Path(tmp) / "a", {"design": "unittest", "protocol": p}, counts={"train": 1},
                            log=lambda stage, **k: logged.append((stage, k)))
            except RuntimeError as e:
                assert "plan attempts exhausted" in str(e) and "under 3 anchor(s)" in str(e), str(e)
            else:
                raise AssertionError("expected exhaustion")
    finally:
        gen.rollout = orig
    anchors = [k for st, k in logged if st == "panda_anchor_reject"]
    att = [a["anchor_attempt"] for a in anchors]
    assert len(anchors) == 3 and att == sorted(att) and len(set(att)) == 3, anchors
    rej = [k for st, k in logged if st == "panda_reject"]
    assert len(rej) == 3 and [r["anchor_attempt"] for r in rej] == att and all(r["cohort"] == 0 for r in rej)


def test_anchor_redraw_accepts():
    p = unit_protocol()
    p["max_plan_attempts"] = 2
    p["max_anchor_redraws"] = 3
    ad = adapter()
    d = gen.design_parents(p, {"train": 1})[0]
    snap0, anchor0 = gen.discover_anchor(ad, p, d)
    first_obs = snap0["obs"].copy()
    orig = gen.rollout

    def patched(adapter_, snap, *a):
        o, v = orig(adapter_, snap, *a)
        if np.array_equal(snap["obs"], first_obs):
            return o, {"t": 0, "reason": "contact", "bodies": ["test", "test"]}
        return o, v
    gen.rollout = patched
    try:
        with tempfile.TemporaryDirectory() as tmp:
            man = gen.collect(Path(tmp) / "a", {"design": "unittest", "protocol": p}, counts={"train": 1}, log=quiet)
            gen.rollout = orig
            par = man["parents"][0]
            assert len(par["anchor_rejections"]) == 1 and man["anchor_rejections"] == par["anchor_rejections"]
            ar = par["anchor_rejections"][0]
            assert ar["anchor_attempt"] == anchor0["attempt"] and ar["state_sha256"] == anchor0["state_sha256"]
            assert ar["exhausted_cohort"] == 0 and par["anchor"]["attempt"] > anchor0["attempt"]
            old = [r for r in man["rejections"] if r["anchor_attempt"] == anchor0["attempt"]]
            assert len(old) == 2 and all(r["cohort"] == 0 for r in old)
            z = np.load(Path(tmp) / "a" / "panda_train.npz")
            assert z["obs"].shape == (12, 301, 14) and not any(np.array_equal(o[0], first_obs) for o in z["obs"])
            from panda.stats import audit_dir
            res = audit_dir(Path(tmp) / "a", replay_all=True)
            tr = res["splits"]["train"]
            assert tr["labels_recomputed_ok"] and tr["health_ok"] and tr["redraw"]["anchor_redraws"] == 1
            assert tr["redraw"]["fraction"] == 1.0
    finally:
        gen.rollout = orig


def test_anchor_redraw_no_effect_without_exhaustion():
    p0 = unit_protocol()
    p0["max_anchor_redraws"] = 0
    p1 = unit_protocol()
    p1["max_anchor_redraws"] = 10
    with tempfile.TemporaryDirectory() as tmp:
        m0 = gen.collect(Path(tmp) / "a", {"design": "unittest", "protocol": p0}, counts={"train": 1, "test": 1}, log=quiet)
        m1 = gen.collect(Path(tmp) / "b", {"design": "unittest", "protocol": p1}, counts={"train": 1, "test": 1}, log=quiet)
    assert not m1["anchor_rejections"]
    for s in ("train", "test"):
        assert m0["files"][s]["array_sha256"] == m1["files"][s]["array_sha256"], s
    assert m0["rejections"] == m1["rejections"] and [x["anchor"] for x in m0["parents"]] == [x["anchor"] for x in m1["parents"]]


def test_cli_smoke():
    from panda import cli
    orig = gen.default_config

    def fake(design="panda"):
        cfg = orig("unittest")
        cfg["design"] = design
        return cfg
    gen.default_config = fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            assert cli.main(["collect", "--design", "panda", "--out", str(t / "main"), "--counts", "train=1,val=1,test=1"]) == 0
            assert cli.main(["collect", "--design", "panda", "--out", str(t / "tb"), "--counts", "test=1", "--seed-offset", "1"]) == 0
            cfg = json.loads((t / "main" / "CONFIG.json").read_text())
            assert "panda/gen.py" in cfg["source_sha256"] and "simcore/gen.py" in cfg["source_sha256"]
            assert "panda_model/mujoco_menagerie_6da8b971/SHA256SUMS.txt" in cfg["source_sha256"]
            assert cli.main(["assemble", "--main", str(t / "main"), "--testb", str(t / "tb"), "--out", str(t / "data")]) == 0
            for f in ("panda_train.npz", "panda_val.npz", "panda_test.npz", "panda_testB.npz", "SHA256SUMS.txt",
                      "DATASET_CARD.json", "PANDA_MANIFEST.json", "PANDA_MANIFEST_testB.json"):
                assert (t / "data" / f).exists(), f
            rc = cli.main(["audit", "--main", str(t / "main" / "data"), "--testb", str(t / "tb" / "data"),
                           "--out", str(t / "audit.json")])
            res = json.loads((t / "audit.json").read_text())
            assert res["hard_checks_ok"] and res["disjoint_ok"] and res["full_replay_done"], res
            assert rc in (0, 4)
            assert cli.main(["compare", "--a", str(t / "data" / "panda_test.npz"), "--b", str(t / "main" / "data" / "panda_test.npz")]) == 0
            assert cli.main(["hashes", "--data", str(t / "main" / "data"), "--out", str(t / "h1.json")]) == 0
            assert cli.main(["collect", "--design", "panda", "--out", str(t / "main2"), "--counts", "train=1,val=1,test=1"]) == 0
            assert cli.main(["hashes", "--data", str(t / "main2" / "data"), "--out", str(t / "h2.json")]) == 0
            assert cli.main(["xcheck", "--a", str(t / "h1.json"), "--b", str(t / "h2.json"), "--out", str(t / "x12.json")]) == 0
            assert cli.main(["hashes", "--data", str(t / "tb" / "data"), "--out", str(t / "h3.json")]) == 0
            assert cli.main(["xcheck", "--a", str(t / "h1.json"), "--b", str(t / "h3.json"), "--out", str(t / "x13.json")]) == 7
            assert cli.main(["compare", "--a", str(t / "data" / "panda_test.npz"), "--b", str(t / "data" / "panda_testB.npz")]) == 3
            for f in ("main", "data"):
                try:
                    cli.main(["collect", "--design", "panda", "--out", str(t / f), "--counts", "train=1"])
                except FileExistsError:
                    pass
                else:
                    raise AssertionError("overwrote an existing output")
    finally:
        gen.default_config = orig


TESTS = [test_designs, test_model_pinned_and_identity, test_condition_draws,
         test_schedules_match_rules, test_action_plans, test_health_detection,
         test_payload_edit_exact_and_reversible, test_dynamics_edits_absolute, test_replay_and_order_independence,
         test_no_label_leakage_in_obs, test_collect_end_to_end_and_determinism, test_rejection_path,
         test_anchor_redraw_exhaustion, test_anchor_redraw_accepts, test_anchor_redraw_no_effect_without_exhaustion,
         test_cli_smoke]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tests = [t for t in TESTS if not ("--quick" in argv and t.__name__ in ("test_collect_end_to_end_and_determinism",
                                                                           "test_cli_smoke"))]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}\n{traceback.format_exc()}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
