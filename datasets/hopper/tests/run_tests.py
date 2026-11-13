from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from hopper import gen

PROBE_NS = gen.PROTOCOL_HOPPER["namespace"].rsplit("-", 1)[0] + "-unittest"


def probe_protocol():
    p = gen.default_config("hopper")["protocol"]
    p["namespace"] = PROBE_NS
    p["parent_prefix"] = "hoppertest"
    return p


_ADAPTER = {}


def adapter_for(p):
    if "a" not in _ADAPTER:
        _ADAPTER["a"] = gen.make_adapter(p)[0]
    return _ADAPTER["a"]


def _ctrl(ad):
    return gen.ctrl_range(ad)


def test_model_files():
    import gymnasium
    from hopper import env
    assert env.check_model_file() == env.MODEL_SHA256
    installed = Path(gymnasium.__file__).resolve().parent / "envs" / "mujoco" / "assets" / "hopper.xml"
    assert hashlib.sha256(installed.read_bytes()).hexdigest() == env.MODEL_SHA256, "differs from the installed Hopper-v5 file"
    assert gymnasium.__version__ == "1.3.0"


def test_env_identity():
    import gymnasium as gym
    import mujoco
    from hopper import env
    p = probe_protocol()
    ad = adapter_for(p)
    ident = ad.env_identity()
    assert ident["obs_dim"] == 11 and ident["action_dim"] == 3, ident
    assert abs(ident["dt"] - 0.05) < 1e-12 and ident["frame_skip"] == 25 and abs(ident["timestep"] - 0.002) < 1e-15
    assert ident["kwargs"] == {"frame_skip": 25, "reset_noise_scale": 0.005, "servo_kp": 800.0, "servo_kv": 60.0,
                               "home": [-0.1, -0.1, 0.17], "cone": "elliptic"}, ident["kwargs"]
    assert ident["class"] == "hopper.env.ServoHopperEnv", ident["class"]
    assert gym.registry[p["env_id"]].entry_point == gen.ENV_ENTRY_POINT
    assert abs(ident["total_mass"] - 15.820013405927003) < 1e-9 and abs(ident["torso_mass"] - 3.6651914291880923) < 1e-12
    assert ident["actuator_kp"] == [800.0] * 3 and ident["actuator_kv"] == [60.0] * 3
    assert ident["actuator_forcerange"] == [[-200.0, 200.0]] * 3
    rad = math.pi / 180.0
    assert np.allclose(ident["actuator_ctrlrange"], [[-150 * rad, 0.0], [-150 * rad, 0.0], [-45 * rad, 45 * rad]], rtol=0, atol=1e-12)
    m = ad._model
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert np.array_equal(m.actuator_gear[:, 0], [1.0, 1.0, 1.0]) and np.all(m.actuator_biasprm[:, 1] == -800.0)
    raw = mujoco.MjModel.from_xml_path(env.MODEL_XML)
    assert np.array_equal(raw.actuator_gear[:, 0], [200.0] * 3) and raw.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    assert np.array_equal(raw.body_mass, m.body_mass) and np.array_equal(raw.geom_friction, m.geom_friction)
    assert np.array_equal(raw.jnt_range, m.jnt_range) and np.array_equal(raw.dof_damping, m.dof_damping)
    ad.reset(123)
    u = ad._env.unwrapped
    rawobs = u._get_obs()
    expect = np.concatenate([u.data.qpos.flat[1:], u.data.qvel.flat]).astype(np.float64)
    assert rawobs.shape == (11,) and np.array_equal(np.asarray(rawobs, np.float64), expect)
    assert np.array_equal(u.data.ctrl, u.home_ctrl) and np.allclose(u.home_ctrl, gen.home_pose(p), rtol=0, atol=0)
    assert np.max(np.abs(u.data.qpos - [0.0, 1.25, 0.0, 0.0, 0.0, 0.0])) <= 0.005
    assert "HalfCheetah-v5" in gym.registry and "Hopper-v5" in gym.registry
    lo, hi = _ctrl(ad)
    a = gen.to_model_action(gen.home_pose(p), lo, hi).astype(np.float32)
    _, env_a = ad._prepare_model_action(a)
    assert np.max(np.abs(env_a - gen.home_pose(p))) < 1e-6


def test_condition_draws():
    p = probe_protocol()
    ds = gen.design_parents(p, {"train": 24, "val": 12, "test": 12})
    assert len(ds) == 48 and [d["index"] for d in ds] == list(range(48))
    reg = [d["contact_regime"] for d in ds]
    assert reg.count("flat") == round(0.6 * 48) and reg.count("transition") == round(0.2 * 48)
    assert reg.count("rocked") == 48 - reg.count("flat") - reg.count("transition")
    md = p["min_condition_distance"]
    for d in ds:
        assert d["cohort_kinds"] == [p["action_kinds"][d["index"] % 4], p["action_kinds"][(d["index"] + 1) % 4]]
        for f, g, m in d["values"].values():
            assert 0.1 <= f <= 3.0 and (g == 1.0 or 0.6 <= g <= 1.4) and (m == 1.0 or 1.3 <= m <= 2.5)
            assert abs(math.log(f)) >= md["log_friction"] or abs(g - 1) >= md["gain"] or abs(math.log(m)) >= md["log_payload"]
    assert ds == gen.design_parents(p, {"train": 24, "val": 12, "test": 12})
    assert ds != gen.design_parents(p, {"train": 24, "val": 12, "test": 12}, seed_offset=1)


def test_schedules_match_rules():
    p = probe_protocol()
    vals = {"A": (0.2, 1.3, 1.8), "B": (2.5, 0.7, 1.0), "C": (1.0, 1.0, 1.5)}
    seqs = gen.condition_sequences(vals, p)
    segs = p["segments"]
    N = (1.0, 1.0, 1.0)
    A = vals["A"]
    want = {"null": [N] * 5, "persistent": [N] + [A] * 4, "switch": [N, A, vals["B"], vals["C"], vals["C"]],
            "recur": [N, A, vals["B"], A, vals["B"]], "recur_nominal": [N, A, N, A, N]}
    want["drift"] = [N, A] + [(min(max(A[0] * math.exp(0.15 * k), 0.1), 3.0), min(max(A[1] + 0.04 * k, 0.5), 1.5),
                               min(max(A[2] * math.exp(0.1 * k), 1.0), 3.0)) for k in (1, 2, 3)]
    for name, plan in want.items():
        fr, gn, pl, seg = seqs[name]
        for i, (lo, hi) in enumerate(zip(segs[:-1], segs[1:])):
            f, g, m = plan[i]
            assert np.all(seg[lo:hi] == i)
            assert np.allclose(fr[lo:hi], f, rtol=0, atol=1e-15), (name, i)
            assert np.allclose(gn[lo:hi], g, rtol=0, atol=1e-15), (name, i)
            assert np.allclose(pl[lo:hi], m, rtol=0, atol=1e-15), (name, i)
    seqs2 = gen.condition_sequences({"A": (2.8, 1.0, 2.5), "B": N, "C": N}, p)
    assert seqs2["drift"][2].max() <= 3.0 and seqs2["drift"][0].max() == 3.0
    assert np.allclose(seqs2["drift"][0][120:190], min(2.8 * math.exp(0.15), 3.0), rtol=0, atol=1e-15)
    d = gen.design_parents(p, {"train": 3})[2]
    for att in (0, 1):
        lab, imps = gen.disturbance_labels(p, d, 1, att)
        for name in p["sequences"]:
            L = lab[name]
            assert np.array_equal(L["payload"], gen.condition_sequences(d["values"], p)[name][2])
            assert L["friction_eff"].max() <= 3.0 + 1e-12
            assert np.all((L["gain_eff"] == L["gain"]) | (np.abs(L["gain_eff"] / L["gain"] - 1) <= 0.25 + 1e-12))
        assert imps.shape == (p["T"], 3) and np.all(imps[:45] == 0)
    l0, i0 = gen.disturbance_labels(p, d, 1, 0)
    l1, i1 = gen.disturbance_labels(p, d, 1, 1)
    assert not np.array_equal(i0, i1) or not np.array_equal(l0["null"]["gain_eff"], l1["null"]["gain_eff"])
    for name in p["sequences"]:
        for k in ("friction", "gain", "payload", "seg"):
            assert np.array_equal(l0[name][k], l1[name][k])


def test_action_plans():
    p = probe_protocol()
    ad = adapter_for(p)
    lo, hi = _ctrl(ad)
    box = np.asarray(p["offset_box_rad"])
    home = gen.home_pose(p)
    bound = np.asarray(p["bounce"]["amp_rad"]) + np.asarray(p["perturbation_rad"])
    for kind in p["action_kinds"]:
        a, info = gen.action_plan(p, kind, 5, 0, 0, lo, hi)
        b, _ = gen.action_plan(p, kind, 5, 0, 0, lo, hi)
        c, _ = gen.action_plan(p, kind, 5, 0, 1, lo, hi)
        assert a.dtype == np.float32 and a.shape == (p["T"], 3)
        assert np.array_equal(a, b) and not np.array_equal(a, c)
        assert np.abs(a).max() <= 1.0
        q = lo + 0.5 * (a.astype(np.float64) + 1.0) * (hi - lo)
        off = q - home
        assert np.all(np.abs(off) <= box + 1e-5)
        assert np.all(np.abs(off) <= bound + 1e-5), "offset exceeds bounce + perturbation bound"
        for lo_, hi_ in p["low_amplitude_windows"]:
            assert np.all(np.abs(off[lo_:hi_]) <= bound * p["low_amplitude_multiplier"] + 1e-5)
        assert 0.5 <= info["bounce"]["freq_hz"] <= 2.0
        assert all(0.5 * x - 1e-12 <= y <= x + 1e-12 for x, y in zip(p["bounce"]["amp_rad"], info["bounce"]["amp_rad"]))


def test_health_rule():
    p = probe_protocol()
    obs = np.zeros((301, 11), np.float32)
    obs[:, 0] = 1.2
    obs[:, 1] = 0.05
    assert gen.health_check(obs, p) is None
    o = obs.copy(); o[100, 0] = 0.7
    assert gen.health_check(o, p)[:2] == (100, "z_low")
    o = obs.copy(); o[150, 0] = 0.7001
    assert gen.health_check(o, p) is None
    o = obs.copy(); o[200, 1] = 0.2
    assert gen.health_check(o, p)[:2] == (200, "angle")
    o = obs.copy(); o[200, 1] = -0.2001
    assert gen.health_check(o, p)[:2] == (200, "angle")
    o = obs.copy(); o[50, 1] = 0.1999
    assert gen.health_check(o, p) is None
    o = obs.copy(); o[60, 7] = 100.0
    assert gen.health_check(o, p)[:2] == (60, "state")
    o = obs.copy(); o[60, 7] = -99.9
    assert gen.health_check(o, p) is None
    assert gen.health_check(obs[:1], p) is None


def test_strength_semantics():
    p = probe_protocol()
    ad = adapter_for(p)
    m = ad._model
    base = (m.actuator_gainprm.copy(), m.actuator_biasprm.copy(), m.actuator_forcerange.copy(), m.actuator_gear.copy())
    ad.reset(11)
    ad.apply_absolute_dynamics({"gain_scale": 0.7})
    assert np.array_equal(m.actuator_gainprm[:, 0], base[0][:, 0] * 0.7)
    assert np.array_equal(m.actuator_biasprm[:, 1], base[1][:, 1] * 0.7)
    assert np.array_equal(m.actuator_forcerange, base[2] * 0.7)
    assert np.array_equal(m.actuator_gear, base[3]), "gear must not change (it would move the set points)"
    ad.reset(11)
    ad._data.ctrl[:] = gen.home_pose(p) - 0.05
    ad.apply_absolute_dynamics({"gain_scale": 1.0})
    f1 = ad._data.actuator_force.copy()
    ad.apply_absolute_dynamics({"gain_scale": 0.7})
    f2 = ad._data.actuator_force.copy()
    assert np.allclose(f2, 0.7 * f1, rtol=1e-12, atol=1e-12)
    ad.apply_absolute_dynamics({"gain_scale": 1.0})
    assert np.array_equal(m.actuator_gainprm, base[0]) and np.array_equal(m.actuator_forcerange, base[2])


def test_payload_edit_exact_and_reversible():
    import mujoco
    p = probe_protocol()
    ad = adapter_for(p)
    m = ad._model
    fields = ("body_mass", "body_inertia", "body_ipos", "body_subtreemass", "body_invweight0", "dof_invweight0",
              "actuator_acc0", "dof_M0", "actuator_length0")

    def snap():
        out = {f: getattr(m, f).copy() for f in fields if hasattr(m, f)}
        out["meaninertia"] = float(m.stat.meaninertia)
        out["meanmass"] = float(m.stat.meanmass)
        return out
    ad.reset(7)
    ad.apply_absolute_dynamics(dict(gen.NOMINAL))
    base = snap()
    st0 = np.empty(ad._integration_state_size); mujoco.mj_getState(m, ad._data, st0, mujoco.mjtState.mjSTATE_INTEGRATION)
    ad.apply_absolute_dynamics({"payload_scale": 1.8})
    st1 = np.empty_like(st0); mujoco.mj_getState(m, ad._data, st1, mujoco.mjtState.mjSTATE_INTEGRATION)
    assert np.array_equal(st0, st1), "payload edit changed the integration state"
    t = ad._torso
    assert m.body_mass[t] == base["body_mass"][t] * 1.8
    assert np.array_equal(m.body_inertia[t], base["body_inertia"][t] * 1.8)
    assert np.array_equal(m.body_ipos, base["body_ipos"])
    assert m.body_subtreemass[t] > base["body_subtreemass"][t]
    ad.apply_absolute_dynamics(dict(gen.NOMINAL))
    after = snap()
    for k, v in base.items():
        assert np.array_equal(np.asarray(v), np.asarray(after[k])), f"{k} not restored bit-exactly"
    assert ad.dynamics_now()["payload_scale"] == 1.0


def test_setconst_canonical():
    import mujoco
    from hopper import env
    p = probe_protocol()
    ad = adapter_for(p)
    m0 = mujoco.MjModel.from_xml_path(env.MODEL_XML)
    env.convert_to_servos(m0, 800.0, 60.0)
    m0.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    d0 = mujoco.MjData(m0)
    mujoco.mj_setConst(m0, d0)
    ad.apply_absolute_dynamics(dict(gen.NOMINAL))
    for f in ("body_subtreemass", "dof_invweight0", "body_invweight0", "actuator_acc0", "dof_M0"):
        assert np.array_equal(getattr(m0, f), getattr(ad._model, f)), f


def test_regimes_and_anchor_health():
    p = probe_protocol()
    assert gen.classify_regime((((0, 4),), 1), None) == "rocked"
    assert gen.classify_regime(((), 0), None) == "rocked"
    assert gen.classify_regime((((0, 4), (0, 4)), 2), None) == "flat"
    assert gen.classify_regime((((0, 4), (0, 4)), 2), (((0, 4),), 1)) == "transition"
    assert gen.classify_regime((((0, 4), (0, 4)), 2), (((0, 4), (0, 4)), 2)) == "flat"
    ad = adapter_for(p)
    for d in gen.design_parents(p, {"train": 5}):
        snap, anc = gen.discover_anchor(ad, p, d)
        assert gen.health_check(snap.flat_observation[None, :], p) is None
        assert p["anchor_step_min"] <= anc["anchor_step"] <= p["anchor_step_max"]


def _one_cohort(p, ad, idx=1, cohort=0, att=0):
    d = gen.design_parents(p, {"train": idx + 1})[idx]
    snap, _ = gen.discover_anchor(ad, p, d)
    labels, imps = gen.disturbance_labels(p, d, cohort, att)
    lo, hi = _ctrl(ad)
    acts, _ = gen.action_plan(p, d["cohort_kinds"][cohort], d["index"], cohort, att, lo, hi)
    return d, snap, labels, imps, acts


def _roll(ad, snap, acts, L, imps):
    return gen.rollout(ad, snap, acts, L["friction_eff"], L["gain_eff"], L["payload"], imps)


def test_replay_and_order_independence():
    p = probe_protocol()
    ad = adapter_for(p)
    d, snap, labels, imps, acts = _one_cohort(p, ad)
    a1 = _roll(ad, snap, acts, labels["drift"], imps)
    for name in ("switch", "recur", "null"):
        _roll(ad, snap, acts, labels[name], imps)
    a2 = _roll(ad, snap, acts, labels["drift"], imps)
    assert np.array_equal(a1, a2), "rollout depends on what ran before"
    fresh = gen.make_adapter(p)[0]
    snap2, _ = gen.discover_anchor(fresh, p, d)
    assert snap2.state_sha256 == snap.state_sha256
    a3 = _roll(fresh, snap2, acts, labels["drift"], imps)
    assert np.array_equal(a1, a3), "fresh adapter differs"
    n = _roll(ad, snap, acts, labels["null"], imps)
    assert np.array_equal(n[:41], a1[:41]) and not np.array_equal(n[41:], a1[41:])


def test_no_label_leakage_in_obs():
    p = probe_protocol()
    ad = adapter_for(p)
    d, snap, labels, imps, acts = _one_cohort(p, ad, idx=2)
    L = labels["persistent"]
    o1 = _roll(ad, snap, acts, L, imps)
    L2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in L.items()}
    o2 = _roll(ad, snap, acts, L2, imps.copy())
    assert np.array_equal(o1, o2) and o1.shape == (301, 11)
    assert np.array_equal(o1[0], snap.flat_observation)


def test_collect_end_to_end_and_determinism():
    p = probe_protocol()
    cfg = {"design": "unittest", "protocol": p}
    with tempfile.TemporaryDirectory() as tmp:
        m1 = gen.collect(Path(tmp) / "a", copy.deepcopy(cfg), counts={"train": 2, "test": 1}, log=lambda *a, **k: None)
        m2 = gen.collect(Path(tmp) / "b", copy.deepcopy(cfg), counts={"train": 2, "test": 1}, log=lambda *a, **k: None)
        for s in ("train", "test"):
            assert m1["files"][s]["array_sha256"] == m2["files"][s]["array_sha256"], s
        z = np.load(Path(tmp) / "a" / "hopper_train.npz")
        assert set(z.files) == set(gen.ARRAY_KEYS)
        assert z["obs"].shape == (24, 301, 11) and z["act"].shape == (24, 300, 3) and z["payload"].shape == (24, 300)
        assert all(r["exact"] for r in m1["replay_audit"])
        from hopper.stats import audit_dir
        res = audit_dir(Path(tmp) / "a")
        tr = res["splits"]["train"]
        for k in ("shapes_ok", "pairing_ok", "pre_switch_identical_ok", "finite_ok", "action_range_ok", "health_ok",
                  "friction_cap_ok", "labels_recomputed_ok"):
            assert tr[k], (k, tr)
        assert res["replay_audit_ok"] and res["disjoint_ok"]


def test_rejection_path():
    p = probe_protocol()
    p["health"] = {"z_min": 1.5, "angle_max": 0.2, "state_abs_max": 100.0}
    p["max_plan_attempts"] = 2
    p["max_anchor_redraws"] = 0
    p["anchor_step_min"] = 16
    cfg = {"design": "unittest", "protocol": p}
    orig = gen.health_check
    def patched(obs, protocol):
        if obs.shape[0] == 1:
            return orig(obs, {**protocol, "health": {"z_min": 0.7, "angle_max": 0.2, "state_abs_max": 100.0}})
        return orig(obs, protocol)
    gen.health_check = patched
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                gen.collect(Path(tmp) / "a", cfg, counts={"train": 1}, log=lambda *a, **k: None)
            except RuntimeError as e:
                assert "plan attempts exhausted" in str(e)
            else:
                raise AssertionError("expected exhaustion")
    finally:
        gen.health_check = orig


def test_anchor_redraw_exhaustion():
    p = probe_protocol()
    p["health"] = {"z_min": 1.5, "angle_max": 0.2, "state_abs_max": 100.0}
    p["max_plan_attempts"] = 1
    p["max_anchor_redraws"] = 2
    cfg = {"design": "unittest", "protocol": p}
    logged = []
    orig = gen.health_check

    def patched(obs, protocol):
        if obs.shape[0] == 1:
            return orig(obs, {**protocol, "health": {"z_min": 0.7, "angle_max": 0.2, "state_abs_max": 100.0}})
        return orig(obs, protocol)
    gen.health_check = patched
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                gen.collect(Path(tmp) / "a", cfg, counts={"train": 1}, log=lambda stage, **k: logged.append((stage, k)))
            except RuntimeError as e:
                assert "plan attempts exhausted" in str(e) and "under 3 anchor(s)" in str(e), str(e)
            else:
                raise AssertionError("expected exhaustion")
    finally:
        gen.health_check = orig
    anchors = [k for st, k in logged if st == "hopper_anchor_reject"]
    assert len(anchors) == 3, anchors
    att = [a["anchor_attempt"] for a in anchors]
    assert att == sorted(att) and len(set(att)) == 3, att
    rej = [k for st, k in logged if st == "hopper_reject"]
    assert len(rej) == 3 and [r["anchor_attempt"] for r in rej] == att and all(r["cohort"] == 0 for r in rej)


def test_anchor_redraw_accepts():
    p = probe_protocol()
    p["max_plan_attempts"] = 2
    p["max_anchor_redraws"] = 3
    ad = adapter_for(p)
    d = gen.design_parents(p, {"train": 1})[0]
    snap0, anchor0 = gen.discover_anchor(ad, p, d)
    first_obs = snap0.flat_observation.copy()
    orig = gen.health_check

    def patched(obs, protocol):
        if obs.shape[0] > 1 and np.array_equal(obs[0], first_obs):
            return (0, "angle", 0.0)
        return orig(obs, protocol)
    cfg = {"design": "unittest", "protocol": p}
    gen.health_check = patched
    try:
        with tempfile.TemporaryDirectory() as tmp:
            man = gen.collect(Path(tmp) / "a", cfg, counts={"train": 1}, log=lambda *a, **k: None)
            gen.health_check = orig
            par = man["parents"][0]
            assert len(par["anchor_rejections"]) == 1 and man["anchor_rejections"] == par["anchor_rejections"]
            ar = par["anchor_rejections"][0]
            assert ar["anchor_attempt"] == anchor0["attempt"] and ar["state_sha256"] == anchor0["state_sha256"]
            assert ar["exhausted_cohort"] == 0 and ar["plan_attempts"] == 2
            assert par["anchor"]["attempt"] > anchor0["attempt"] and par["anchor"]["state_sha256"] != anchor0["state_sha256"]
            old = [r for r in man["rejections"] if r["anchor_attempt"] == anchor0["attempt"]]
            assert len(old) == 2 and all(r["cohort"] == 0 for r in old)
            z = np.load(Path(tmp) / "a" / "hopper_train.npz")
            assert z["obs"].shape == (12, 301, 11)
            assert not any(np.array_equal(o[0], first_obs) for o in z["obs"])
            assert all(r["exact"] for r in man["replay_audit"])
            from hopper.stats import audit_dir
            res = audit_dir(Path(tmp) / "a")
            tr = res["splits"]["train"]
            for k in ("shapes_ok", "pairing_ok", "pre_switch_identical_ok", "finite_ok", "action_range_ok",
                      "health_ok", "friction_cap_ok", "labels_recomputed_ok"):
                assert tr[k], (k, tr)
            assert tr["redraw"]["anchor_redraws"] == 1 and tr["redraw"]["fraction"] == 1.0, tr["redraw"]
    finally:
        gen.health_check = orig


def test_anchor_redraw_no_effect_without_exhaustion():
    p0 = probe_protocol(); p0["max_anchor_redraws"] = 0
    p1 = probe_protocol(); p1["max_anchor_redraws"] = 10
    with tempfile.TemporaryDirectory() as tmp:
        m0 = gen.collect(Path(tmp) / "a", {"design": "unittest", "protocol": p0}, counts={"train": 1, "test": 1},
                         log=lambda *a, **k: None)
        m1 = gen.collect(Path(tmp) / "b", {"design": "unittest", "protocol": p1}, counts={"train": 1, "test": 1},
                         log=lambda *a, **k: None)
    if not m0["anchor_rejections"]:
        assert not m1["anchor_rejections"]
        for s in ("train", "test"):
            assert m0["files"][s]["array_sha256"] == m1["files"][s]["array_sha256"], s
        assert m0["rejections"] == m1["rejections"]
        assert [x["anchor"] for x in m0["parents"]] == [x["anchor"] for x in m1["parents"]]


def test_designs():
    assert gen.DESIGNS == ("hopper",)
    a = gen.default_config("hopper")["protocol"]
    assert a["namespace"] == "Code5.TSD.MTX-Hopper-v1"
    assert a["perturbation_rad"] == [0.03] * 3 and a["bounce"]["amp_rad"] == [0.03] * 3


def test_cli_smoke():
    from hopper import cli
    orig = gen.default_config

    def fake(design="hopper"):
        cfg = orig(design)
        cfg["protocol"]["namespace"] = PROBE_NS
        cfg["protocol"]["parent_prefix"] = "hoppertest"
        return cfg
    gen.default_config = fake
    try:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            assert cli.main(["collect", "--design", "hopper", "--out", str(t / "main"), "--counts", "train=1,val=1,test=1"]) == 0
            assert cli.main(["collect", "--design", "hopper", "--out", str(t / "tb"), "--counts", "test=1", "--seed-offset", "1"]) == 0
            cfg = json.loads((t / "main" / "CONFIG.json").read_text())
            assert "hopper/gen.py" in cfg["source_sha256"] and "simcore/gen.py" in cfg["source_sha256"]
            assert cfg["source_sha256"]["gymnasium/envs/mujoco/assets/hopper.xml"] == "3ce93a055ffdcd83c0c701d2400768e40d2cbb9532f3c4ae33377c27f8b39f9e"
            assert cli.main(["assemble", "--main", str(t / "main"), "--testb", str(t / "tb"), "--out", str(t / "data")]) == 0
            for f in ("hopper_train.npz", "hopper_val.npz", "hopper_test.npz", "hopper_testB.npz", "SHA256SUMS.txt", "DATASET_CARD.json"):
                assert (t / "data" / f).exists(), f
            rc = cli.main(["audit", "--main", str(t / "main" / "data"), "--testb", str(t / "tb" / "data"),
                           "--out", str(t / "audit.json")])
            res = json.loads((t / "audit.json").read_text())
            assert res["hard_checks_ok"] and res["disjoint_ok"], res
            assert rc in (0, 4)
            assert cli.main(["compare", "--a", str(t / "data" / "hopper_test.npz"), "--b", str(t / "main" / "data" / "hopper_test.npz")]) == 0
            assert cli.main(["hashes", "--data", str(t / "main" / "data"), "--out", str(t / "h1.json")]) == 0
            assert cli.main(["collect", "--design", "hopper", "--out", str(t / "main2"), "--counts", "train=1,val=1,test=1"]) == 0
            assert cli.main(["hashes", "--data", str(t / "main2" / "data"), "--out", str(t / "h2.json")]) == 0
            assert cli.main(["xcheck", "--a", str(t / "h1.json"), "--b", str(t / "h2.json"), "--out", str(t / "x12.json")]) == 0
            assert cli.main(["hashes", "--data", str(t / "tb" / "data"), "--out", str(t / "h3.json")]) == 0
            assert cli.main(["xcheck", "--a", str(t / "h1.json"), "--b", str(t / "h3.json"), "--out", str(t / "x13.json")]) == 7
            assert cli.main(["compare", "--a", str(t / "data" / "hopper_test.npz"), "--b", str(t / "data" / "hopper_testB.npz")]) == 3
    finally:
        gen.default_config = orig


TESTS = [test_designs, test_model_files, test_cli_smoke, test_env_identity,
         test_condition_draws, test_schedules_match_rules, test_action_plans, test_health_rule,
         test_strength_semantics, test_payload_edit_exact_and_reversible, test_setconst_canonical,
         test_regimes_and_anchor_health, test_replay_and_order_independence, test_no_label_leakage_in_obs,
         test_collect_end_to_end_and_determinism, test_rejection_path, test_anchor_redraw_exhaustion,
         test_anchor_redraw_accepts, test_anchor_redraw_no_effect_without_exhaustion]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tests = [t for t in TESTS if not ("--quick" in argv and t.__name__ == "test_collect_end_to_end_and_determinism")]
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
