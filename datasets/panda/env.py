from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

MODEL_DIR_REL = ("panda_model", "mujoco_menagerie_6da8b971", "franka_emika_panda")
MENAGERIE_COMMIT = "6da8b971844845de6ab27315aacc3a3fea4ddb8e"
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRAVCOMP_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")
LIBFRANKA_K = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)
LIBFRANKA_D = (50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0)

PLANT = {
    "menagerie_commit": MENAGERIE_COMMIT,
    "model_file": "franka_emika_panda/panda.xml",
    "frame_skip": 25,
    "impedance_scale": 0.1,
    "K": [round(k * 0.1, 6) for k in LIBFRANKA_K],
    "D": [round(d * float(np.sqrt(0.1)), 6) for d in LIBFRANKA_D],
    "torque_limit": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
    "gravcomp_bodies": list(GRAVCOMP_BODIES),
    "payload": {"parent": "hand", "pos": [0.0, 0.0, 0.1034], "sphere_radius": 0.03},
    "joint_friction": {"frictionloss": 0.5, "damping": 1.0},
    "floor": {"type": "plane", "size": [0.0, 0.0, 0.05]},
}


def model_dir() -> Path:
    return Path(__file__).resolve().parent.parent.joinpath(*MODEL_DIR_REL)


def verify_model_files(mdir: Path | None = None) -> dict:
    root = (mdir or model_dir()).parent
    sums = (root / "SHA256SUMS.txt").read_text().split("\n")
    n = 0
    for line in sums:
        if not line.strip():
            continue
        h, name = line.split("  ", 1)
        p = root / name[2:] if name.startswith("./") else root / name
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != h:
            raise RuntimeError(f"pinned Menagerie file changed: {name} {got} != {h}")
        n += 1
    return {"files_checked": n, "sha256sums_sha256": hashlib.sha256((root / "SHA256SUMS.txt").read_bytes()).hexdigest(),
            "panda_xml_sha256": hashlib.sha256((root / "franka_emika_panda" / "panda.xml").read_bytes()).hexdigest()}


def build_spec(plant=None):
    import mujoco
    plant = plant or PLANT
    spec = mujoco.MjSpec.from_file(str(model_dir() / "panda.xml"))
    for n in ("finger_joint1", "finger_joint2"):
        spec.delete(spec.joint(n))
    spec.delete(spec.tendon("split"))
    for eq in list(spec.equalities):
        spec.delete(eq)
    spec.delete(spec.actuator("actuator8"))
    for k in list(spec.keys):
        spec.delete(k)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=list(plant["floor"]["size"]))
    pb = spec.body(plant["payload"]["parent"]).add_body(name="payload", pos=list(plant["payload"]["pos"]))
    pb.mass = 0.0
    pb.ipos = [0.0, 0.0, 0.0]
    pb.inertia = [0.0, 0.0, 0.0]
    pb.explicitinertial = True
    for b in plant["gravcomp_bodies"]:
        spec.body(b).gravcomp = 1.0
    for i, (jn, an) in enumerate(zip(ARM_JOINTS, ARM_ACTUATORS)):
        j = spec.joint(jn)
        j.actgravcomp = True
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.actfrcrange = [-plant["torque_limit"][i], plant["torque_limit"][i]]
        j.frictionloss = float(plant["joint_friction"]["frictionloss"])
        dmp = np.array(j.damping, dtype=np.float64)
        dmp[0] = float(plant["joint_friction"]["damping"])
        j.damping = dmp
        a = spec.actuator(an)
        gp = np.array(a.gainprm, dtype=np.float64)
        bp = np.array(a.biasprm, dtype=np.float64)
        gp[0] = plant["K"][i]
        bp[1] = -plant["K"][i]
        bp[2] = -plant["D"][i]
        a.gainprm = gp
        a.biasprm = bp
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec


def _model_sha256(mujoco, m) -> str:
    buf = np.empty(int(mujoco.mj_sizeModel(m)), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


class PandaAdapter:

    obs_dim = 14
    act_dim = 7

    def __init__(self, plant=None):
        import mujoco
        self.mujoco = mujoco
        self.plant = plant or PLANT
        self.files = verify_model_files()
        self.spec = build_spec(self.plant)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        if (m.nq, m.nv, m.nu) != (7, 7, 7):
            raise RuntimeError(f"unexpected Panda sizes nq/nv/nu = {m.nq}/{m.nv}/{m.nu}")
        self.frame_skip = int(self.plant["frame_skip"])
        self.dt = float(m.opt.timestep) * self.frame_skip
        self.lo = m.actuator_ctrlrange[:, 0].copy()
        self.hi = m.actuator_ctrlrange[:, 1].copy()
        self.mid = 0.5 * (self.lo + self.hi)
        self.half = 0.5 * (self.hi - self.lo)
        if not (np.array_equal(self.lo, m.jnt_range[:7, 0]) and np.array_equal(self.hi, m.jnt_range[:7, 1])):
            raise RuntimeError("actuator ctrlrange != joint range")
        self.payload_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.gc_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.plant["gravcomp_bodies"]])
        self.LIMIT = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state_size = int(mujoco.mj_stateSize(m, self.state_spec))
        self.base = {k: getattr(m, k).copy() for k in ("actuator_gainprm", "actuator_biasprm", "jnt_actfrcrange",
                                                       "body_gravcomp", "dof_frictionloss", "dof_damping",
                                                       "body_mass", "body_inertia")}
        self.model_sha256 = _model_sha256(mujoco, m)
        self.spec_xml_sha256 = hashlib.sha256(self.spec.to_xml().encode()).hexdigest()
        self._payload = None
        self._set_payload(0.0)
        self.set_dynamics(1.0, 1.0, 0.0)
        self.violation = None
        self.cur = (1.0, 1.0, 0.0)

    def _set_payload(self, mass: float):
        mj, m, d = self.mujoco, self.model, self.data
        st = np.empty(self.state_size, dtype=np.float64)
        mj.mj_getState(m, d, st, self.state_spec)
        m.body_mass[...] = self.base["body_mass"]
        m.body_inertia[...] = self.base["body_inertia"]
        r = float(self.plant["payload"]["sphere_radius"])
        m.body_mass[self.payload_id] = mass
        m.body_inertia[self.payload_id] = mass * 0.4 * r * r
        mj.mj_setConst(m, d)
        mj.mj_setState(m, d, st, self.state_spec)
        self._payload = float(mass)

    def set_dynamics(self, friction: float, gain: float, payload: float):
        for v, nm in ((friction, "friction"), (gain, "gain")):
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{nm} must be finite and positive")
        if not np.isfinite(payload) or payload < 0:
            raise ValueError("payload must be finite and >= 0")
        m, b = self.model, self.base
        m.actuator_gainprm[...] = b["actuator_gainprm"]
        m.actuator_biasprm[...] = b["actuator_biasprm"]
        m.actuator_gainprm[:7, 0] = b["actuator_gainprm"][:7, 0] * gain
        m.actuator_biasprm[:7, 1] = b["actuator_biasprm"][:7, 1] * gain
        m.actuator_biasprm[:7, 2] = b["actuator_biasprm"][:7, 2] * gain
        m.jnt_actfrcrange[...] = b["jnt_actfrcrange"]
        m.jnt_actfrcrange[:7] = b["jnt_actfrcrange"][:7] * gain
        m.body_gravcomp[...] = b["body_gravcomp"]
        m.body_gravcomp[self.gc_ids] = b["body_gravcomp"][self.gc_ids] * gain
        m.dof_frictionloss[...] = b["dof_frictionloss"]
        m.dof_damping[...] = b["dof_damping"]
        m.dof_frictionloss[:7] = b["dof_frictionloss"][:7] * friction
        m.dof_damping[:7] = b["dof_damping"][:7] * friction
        if float(payload) != self._payload:
            self._set_payload(float(payload))
        self.mujoco.mj_forward(m, self.data)
        self.cur = (float(friction), float(gain), float(payload))

    def obs(self) -> np.ndarray:
        d = self.data
        o = np.concatenate([d.qpos[:7], d.qvel[:7]]).astype(np.float32)
        if not np.isfinite(o).all():
            raise FloatingPointError("non-finite observation")
        return o

    def reset_to(self, qpos):
        mj, m, d = self.mujoco, self.model, self.data
        self.set_dynamics(1.0, 1.0, 0.0)
        mj.mj_resetData(m, d)
        d.qpos[:7] = np.asarray(qpos, np.float64)
        d.ctrl[:7] = np.asarray(qpos, np.float64)
        mj.mj_forward(m, d)
        self.violation = None

    def capture(self) -> dict:
        st = np.empty(self.state_size, dtype=np.float64)
        self.mujoco.mj_getState(self.model, self.data, st, self.state_spec)
        if not np.isfinite(st).all():
            raise FloatingPointError("non-finite integration state")
        o = self.obs()
        return {"state": st, "obs": o, "state_sha256": hashlib.sha256(st.tobytes()).hexdigest()}

    def restore(self, snap: dict):
        mj, m, d = self.mujoco, self.model, self.data
        self.set_dynamics(1.0, 1.0, 0.0)
        mj.mj_setState(m, d, snap["state"], self.state_spec)
        back = np.empty(self.state_size, dtype=np.float64)
        mj.mj_getState(m, d, back, self.state_spec)
        if not np.array_equal(back, snap["state"]):
            raise RuntimeError("mj_setState read-back differs")
        mj.mj_forward(m, d)
        if not np.array_equal(self.obs(), snap["obs"]):
            raise RuntimeError("restored observation differs")
        self.violation = None

    def step(self, action, ee_force=None, t=None) -> np.ndarray:
        mj, m, d = self.mujoco, self.model, self.data
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (7,) or not np.isfinite(a).all() or np.abs(a).max() > 1.0:
            raise ValueError("action must be 7 finite values in [-1, 1] (no clipping)")
        d.ctrl[:7] = self.mid + self.half * a.astype(np.float64)
        d.xfrc_applied[...] = 0.0
        if ee_force is not None:
            d.xfrc_applied[self.payload_id, :3] = np.asarray(ee_force, np.float64)
        try:
            for _ in range(self.frame_skip):
                mj.mj_step(m, d)
                if self.violation is None:
                    if d.ncon > 0:
                        g1, g2 = int(d.contact[0].geom1), int(d.contact[0].geom2)
                        self.violation = {"t": t, "reason": "contact",
                                          "geoms": [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g1) or str(g1),
                                                    mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g2) or str(g2)],
                                          "bodies": [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g1])),
                                                     mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g2]))]}
                    elif d.nefc > 0 and (np.asarray(d.efc_type[:d.nefc]) == self.LIMIT).any():
                        self.violation = {"t": t, "reason": "joint_limit"}
        finally:
            d.xfrc_applied[...] = 0.0
        return self.obs()

    def tcp(self) -> np.ndarray:
        return self.data.xpos[self.payload_id].copy()

    def identity(self) -> dict:
        m = self.model
        names = lambda obj, n: [self.mujoco.mj_id2name(m, obj, i) for i in range(n)]
        return {"menagerie_commit": MENAGERIE_COMMIT, "files": self.files, "plant": json.loads(json.dumps(self.plant)),
                "mujoco": self.mujoco.__version__, "model_sha256": self.model_sha256,
                "spec_xml_sha256": self.spec_xml_sha256, "timestep": float(m.opt.timestep),
                "integrator": int(m.opt.integrator), "frame_skip": self.frame_skip, "dt": self.dt,
                "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu), "nbody": int(m.nbody), "ngeom": int(m.ngeom),
                "bodies": names(self.mujoco.mjtObj.mjOBJ_BODY, m.nbody),
                "joints": names(self.mujoco.mjtObj.mjOBJ_JOINT, m.njnt),
                "actuators": names(self.mujoco.mjtObj.mjOBJ_ACTUATOR, m.nu),
                "ctrlrange": m.actuator_ctrlrange.tolist(), "body_mass": m.body_mass.tolist(),
                "total_mass_nominal": float(self.base["body_mass"].sum()),
                "obs": "qpos[0:7] (rad) + qvel[0:7] (rad/s)", "obs_dim": 14, "act_dim": 7}
