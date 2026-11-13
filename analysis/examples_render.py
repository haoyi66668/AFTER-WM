from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys

import numpy as np

GREY = (0.62, 0.62, 0.64, 1.0)
GHOST_A = 0.55
KEYS = (1, 5, 10, 20, 40)
ROBOT = {
    "panda": {"groups": [(f"link{i}", (f"link{i}",)) for i in range(1, 8)] + [("hand", ("hand", "left_finger", "right_finger"))],
              "npos": 7},
    "ur5e": {"groups": [(b, (b,)) for b in ("shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link",
                                            "wrist_2_link", "wrist_3_link")], "npos": 6},
}


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def geom_points(mj, m, g):
    t, s = int(m.geom_type[g]), m.geom_size[g]
    T = mj.mjtGeom
    ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], 1)
    if t == T.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        return m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
    if t == T.mjGEOM_SPHERE:
        return fib_sphere(64) * s[0]
    if t == T.mjGEOM_ELLIPSOID:
        return fib_sphere(64) * s[:3]
    if t in (T.mjGEOM_CAPSULE, T.mjGEOM_CYLINDER):
        r, h = s[0], s[1]
        P = [np.c_[r * ring, np.full(12, z)] for z in np.linspace(-h, h, 7)]
        if t == T.mjGEOM_CAPSULE:
            c = math.cos(math.pi / 4)
            P += [np.c_[r * c * ring, np.full(12, sg * (h + r * c))] for sg in (-1, 1)]
            P += [np.array([[0, 0, h + r], [0, 0, -h - r]])]
        else:
            P += [np.array([[0, 0, h], [0, 0, -h]])]
        return np.concatenate(P, 0)
    if t == T.mjGEOM_BOX:
        c = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        f = np.vstack([np.eye(3), -np.eye(3)])
        return np.vstack([c, f]) * s[:3]
    return np.zeros((1, 3))


class Scene:
    def __init__(self, robot, W, H, render=True):
        import mujoco
        self.mj, self.robot, self.R = mujoco, robot, ROBOT[robot]
        env = importlib.import_module(f"{robot}.env")
        self.prov = {"verify_model_files": env.verify_model_files(), "mujoco": mujoco.__version__}
        spec = env.build_spec()
        for tex in list(spec.textures):
            if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                spec.delete(tex)
        spec.add_texture(name="viz_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[1.0, 1.0, 1.0], rgb2=[0.86, 0.89, 0.93], width=512, height=3072)
        spec.add_texture(name="viz_grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.93, 0.93, 0.92], rgb2=[0.86, 0.86, 0.85], width=512, height=512)
        mat = spec.add_material(name="viz_grid_mat", texrepeat=[6, 6], texuniform=True, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "viz_grid"
        spec.geom("floor").material = "viz_grid_mat"
        spec.geom("floor").rgba = [1.0, 1.0, 1.0, 1.0]
        lt = spec.worldbody.add_light(name="viz_sun", pos=[0.5, -0.5, 3.0], dir=[-0.15, 0.15, -1.0], castshadow=False,
                                      diffuse=[0.55, 0.55, 0.55], specular=[0.05, 0.05, 0.05])
        try:
            lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        except Exception:
            lt.directional = True
        spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        spec.visual.headlight.specular = [0.05, 0.05, 0.05]
        spec.visual.global_.offwidth = max(W, 1280)
        spec.visual.global_.offheight = max(H, 960)
        self.m = spec.compile()
        m = self.m
        self.d, self.d2 = mujoco.MjData(m), mujoco.MjData(m)
        self.W, self.H = W, H
        self.r = mujoco.Renderer(m, H, W) if render else None
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.groups = [g for g, _ in self.R["groups"]]
        self.body_group = {}
        for gi, (g, bodies) in enumerate(self.R["groups"]):
            for b in bodies:
                if bid(b) >= 0:
                    self.body_group[bid(b)] = gi
        self.payload_bid = bid("payload")
        self.gpts = {gi: [] for gi in range(len(self.groups))}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b in self.body_group and m.geom_group[g] <= 2 and m.geom_rgba[g][3] > 0:
                self.gpts[self.body_group[b]].append((g, geom_points(mujoco, m, g)))

    def qpos(self, obs_pos):
        q = np.zeros(self.m.nq)
        q[:self.R["npos"]] = obs_pos[:self.R["npos"]]
        return q

    def set_q(self, d, q):
        d.qpos[:] = q
        d.qvel[:] = 0
        self.mj.mj_kinematics(self.m, d)

    def pts(self, d):
        return [np.concatenate([d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T for g, v in self.gpts[gi]], 0)
                for gi in range(len(self.groups))]

    def link_err(self, qt, qp):
        self.set_q(self.d, qt)
        a = self.pts(self.d)
        self.set_q(self.d2, qp)
        b = self.pts(self.d2)
        return np.array([np.linalg.norm(x - y, axis=1).mean() for x, y in zip(a, b)]) * 100.0

    def tcp(self, q):
        self.set_q(self.d, q)
        return self.d.xpos[self.payload_bid].copy() if self.payload_bid >= 0 else self.d.xpos[1].copy()

    def _scene(self, d, look, colour):
        mj = self.mj
        self.cam.lookat[:] = look["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = look["distance"], look["azimuth"], look["elevation"]
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        scn = self.r.scene
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == mj.mjtObj.mjOBJ_GEOM and int(self.m.geom_bodyid[g.objid]) in self.body_group:
                colour(g, self.body_group[int(self.m.geom_bodyid[g.objid])])
        return scn

    def frame(self, q_true, ghosts=(), look=None):
        mj = self.mj
        self.set_q(self.d, q_true)

        def paint_true(g, gi):
            g.rgba[:] = GREY
            g.matid = -1
        self._scene(self.d, look, paint_true)
        img = self.r.render().astype(np.float64)
        for q, cols in ghosts:
            self.set_q(self.d2, q)

            def paint(g, gi, cols=cols):
                g.rgba[:] = (*cols[gi], 1.0)
                g.matid = -1
            self._scene(self.d2, look, paint)
            gimg = self.r.render().astype(np.float64)
            self.r.enable_segmentation_rendering()
            self._scene(self.d2, look, lambda g, gi: None)
            seg = self.r.render().copy()
            self.r.disable_segmentation_rendering()
            gid = seg[:, :, 0]
            mask = (seg[:, :, 1] == int(mj.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
            mask[mask] = np.isin(self.m.geom_bodyid[gid[mask]], list(self.body_group))
            img[mask] = (1 - GHOST_A) * img[mask] + GHOST_A * gimg[mask]
        return np.clip(np.round(img), 0, 255).astype(np.uint8)

    def look(self, qs):
        P = np.array([self.tcp(q) for q in qs])
        c = 0.5 * (P.min(0) + P.max(0))
        lookat = np.array([0.5 * c[0], 0.5 * c[1], 0.5 * (c[2] + 0.55)])
        az = math.degrees(math.atan2(c[1], c[0])) + 90.0 + 35.0
        return {"lookat": lookat, "distance": 1.45 if self.robot == "panda" else 1.75, "azimuth": az, "elevation": -14.0}


def colours(errs, vmax, cmap):
    return [cmap(min(max(e / vmax, 0.0), 1.0))[:3] for e in errs]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Link deviations and pose renders of one robot-arm example (poses only: qpos "
                                             "and forward kinematics; physics is never stepped).")
    ap.add_argument("--robot", required=True, choices=sorted(ROBOT))
    ap.add_argument("--example", required=True, help="npz written by analysis.examples_extract")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-render", action="store_true", help="link deviations only (no OpenGL needed)")
    a = ap.parse_args(argv)
    V = np.load(a.example, allow_pickle=False)
    W, H = 420, 360
    sc = Scene(a.robot, W, H, render=not a.no_render)
    obs, t0 = V["obs_seq"], int(V["t0"])
    errs, qtrue = {}, {}
    for row in ("afterwm", "compared"):
        pred = V[f"{row}|pred_raw"]
        for k in KEYS:
            qt = sc.qpos(obs[t0 + k])
            errs[(row, k)] = sc.link_err(qt, sc.qpos(pred[k - 1]))
            qtrue[k] = qt
    vmax = max(float(math.ceil(max(e.max() for e in errs.values()))), 1.0)
    os.makedirs(a.out, exist_ok=True)
    frames = {}
    for (row, k), e in errs.items():
        frames[f"{row}|k{k}"] = {"mean_cm": float(e.mean()), "max_cm": float(e.max()), "max_link": sc.groups[int(e.argmax())],
                                 "label": f"{e.mean():.1f} cm", "link_cm": dict(zip(sc.groups, e.tolist()))}
    if not a.no_render:
        import matplotlib
        from PIL import Image
        cmap = matplotlib.colormaps["viridis"]
        look = sc.look([qtrue[k] for k in KEYS])
        for (row, k), e in errs.items():
            img = sc.frame(qtrue[k], ghosts=[(sc.qpos(V[f"{row}|pred_raw"][k - 1]), colours(e, vmax, cmap))], look=look)
            Image.fromarray(img).save(os.path.join(a.out, f"{a.robot}_{row}_k{k}_render.png"))
    with open(os.path.join(a.out, f"{a.robot}_link_deviation.json"), "w") as f:
        json.dump({"robot": a.robot, "query_id": str(V["query_id"]), "compared": str(V["compared"]), "vmax_cm": vmax,
                   "groups": sc.groups, "frames": frames, "provenance": sc.prov}, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
