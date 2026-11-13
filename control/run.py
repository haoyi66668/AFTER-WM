from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from control import spec as SP


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_range(s, n):
    if not s:
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


class Loaded:
    def __init__(self, row, seed, device, runs_root, generic_only=False):
        from control import world_models as wm
        from pipeline import evaluate as E
        cfg = SP.MODEL_ROWS[row]
        if row in SP.RSSM_EVAL_MODE:
            from baselines import keyed_sampling as KS
            KS.set_eval_mode(SP.RSSM_EVAL_MODE[row])
        self.run_dir = os.path.join(runs_root, "runs", SP.DATASET, cfg, str(seed))
        comp = json.load(open(os.path.join(self.run_dir, "COMPLETE.json")))
        got = sha256_file(os.path.join(self.run_dir, "best.pt"))
        if got != comp.get("best_pt_sha256"):
            raise RuntimeError(f"best.pt sha256 differs from COMPLETE.json ({cfg}/{seed})")
        mod, model, mcfg, prot, st, ck = E.load_run(self.run_dir, device=device)
        self.mod, self.model, self.mcfg = mod, model, mcfg
        self.plan = wm.make_plan(row, mod, model, mcfg, device, generic_only=generic_only)
        self.genc = None
        self.gplan = None
        self.info = {"row": row, "config": cfg, "seed": seed, "best_pt_sha256": got, "module": prot["module"],
                     "forecaster": self.plan.name, "split_path": self.plan.split, "needs_grad": self.plan.needs_grad,
                     "stochastic": self.plan.stochastic}


def main(argv=None):
    from pipeline import registry as P
    ap = argparse.ArgumentParser(description="Closed-loop CEM control on JSHC with one world model (one row, given seeds).")
    P.add_cli_options(ap)
    ap.add_argument("--split", default="test", choices=sorted(SP.SEED_OFFSET))
    ap.add_argument("--spec", required=True, help="episode file (data/control_episodes/episodes_test_v1.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", required=True, choices=sorted(SP.MODEL_ROWS))
    ap.add_argument("--seeds", default=",".join(map(str, SP.SEEDS)))
    ap.add_argument("--episodes", default="")
    ap.add_argument("--k-mpc", type=int, default=SP.K_MPC)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    from control import planner as PL
    from control import plant as PT
    from control import world_models as wm
    from pipeline import evaluate as E
    import control_episodes as EP

    if P.DATASET != SP.DATASET:
        raise SystemExit(f"--dataset {P.DATASET} != {SP.DATASET}")
    torch.set_num_threads(1)
    E.setup_runtime(a.device if a.device != "cpu" else "cpu", deterministic=False)
    spec_sha = sha256_file(a.spec)
    S = np.load(a.spec, allow_pickle=False)
    meta = json.loads(str(S["meta_json"]))
    if meta["split"] != a.split or meta["namespace"] != SP.NS[a.split]:
        raise SystemExit(f"spec {meta['namespace']}/{meta['split']} is not the {a.split} episode file")
    n_ep = int(meta["n"])
    episodes = parse_range(a.episodes, n_ep)
    seeds = [int(x) for x in a.seeds.split(",")]
    T_WARM, K_MPC = int(meta["T_warm"]), int(a.k_mpc)
    so = int(meta["seed_offset"])
    plant = PT.Plant(SP.NS[a.split])
    prot = plant.protocol
    designs = EP.designs(prot, n_ep, so)
    stats = E.get_stats()
    om, osd, am, asd = stats.arrays()
    dev = a.device
    tdev = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
    am_t, asd_t = tdev(am), tdev(asd)
    om_v, osd_v = float(om[SP.VEL_INDEX]), float(osd[SP.VEL_INDEX])
    row = a.row
    for seed in seeds:
        odir = os.path.join(a.out, row, str(seed))
        os.makedirs(odir, exist_ok=True)
        todo = [i for i in episodes if not os.path.exists(os.path.join(odir, f"ep_{i:03d}.npz"))]
        if not todo:
            continue
        cem = PL.CEM(P.ACT_DIM, device=dev)
        L = Loaded(row, seed, dev, P.RUNS_ROOT)
        unit = {"row": row, "seed": seed, "split": a.split, "spec_sha256": spec_sha, "planner": PL.planner_config(),
                "gpu": torch.cuda.get_device_name(0) if dev.startswith("cuda") else None, "torch": torch.__version__,
                "k_mpc": K_MPC, "model": L.info}
        for i in todo:
            t_ep = time.time()
            fail = S["fail"][i]
            anchor, wa, wo = EP.start_episode(plant, prot, designs[i], so, fail, S["warm_act"][i])
            anchor_match = anchor["state_sha256"] == meta["episodes"][i]["anchor"]["state_sha256"]
            warm_match = bool(np.array_equal(wo, S["warm_obs"][i]))
            if not (anchor_match and warm_match):
                raise RuntimeError(f"episode {i}: anchor / warm-up does not reproduce the episode file")
            obs = np.zeros((T_WARM + K_MPC + 1, wo.shape[1]), np.float32)
            act = np.zeros((T_WARM + K_MPC, wa.shape[1]), np.float32)
            obs[:T_WARM + 1], act[:T_WARM] = wo, wa
            gen = torch.Generator().manual_seed(PL.noise_seed(a.split, seed, i))
            mean = cem.initial_mean()
            rew, xvel, ctrl, rgym = np.zeros(K_MPC), np.zeros(K_MPC), np.zeros(K_MPC), np.zeros(K_MPC)
            ptime, best, mstd = np.zeros(K_MPC), np.full(K_MPC, np.nan), np.full(K_MPC, np.nan)
            for k in range(K_MPC):
                t = T_WARM + k
                t0 = time.time()
                if L.plan.split and L.plan.graph_plan:
                    co_n, ca_n = (obs[t - 64:t + 1] - om) / osd, (act[t - 64:t] - am) / asd
                    if L.plan.graph_encode:
                        if L.genc is None:
                            _, _, co_t, ca_t = wm.make_ctx_tensors(co_n, ca_n, dev)
                            L.genc = wm.GraphedEncode(L.plan, co_t, ca_t)
                        enc = L.genc(co_n, ca_n)
                    else:
                        enc = L.plan.encode(co_n, ca_n)
                    if L.gplan is None:
                        L.gplan = PL.GraphedPlan(cem, L.plan, am_t, asd_t, om_v, osd_v, enc)
                        unit["graph_status"] = {"encode": None if L.genc is None else L.genc.status,
                                                "plan": L.gplan.status}
                    sol, info = L.gplan(enc, cem.draw(gen), mean)
                else:
                    enc = L.plan.encode((obs[t - 64:t + 1] - om) / osd, (act[t - 64:t] - am) / asd)

                    def cost_fn(samples, enc=enc):
                        pred = L.plan.predict(enc, (samples - am_t) / asd_t)
                        vel = pred[..., SP.VEL_INDEX] * osd_v + om_v
                        return PL.run_cost(vel, samples)
                    sol, info = cem.plan(cost_fn, mean, gen)
                mean = cem.shift(sol)
                a_t = sol[0].detach().cpu().numpy().astype(np.float32)
                best[k], mstd[k] = info["best_cost_last_iter"], info["max_std"]
                ptime[k] = time.time() - t0
                res, applied = plant.step(a_t, fail[t])
                act[t] = a_t
                obs[t + 1] = res.flat_observation
                xvel[k] = float(res.info["x_velocity"])
                rew[k] = PT.reward(xvel[k], applied)
                ctrl[k] = float(np.sum(np.square(applied.astype(np.float64))))
                rgym[k] = float(res.reward)
            rec = {"obs": obs, "act": act, "fail": np.asarray(fail[:T_WARM + K_MPC]), "reward": rew,
                   "x_velocity": xvel, "ctrl_sq": ctrl, "reward_gym": rgym, "plan_time": ptime, "best_cost": best,
                   "max_std": mstd, "warm_match": np.asarray(warm_match), "anchor_match": np.asarray(anchor_match)}
            fn = os.path.join(odir, f"ep_{i:03d}.npz")
            tmp = fn + f".tmp{os.getpid()}.npz"
            np.savez_compressed(tmp, **rec)
            if os.path.exists(fn):
                os.remove(tmp)
                raise RuntimeError(f"{fn} appeared meanwhile; not overwritten")
            os.replace(tmp, fn)
            print(json.dumps({"row": row, "seed": seed, "ep": i, "s": round(time.time() - t_ep, 1),
                              "plan_ms": round(1000 * ptime.mean(), 1), "return": float(rew.sum()),
                              "finite": bool(np.isfinite(rew).all())}), flush=True)
        with open(os.path.join(odir, f"UNIT_{a.split}_{os.getpid()}.json"), "w") as f:
            json.dump(unit, f, indent=1, default=str)
        del L
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
