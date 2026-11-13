from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import torch

from pipeline import registry as P
from pipeline.eval_api import Forecast, QueryBatch

warnings.filterwarnings("ignore", message='"is" with a literal', category=SyntaxWarning)
from baselines.third_party.mts3.agent.worldModels.MTS3 import MTS3

NAME = "mts3"
FAMILY = "baseline"
EVAL_CHUNK = 720
DISCLOSURES = (
    "D1. Causal manager action mask (edit 2 of baselines/third_party/mts3/apply_patch.py). Upstream lets the manager's abstract-action Bayesian aggregation see ALL actions of the next block; here only actions with window index <= 64 (a_t and earlier) are valid for the manager. Block 4 (indices 60..74) uses a_{t-4..t}; blocks 5-6 (indices 75..104) get the BA prior (no abstract-action information). The worker still receives every known action a_t..a_{t+39}. An exact per-step mask would need 40 passes; the mask is used identically in training and evaluation.",
    'D2. Package-relative imports in the fetched MTS3.py (edit 1 of baselines/third_party/mts3/apply_patch.py); no behaviour change.',
    'D3. Observation mask fixed to the evaluation mask (indices 0..64 valid, 65..104 invalid) in training. The official random 15% observation dropout + whole-window task-imputation curriculum is not used.',
    'D4. Loss = upstream utils/Losses.gaussian_nll (0.5*(log var + log 2pi + (y-m)^2/var), var+1e-12, sum over the observation dims, mean over samples) restricted to the 40 future predictions (indices 64..103 -> targets z_{t+1..t+40}); weight 1 there, 0 elsewhere. Upstream averages over all time steps of randomly masked sequences.',
    'D5. Training windows come from the shared schedule (pipeline.data.make_schedule, t ~ U{64..260}, 32 windows of 105 obs per update) instead of upstream 24x15-step sequences with batch 1500; number of updates and VAL interval set by the shared protocol (pipeline.trainer).',
    'D6. Optimizer = upstream: Adam lr 3e-3 (halfCheetah learn/default.yaml), grad-norm clip 5.0 (repre_learn_mts3.py). Upstream batch 1500 / 1500 epochs are not used (protocol budget).',
    'D7. Architecture = upstream halfCheetah config (default_mts3.yaml): latent_obs_dim 30, H=15, one hidden layer per sub-network, ReLU, softplus variances, initial covar 10, trans covar 0.1. Only the hidden widths are scaled with the official ratio 120:240:240:240:120 (obs enc : abs-obs enc : abs-act enc : decoder : control net) to reach 700,000 +-1% total parameters (734/1468/1468/1468/734 -> 699,676 for 17-dim observations and 6-dim actions; official halfCheetah size: 129,214 incl. 2,400 unused LayerNorm parameters, see D8).',
    "D8. The upstream modules create LayerNorms that the upstream forward never uses (Encoder ln_pre/ln_post with out_norm 'none'; SplitDiagGaussianDecoder ln_mean/ln_var whose calls are commented out). They are deleted from the instance after construction, so the budget counts only functional parameters and every trainable tensor receives gradients. task_net_hidden_units / bandwidth / process_noise_hidden_units are read by nothing upstream (no effect, kept as in the yaml).",
    "D9. Upstream behaviour kept unchanged: gaussian_marginalization.Predict (manager) adds the two covariance lists with '+' (list concatenation); the zip with the process noise then drops the abstract-action covariance. Upstream also detaches the worker state at block boundaries (kept).",
    "D10. Upstream device handling (modules pick 'cuda' whenever available) is overridden by setting each module's _device attribute to the parameter device after construction / .to(); no math change.",
    'D11. Upstream BA squeezes the batch dimension, which crashes for batch size 1. forecast() never calls the core with fewer than EVAL_CHUNK rows (see D14), so this cannot occur.',
    'D12. Configuration is a SimpleNamespace transcription of the halfCheetah yaml (no omegaconf dependency).',
    'D13. Evaluation is deterministic (upstream is deterministic); the native per-step diagonal variance of the decoder is returned as Forecast.comp_var (K=1).',
    "D14. Implementation detail without effect on the math: forecast() runs the model on fixed-size chunks of EVAL_CHUNK=720 queries (the last chunk padded by repeating its last query, padded rows discarded), so every GEMM has the same shape whatever the caller's batch size and per-query outputs are bitwise independent of batch size and order (without this they differ at the 1e-6 level on GPU).",
)

OFFICIAL_WIDTHS = {"obs_encoder": 120, "abstract_obs_encoder": 240, "abstract_act_encoder": 240,
                   "obs_decoder": 240, "control_net": 120}
OFFICIAL_LATENT_OBS_DIM = 30
OFFICIAL_H = 15


def param_count(L: int, w: dict, obs_dim: int = P.OBS_DIM, act_dim: int = P.ACT_DIM) -> int:
    w1, w2, w3, w4, w5 = (w["obs_encoder"], w["abstract_obs_encoder"], w["abstract_act_encoder"],
                          w["obs_decoder"], w["control_net"])
    enc_obs = (obs_dim + 1) * w1 + 2 * (w1 * L + L)
    enc_absobs = (obs_dim + 2) * w2 + 2 * (w2 * L + L)
    enc_absact = (act_dim + 2) * w3 + 2 * (w3 * 2 * L + 2 * L)
    dec = (2 * L + 1) * w4 + (3 * L + 1) * w4 + 2 * (w4 * obs_dim + obs_dim)
    worker = 8 * L * L + (act_dim + 1) * w5 + 2 * w5 + (w5 * 2 * L + 2 * L) + 2 * L
    manager = 8 * L * L + 2 * L
    return enc_obs + enc_absobs + enc_absact + dec + worker + manager


def solve_widths(target: int, L: int) -> tuple[dict, int]:
    best = None
    for a in range(8, 20000):
        w = {"obs_encoder": a, "abstract_obs_encoder": 2 * a, "abstract_act_encoder": 2 * a,
             "obs_decoder": 2 * a, "control_net": a}
        n = param_count(L, w)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (w, n)
        if n > target:
            break
    return best


def default_config(budget_class: str = "baseline", **variant) -> dict:
    if budget_class != "baseline":
        raise ValueError(f"mts3 is a compared model; budget_class must be 'baseline', got {budget_class!r}")
    allowed = {"latent_obs_dim", "H_manager", "lr", "grad_clip"}
    bad = set(variant) - allowed
    if bad:
        raise ValueError(f"unknown mts3 variant keys {sorted(bad)}")
    L = int(variant.get("latent_obs_dim", OFFICIAL_LATENT_OBS_DIM))
    H = int(variant.get("H_manager", OFFICIAL_H))
    widths, total = solve_widths(P.BUDGET_BASELINE, L)
    t_obs = P.CONTEXT + 1 + P.MAX_H
    t_pad = int(math.ceil(t_obs / H) * H)
    return {
        "name": NAME, "family": FAMILY, "budget_class": budget_class,
        "budget_target": P.BUDGET_BASELINE, "budget_tol": P.BUDGET_TOL, "param_total": total,
        "latent_obs_dim": L, "H_manager": H, "widths": widths,
        "width_scale_vs_official": widths["obs_encoder"] / OFFICIAL_WIDTHS["obs_encoder"],
        "window_obs": t_pad, "window_act": t_pad + H,
        "obs_valid_upto": P.CONTEXT, "manager_action_valid_upto": P.CONTEXT,
        "loss": "upstream gaussian_nll on future indices 64..103 (sum dims, mean samples)",
        "optimizer": "Adam", "lr": float(variant.get("lr", 3e-3)), "adam_eps": 1e-8,
        "grad_clip": float(variant.get("grad_clip", 5.0)),
        "eval_chunk": EVAL_CHUNK,
        "variant": dict(variant),
    }


def _official_cfg(cfg: dict) -> dict:
    w = cfg["widths"]
    enc = lambda width: {"hidden_units_list": [width], "out_norm": "none", "variance_activation": "softplus"}
    abs_enc = lambda width, extra: dict(enc(width), learn_initial_state_covar=False, initial_state_covar=10,
                                        state_abs=True, time_embed={"type": "time", "dim": 1}, **extra)
    return {"mts3": {
        "clip_gradients": True,
        "latent_obs_dim": cfg["latent_obs_dim"],
        "pixel_obs": False,
        "time_scale_multiplier": cfg["H_manager"],
        "initial_state_covar": 10,
        "manager": {
            "bandwidth": 3, "process_noise_hidden_units": [30], "trans_net_hidden_activation": "Tanh",
            "process_noise_hidden_activation": "ReLU", "learn_trans_covar": True,
            "multi_gaussian_l_transform": False, "trans_covar": 0.1, "learn_initial_state_covar": False,
            "initial_state_covar": 10,
            "abstract_obs_encoder": abs_enc(w["abstract_obs_encoder"], {}),
            "abstract_act_encoder": abs_enc(w["abstract_act_encoder"], {"aggregator": "BA"}),
        },
        "worker": {
            "bandwidth": 3, "decode_var": True,
            "obs_encoder": enc(w["obs_encoder"]),
            "trans_net_hidden_units": [], "control_net_hidden_units": [w["control_net"]],
            "task_net_hidden_units": [w["control_net"]], "process_noise_hidden_units": [30],
            "trans_net_hidden_activation": "Tanh", "control_net_hidden_activation": "ReLU",
            "process_noise_hidden_activation": "ReLU", "task_net_hidden_activation": "ReLU",
            "learn_trans_covar": True, "trans_covar": 0.1, "learn_initial_state_covar": False,
            "initial_state_covar": 10, "clip_gradients": True,
            "obs_decoder": {"hidden_units_list": [w["obs_decoder"]], "variance_activation": "softplus"},
        },
        "decode": {"reward": False, "obs": True},
    }}


def _ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_ns(v) for v in x]
    return x


class MTS3Model(torch.nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        self.H = int(cfg["H_manager"])
        self.obs_valid_upto = int(cfg["obs_valid_upto"])
        self.mgr_valid_upto = int(cfg["manager_action_valid_upto"])
        want_cuda = torch.cuda.is_available()
        core = MTS3(input_shape=[P.OBS_DIM], action_dim=P.ACT_DIM, config=_ns(_official_cfg(cfg)),
                    use_cuda_if_available=want_cuda)
        for enc in (core._obsEnc, core._absObsEnc, core._absActEnc):
            assert enc._output_normalization.lower() == "none"
            del enc._ln_pre
            del enc._ln_post
        del core._obsDec._ln_mean
        del core._obsDec._ln_var
        self.core = core
        self._sync_device()

    def _sync_device(self):
        p = next(self.parameters(), None)
        if p is None:
            return
        for m in self.core.modules():
            if hasattr(m, "_device"):
                m._device = p.device

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        self._sync_device()
        return out

    def assemble(self, ctx_obs, ctx_act, fut_act):
        B, Hf = ctx_obs.shape[0], fut_act.shape[1]
        c = P.CONTEXT
        t_obs = c + 1 + Hf
        t_pad = int(math.ceil(t_obs / self.H) * self.H)
        a_len = t_pad + self.H
        obs = ctx_obs.new_zeros(B, t_pad, P.OBS_DIM)
        obs[:, : c + 1] = ctx_obs
        act = ctx_obs.new_zeros(B, a_len, P.ACT_DIM)
        act[:, :c] = ctx_act
        act[:, c: c + Hf] = fut_act
        obs_valid = torch.zeros(B, t_pad, 1, dtype=torch.bool, device=ctx_obs.device)
        obs_valid[:, : self.obs_valid_upto + 1] = True
        act_valid = torch.zeros(B, a_len, 1, dtype=torch.bool, device=ctx_obs.device)
        act_valid[:, : self.mgr_valid_upto + 1] = True
        return obs, act, obs_valid, act_valid

    def predict(self, ctx_obs, ctx_act, fut_act):
        obs, act, ov, av = self.assemble(ctx_obs, ctx_act, fut_act)
        mean, var = self.core(obs, act, ov, av)[:2]
        c, Hf = P.CONTEXT, fut_act.shape[1]
        return mean[:, c: c + Hf], var[:, c: c + Hf]


def build(cfg: dict, seed: int, device: str, stats=None) -> MTS3Model:
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        model = MTS3Model(cfg)
    return model.to(device)


def make_optimizer(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=cfg.get("adam_eps", 1e-8))


def native_gaussian_nll(target, mean, var):
    var = var + 1e-12
    mahal = (target - mean) ** 2 / var
    element_wise_nll = 0.5 * (torch.log(var) + math.log(2 * math.pi) + mahal)
    return element_wise_nll.sum(dim=-1).mean()


def train_step(model, batch: dict, step: int, cfg: dict):
    mean, var = model.predict(batch["ctx_obs"], batch["ctx_act"], batch["fut_act"])
    target = batch["fut_obs"]
    loss = native_gaussian_nll(target, mean, var)
    with torch.no_grad():
        err2 = (mean - target) ** 2
        vals = torch.stack([loss.detach(), err2.mean(), err2[:, 0].mean(), err2[:, -1].mean(), var.mean()])
        v = vals.tolist()
    return loss, {"nll_future": v[0], "mse_future": v[1], "mse_h1": v[2], "mse_h40": v[3], "var_mean": v[4]}


def parameter_report(model: MTS3Model, cfg: dict | None = None) -> dict:
    core = model.core
    groups = {
        "worker_obs_encoder": core._obsEnc, "manager_abstract_obs_encoder": core._absObsEnc,
        "manager_abstract_act_encoder": core._absActEnc, "worker_obs_decoder": core._obsDec,
        "worker_transition_A_C_control_noise": core._state_predict,
        "manager_transition_A_B_noise": core._task_predict,
    }
    g = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert sum(g.values()) == total, (g, total)
    target = P.BUDGET_BASELINE
    L = core._lod
    return {
        "total": total, "trainable": trainable, "frozen": total - trainable, "stored_nongrad": 0,
        "buffers": sum(b.numel() for b in model.buffers()), "groups": g,
        "budget_class": "baseline", "budget_target": target, "budget_tol": P.BUDGET_TOL,
        "within_budget": abs(total - target) <= P.BUDGET_TOL * target,
        "per_query_state": {"worker_state_dim": 2 * L, "worker_cov_dims": 3 * L,
                            "manager_task_dim": 2 * L, "manager_cov_dims": 3 * L},
        "removed_unused_upstream_layernorm_params": _removed_ln_count(cfg) if cfg else None,
    }


def _removed_ln_count(cfg: dict) -> int:
    w, L = cfg["widths"], cfg["latent_obs_dim"]
    enc = 2 * w["obs_encoder"] + 2 * L + 2 * w["abstract_obs_encoder"] + 2 * L + 2 * w["abstract_act_encoder"] + 4 * L
    return enc + 4 * w["obs_decoder"]


class MTS3Forecaster:
    name = NAME
    family = FAMILY
    budget_class = "baseline"
    needs_grad_at_eval = False
    stochastic = False

    def __init__(self, model: MTS3Model, cfg: dict):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def forecast(self, q: QueryBatch) -> Forecast:
        dev = next(self.model.parameters()).device
        xs = [q.ctx_obs.to(dev, torch.float32), q.ctx_act.to(dev, torch.float32), q.fut_act.to(dev, torch.float32)]
        B = xs[0].shape[0]
        chunk = int(self.cfg.get("eval_chunk", EVAL_CHUNK))
        means, vars_ = [], []
        for s in range(0, B, chunk):
            part = [x[s: s + chunk] for x in xs]
            n = part[0].shape[0]
            if n < chunk:
                part = [torch.cat([x, x[-1:].expand(chunk - n, *x.shape[1:])], 0) for x in part]
            m, v = self.model.predict(*part)
            means.append(m[:n])
            vars_.append(v[:n])
        mean, var = torch.cat(means, 0), torch.cat(vars_, 0)
        return Forecast(mean=mean, comp_mean=mean[:, :, None, :],
                        comp_logw=torch.zeros(B, 1, device=dev, dtype=mean.dtype),
                        comp_var=var[:, :, None, :])

    def parameter_report(self) -> dict:
        return parameter_report(self.model, self.cfg)


def forecaster(model, cfg) -> MTS3Forecaster:
    return MTS3Forecaster(model, cfg)
