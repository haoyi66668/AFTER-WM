from __future__ import annotations
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import nn

from .protocol import ARMS, SLOW_ARMS
from .memory import residual_step


@dataclass(frozen=True)
class Prediction:
    d_base: torch.Tensor
    d_fast: torch.Tensor
    d_slow: torch.Tensor
    d_hat: torch.Tensor
    z_hat: torch.Tensor


@dataclass(frozen=True)
class Transition:
    belief: object
    route: object
    prediction: Prediction
    fast_update: object | None
    diagnostics: dict


@dataclass(frozen=True)
class Query:
    predicted_states: torch.Tensor
    fast_states: torch.Tensor
    slow_states: torch.Tensor
    slow_corrections: torch.Tensor
    route_log_p: torch.Tensor


def project_ball(value, radius):

    if not 0 < radius:
        raise ValueError("positive radius required")
    norm = torch.linalg.vector_norm(value.double(), dim=-1, keepdim=True)
    margin=1-4*torch.finfo(value.dtype).eps
    scale=torch.where(norm>radius,margin*radius/norm.clamp_min(1e-12),torch.ones_like(norm)).to(value.dtype)
    return value * scale


def bounded_coordinate(value,scale):

    rms=value.norm(dim=-1,keepdim=True)/(value.shape[-1]**.5)
    return value/(scale+rms)


def centered_difference(module,with_input,zero_input,hidden=None):

    state={n:v.double() if v.is_floating_point() else v
           for n,v in module.named_parameters()}
    state.update({n:v.double() if v.is_floating_point() else v for n,v in module.named_buffers()})
    a=(with_input.double(),) if hidden is None else (with_input.double(),hidden.double())
    b=(zero_input.double(),) if hidden is None else (zero_input.double(),hidden.double())
    left=torch.func.functional_call(module,state,a)
    right=torch.func.functional_call(module,state,b)
    return (left-right).to(with_input.dtype)


def fields(b):
    return b.fast.q_previous, b.fast.F, b.H, b.S


def combine_beliefs(beliefs, dim=0):
    if not beliefs:
        raise ValueError("empty belief list")
    b = beliefs[0]
    values = [torch.cat([fields(x)[j] for x in beliefs], dim=dim) for j in range(4)]
    return type(b)(type(b.fast)(values[0], values[1]), values[2], values[3])


FS_A_MATCH_TARGET = 242_352
FS_A_CONTEXT_HIDDEN = 133


class ContextBasis(nn.Module):

    def __init__(self,inputs,outputs,dim,hidden=64):
        super().__init__()
        if type(hidden) is not int or hidden < 8:
            raise ValueError("positive context hidden width required")
        self.outputs,self.dim,self.hidden=outputs,dim,hidden
        self.network=nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Linear(hidden,outputs*dim))
        nn.init.normal_(self.network[-1].weight,std=.001)
        nn.init.zeros_(self.network[-1].bias)
        raw=torch.randn(max(outputs,dim),min(outputs,dim),dtype=torch.float64)
        q,_=torch.linalg.qr(raw,mode="reduced")
        initial=q.T if dim>=outputs else q
        self.offset=nn.Parameter(initial.float())

    def forward(self,x):
        raw=self.offset+self.network(x).reshape(*x.shape[:-1],self.outputs,self.dim)
        norm=raw.double().flatten(-2).norm(dim=-1,keepdim=True)
        scale=torch.clamp(4./norm.clamp_min(1e-30),max=1.).to(raw.dtype)
        return raw*scale.unsqueeze(-1)


class Candidate(nn.Module):
    def __init__(self, parent_fast, core_factory, base_prediction_factory, arm, seed,
                 *, target_new_parameters=None):
        super().__init__()
        if arm not in (*ARMS, "anchor") or type(seed) is not int:
            raise ValueError("known arm and integer seed required")
        if any(p.grad is not None for p in parent_fast.parameters()):
            raise ValueError("gradient-free parent required")
        self.arm,self.seed=arm,seed
        self.family="anchor" if arm=="anchor" else arm.split("_",1)[0]
        self._base_prediction_factory=base_prediction_factory
        self._slow_enabled,self._error_enabled=True,True
        self._slow_current_error=True
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            fast=copy.deepcopy(parent_fast).requires_grad_(False).eval()
            self.core=core_factory(fast)
            c,sc=fast.config,self.core.config
            self.static_readouts=nn.ModuleList()
            self.register_buffer("history_scale",torch.ones(c.num_experts))
            self.register_buffer("state_scale",torch.ones(c.num_experts))
            self.core.slow_readouts=nn.ModuleList()
            self.core.consolidators=nn.ModuleList()
            self.core.slow_updaters=nn.ModuleList()
            if self.family in ("FS","D"):
                from dataclasses import is_dataclass,replace
                if is_dataclass(sc):
                    self.core.config=replace(sc,slow_max_norm=4.,slow_step_limit=.1 if self.family=="FS" else 4.)
                else:
                    sc.slow_max_norm=4.;sc.slow_step_limit=.1 if self.family=="FS" else 4.
                context_hidden = FS_A_CONTEXT_HIDDEN if self.arm == "FS_A" else 64
                self.core.slow_readouts=nn.ModuleList(ContextBasis(
                    c.observation_dim+c.action_dim+c.fast_dim,
                    c.observation_dim,sc.slow_dim,hidden=context_hidden) for _ in range(c.num_experts))
                if self.arm == "FS_A":
                    actual = self.slow_parameter_count()
                    self.matching = {"target": FS_A_MATCH_TARGET, "actual": actual,
                                     "context_hidden": context_hidden,
                                     "active_weights_only": True,
                                     "target_total_parameters": 861705,
                                     "actual_total_parameters": 446774 + 172579 + actual}
            if self.family=="D":
                count=self.slow_parameter_count()
                if target_new_parameters!=count:
                    raise ValueError("recent-error control must match every active slow parameter")
                self.matching={"target":target_new_parameters,"actual":count,
                    "control":"same basis; only most recent observed residual; eta=1; no persistent S"}
        self.to(next(parent_fast.parameters()))
        self.set_training(False)

    @property
    def fast_model(self):
        return self.core.fast_model

    @property
    def config(self):
        return self.core.config

    @property
    def query_mode(self):
        return "full"

    def learning_modules(self):
        if self.family == "anchor":
            return {}
        f = self.fast_model
        result = {"Uq": f.accepted_model.q_only_model.belief_updater,
                  "UF": f.fast_updater, "GF": f.fast_readout_experts}
        if self.family in ("FS","D"):
            result.update(GS_basis=self.core.slow_readouts)
        return result

    def slow_parameter_count(self):
        return sum(p.numel() for m in (self.core.consolidators, self.core.slow_updaters,
                                      self.core.slow_readouts) for p in m.parameters())

    @torch.no_grad()
    def set_scales(self, history, state):
        for name, values in (("history_scale", history), ("state_scale", state)):
            target = getattr(self, name)
            value = torch.as_tensor(values, dtype=target.dtype, device=target.device)
            if value.shape != target.shape or not torch.isfinite(value).all() or (value <= 0).any():
                raise ValueError("invalid fixed training-only scale: " + name)
            target.copy_(value)
        return self

    def declared_trainable_named(self):
        allowed = {id(p) for m in self.learning_modules().values() for p in m.parameters()}
        return [(n,p) for n,p in self.named_parameters() if id(p) in allowed]

    def set_training(self, mode):
        if type(mode) is not bool:
            raise TypeError("boolean required")
        self.requires_grad_(False)
        self.train(mode)
        self.fast_model.eval()
        for m in self.learning_modules().values():
            m.requires_grad_(mode)
            m.train(mode)
        if not mode:
            self.zero_grad(set_to_none=True)
        return self

    def initialize_belief(self, state):
        return self.core.initialize_belief(state)

    def validate_belief(self, belief, state):
        self.core.validate_belief(belief, state)
        if self.family in ("FS","D"):
            if torch.count_nonzero(belief.H.detach()):
                raise ValueError("direct residual memory has no H; H must remain zero")
            radius=self.config.slow_max_norm
            if (belief.S.detach().double().flatten(1).norm(dim=-1)>radius+1e-5).any():
                raise ValueError("direct residual global state bound")

    @contextmanager
    def intervention(self, *, slow=True, error=True):
        if type(slow) is not bool or type(error) is not bool:
            raise TypeError("boolean intervention required")
        previous = self._slow_enabled, self._error_enabled
        self._slow_enabled, self._error_enabled = slow, error
        try:
            yield self
        finally:
            self._slow_enabled, self._error_enabled = previous

    def correction(self, z, a, F, S, k, *, enabled=True):
        if not enabled or not self._slow_enabled or self.family in ("anchor", "F"):
            return torch.zeros_like(z)
        basis=self.core.slow_readouts[k](torch.cat((z,a,F),-1))
        return (basis.double()@S.double().unsqueeze(-1)).squeeze(-1).to(z.dtype)

    def basis(self,z,a,b):
        return torch.stack([net(torch.cat((z,a,b.fast.F[:,k]),-1))
                            for k,net in enumerate(self.core.slow_readouts)],1)

    def evidence_gradient_report(self):
        if self.family not in ("FS","D"):return {}
        return {"basis_gradient_l2":sum(float(p.grad.detach().double().square().sum())
            for p in self.core.slow_readouts.parameters() if p.grad is not None)**.5}

    def predict_experts(self, z, a, b, *, mode="full"):
        if mode not in ("off","shadow","full"):raise ValueError("invalid slow mode")
        self.validate_belief(b,z)
        f = self.fast_model.predict_experts(z,a,b.fast)
        slow = torch.stack([self.correction(z,a,b.fast.F[:,k],b.S[:,k],k,
            enabled=mode=="full") for k in range(self.fast_model.config.num_experts)], -2)
        return Prediction(f.d_base,f.d_fast,slow,f.d_hat+slow,f.z_hat+slow)

    def observe_transition(self,z,a,next_z,b,*,mode="full"):
        if mode not in ("off","shadow","full"):raise ValueError("invalid slow mode")
        self.validate_belief(b,z)
        route = self.fast_model.prior_routing(z,a,b.fast)
        f = self.fast_model.predict_experts(z,a,b.fast)
        delta = torch.stack([self.correction(z,a,b.fast.F[:,k],b.S[:,k],k,
            enabled=mode=="full") for k in range(self.fast_model.config.num_experts)], -2)
        prediction = Prediction(f.d_base,f.d_fast,delta,f.d_hat+delta,f.z_hat+delta)
        if self._error_enabled:
            feedback = f
            native = self._base_prediction_factory(feedback.d_base,feedback.d_hat,feedback.z_hat)
            update = self.fast_model.post_action_update(z,a,next_z,route,native,b.fast)
            fast = update.belief
        else:
            update = None
            fast = type(b.fast)(route.p,self.fast_model.fast_retention*b.fast.F)
        H,S,diag=torch.zeros_like(b.H),b.S,{}
        if (self.family in ("FS","D") and mode!="off" and self._slow_enabled
                and self._error_enabled and self._slow_current_error):
            matrices=self.basis(z,a,b)
            p=torch.softmax(route.log_p.double(),-1)
            fast_mean=(p[:,:,None]*f.z_hat.double()).sum(1)
            write_state=b.S if self.family=="FS" else torch.zeros_like(b.S)
            slow_mean=(p[:,:,None]*(matrices.double()@write_state.double().unsqueeze(-1)).squeeze(-1)).sum(1)
            residual=next_z.double()-fast_mean-slow_mean
            S,diag=residual_step(write_state,matrices,p,residual,
                eta=.05 if self.family=="FS" else 1.,ridge=.01,radius=4.,
                step_limit=.1 if self.family=="FS" else 4.)
            if not torch.is_grad_enabled():
                tolerance=1e-8+2e-6*diag["observed_mse_before"]
                if (diag["observed_mse_change"]>tolerance).any():
                    raise AssertionError("observed-context residual descent violated")
        elif self.family in ("FS","D"):
            if self.family=="D":S=torch.zeros_like(b.S)
            zero=z.new_zeros(len(z))
            diag={"update_audit_active":zero,
                  "observed_mse_before":zero,"observed_mse_after_fixed_context":zero,
                  "observed_mse_change":zero,"state_step_norm":zero,
                  "state_global_norm":S.double().flatten(1).norm(dim=-1),
                  "lipschitz_upper":zero,"effective_step":zero,
                  "step_clipped":zero,"state_projected":zero,"residual_norm":zero}
        after = type(b)(fast,H,S)
        self.validate_belief(after,z)
        diag.update(fast_prediction=f.z_hat.detach(), slow_correction=delta.detach())
        return Transition(after,route,prediction,update,diag)

    @torch.no_grad()
    def natural_no_current_error(self,z,a,b):
        with self.intervention(slow=self._slow_enabled,error=False):
            return self.observe_transition(z,a,z,b).belief.detach()

    @torch.no_grad()
    def burn_in(self,observations,actions,*,mode="full"):
        if observations.shape[:2] != (actions.shape[0],actions.shape[1]+1):
            raise ValueError("burn-in transition geometry differs")
        b=self.initialize_belief(observations[:,0])
        for t in range(actions.shape[1]):
            b=self.observe_transition(observations[:,t],actions[:,t],observations[:,t+1],b,mode=mode).belief
        return b.detach()

    def predict_future(self,z,actions,b,*,mode="full"):
        if mode not in ("off","shadow","full"):raise ValueError("invalid slow mode")
        self.validate_belief(b,z)
        if (actions.ndim!=3 or actions.shape[0]!=z.shape[0] or actions.shape[1]<1 or
            actions.shape[2]!=self.fast_model.config.action_dim or actions.dtype!=z.dtype or
            actions.device!=z.device or not torch.isfinite(actions).all()):
            raise ValueError("query actions must be [B,H,A]")
        route=self.fast_model.prior_routing(z,actions[:,0],b.fast)
        paths,fs,ss,cs=[],[],[],[]
        for k in range(self.fast_model.config.num_experts):
            state,F,S=z,b.fast.F[:,k],b.S[:,k]
            path,fp,sp,cp=[],[],[],[]
            for h in range(actions.shape[1]):
                f=self.fast_model.predict_expert(state,actions[:,h],F,k)
                delta=self.correction(state,actions[:,h],F,S,k,enabled=mode=="full")
                state=f.z_hat+delta
                path.append(state);fp.append(F);sp.append(S);cp.append(delta)
                F=self.fast_model.fast_retention*F
            for dest,value in ((paths,path),(fs,fp),(ss,sp),(cs,cp)):
                dest.append(torch.stack(value,1))
        return Query(*(torch.stack(v,-2) for v in (paths,fs,ss,cs)),route.log_p)


def trajectory_nll(query,targets,sigma=.5):
    states=query.predicted_states
    if targets.shape != (states.shape[0],states.shape[1],states.shape[3]) or not torch.isfinite(targets).all():
        raise ValueError("invalid targets")
    error=states-targets.detach().unsqueeze(-2)
    cost=torch.zeros_like(query.route_log_p,dtype=torch.float64)
    for h in range(states.shape[1]):
        cost=cost+error[:,h].double().square().sum(-1)/(2*sigma*sigma)
    logp=torch.log_softmax(query.route_log_p.double(),-1)
    return (-torch.logsumexp(logp-cost,-1)/(states.shape[1]*states.shape[3])).to(states.dtype)
