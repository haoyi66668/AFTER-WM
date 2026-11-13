import torch


def residual_step(state, basis, probabilities, residual, *, eta=.05,
                  ridge=.01, radius=4., step_limit=.1):

    if not (0 < eta < 2 and ridge > 0 and radius > 0 and step_limit > 0):
        raise ValueError("invalid residual memory constants")
    b,k,d=state.shape
    if basis.ndim!=4 or basis.shape[:2]!=(b,k) or basis.shape[-1]!=d:
        raise ValueError("basis/state geometry mismatch")
    o=basis.shape[2]
    if probabilities.shape!=(b,k) or residual.shape!=(b,o):
        raise ValueError("route/residual geometry mismatch")
    s=state.double().reshape(b,k*d)
    A=(basis.double()*probabilities.double()[:,:,None,None]).permute(0,2,1,3).reshape(b,o,k*d)
    r=residual.double()
    gram=A@A.transpose(-1,-2)
    upper=gram.abs().sum(-1).amax(-1,keepdim=True)+ridge
    step=(eta/upper)*(A.transpose(-1,-2)@r.unsqueeze(-1)).squeeze(-1)
    inward=1-16*torch.finfo(state.dtype).eps
    norm=step.norm(dim=-1,keepdim=True)
    fraction=torch.clamp((step_limit*inward)/norm.clamp_min(1e-30),max=1.)
    alpha=(eta/upper)*fraction
    proposed=s+fraction*step
    size=proposed.norm(dim=-1,keepdim=True)
    projected=proposed*torch.clamp((radius*inward)/size.clamp_min(1e-30),max=1.)
    updated=projected.reshape_as(state).to(state.dtype)
    updated=torch.where((norm==0).reshape(b,1,1),state,updated)
    delta=updated.double().reshape(b,k*d)-s
    after=r-(A@delta.unsqueeze(-1)).squeeze(-1)
    before_loss=r.square().mean(-1)
    after_loss=after.square().mean(-1)
    info={"update_audit_active":torch.ones_like(before_loss),
          "observed_mse_before":before_loss,"observed_mse_after_fixed_context":after_loss,
          "observed_mse_change":after_loss-before_loss,
          "state_step_norm":delta.norm(dim=-1),"state_global_norm":updated.double().reshape(b,-1).norm(dim=-1),
          "lipschitz_upper":upper.squeeze(-1),"effective_step":alpha.squeeze(-1),
          "step_clipped":(fraction.squeeze(-1)<1).to(torch.float64),
          "state_projected":(size.squeeze(-1)>radius*inward).to(torch.float64),
          "residual_norm":r.norm(dim=-1)}
    return updated,info
