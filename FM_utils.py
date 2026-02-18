#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
from tqdm import tqdm


def _inference_steps_from_mask(mask, default_n):
    """Number of inference steps = number of tokens to generate (mask < 0.5). Max over batch."""
    if mask is None:
        return default_n
    n = (mask < 0.5).sum(dim=1).max().item()
    return max(1, int(n))


def CE_loss_logit(model,t,xt,x1):
    l1_hat = model(t,xt)
    loss = torch.nn.CrossEntropyLoss()(torch.einsum("b...i->bi...",l1_hat),x1.argmax(dim=-1))
    return loss
def CE_loss_logit_on_logits(model,t,xt,xtdt):
    ltdt_hat = model(t,xt)
    loss = torch.nn.CrossEntropyLoss()(
        torch.einsum("b...i->bi...",ltdt_hat),
        torch.einsum("b...i->bi...",xtdt)
    )
    return loss
def CE_loss_sphere(model,t,xt,x1):
    l1_hat = torch.log(torch.nn.functional.normalize(model(t,xt)).pow(2))
    loss = torch.nn.CrossEntropyLoss()(torch.einsum("b...i->bi...",l1_hat),x1.argmax(dim=-1))
    return loss


def int_quat(q0,q1,t):
    that = -torch.log(1-t[:,None,None])
    rho = torch.einsum("...t,...t->...",q0.float(),q1.float())[...,None]
    res = ((1+rho)*q1+2*torch.exp(-that)*(q0-q1*rho) + (rho-1)*q1*torch.exp(-2*that)) / (1+rho + (1-rho)*torch.exp(-2*that))
    return res

# class LogitEmbed(object):
#     def __init__(self,max_t=0.99,N=100,beta=0.01,method="sample1",dtype=torch.float,t_split=0.,embed_model="Qwen/Qwen2.5-0.5B-Instruct"):
#         self.beta = beta
#         self.max_that = -np.log(1-max_t)
#         self.max_t = max_t
#         self.N = N
#         self.step_size = self.max_that / N
#         self.method = method
#         self.dtype = dtype
#         self.t_split = t_split
#         model_gpt = AutoModelForCausalLM.from_pretrained(
#             embed_model,
#             torch_dtype="auto")
#         model_gpt.cpu()
#         self.embeds = model_gpt.model.embed_tokens.weight.data.float()
#         self.std_0 = self.embeds.pow(2).mean(dim=0).max().sqrt().item() * 2
#     def sampler_0(self,x,last_dim):
#         res = torch.randn(list(x.shape)+[last_dim],device=x.device) * self.std_0
#         return res
#     def interpolate(self,x0,x1,t=None,mask=None,return_logits=False):
#         if t is None:
#             t = torch.rand(x0.size(0),device=x0.device) * self.max_t
#         t = t.to(x0.dtype)
        
#         beta, N = self.beta, x0.size(-1)
#         x = (1-t[:,None,None]) * x0
#         x += self.embeds.to(x0.device)[x1] * t[:,None,None]
#         if mask is not None:
#             x[mask] = self.embeds.to(x0.device)[x1][mask]
#         return t,x
#     def rhs(self,model,that,xt,k=1):
#         t = 1-torch.exp(-that)
#         x1 = model(t,xt).softmax(dim=-1) # (B,T,V)
#         e1 = torch.einsum("btv,ve->bte",x1,self.embeds.to(xt.device))
#         return e1
#     def rhs_sample(self,model,that,xt,k=1):
#         t = 1-torch.exp(-that)
#         l1 = model(t,xt) # p(x1|xt), model = l1, x1 = softmax(l1)
#         if k == 1:
#             x1 = l1.argmax(dim=-1)
#         else:
#             vals,ids = l1.topk(k,dim=-1)
            
#             x1 = ids[torch.arange(ids.size(0))[:,None],
#                     torch.arange(ids.size(1))[None],
#                     torch.distributions.Categorical(logits=vals).sample()
#                    ]
        
#         return x1
#     @torch.no_grad()
#     def inference_mixed(self,model,x0, mask=None, k=1, t_split=None):
#         if t_split is None:
#             t_split = self.t_split
#         # thats = -torch.log(1-torch.linspace(0,self.max_t,self.N+1,device=x0.device))
#         thats = torch.linspace(0,self.max_that,self.N+1,device=x0.device)
#         if mask is None:
#             mask = torch.zeros_like(x0[...,0])
#         xt = x0.clone()
#         for that in tqdm(thats[:-1]):
#             t = 1 - torch.exp(-that)
#             if t < t_split:
#                 x1 = self.rhs(model,that[None].repeat(x0.size(0)), xt, k=-1)
#                 xt = torch.where(mask[...,None]>0.5,x0,(1-self.step_size)*xt + self.step_size*x1)
#             else:
#                 x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),xt,k=k)
#                 x0_ = self.sampler_0(x1,x0.size(-1))
#                 tdt = 1 - torch.exp(-that-self.step_size)[None].repeat(x1.size(0))
#                 xt = torch.where(mask[...,None]>0.5, x0, self.interpolate(x0_,x1,t=tdt)[1])
            
#         t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
#         x1_hat = x0 * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
#         return x1_hat
#     def inference(self,model,x0,k=1,mask=None,t_split=None):
#         return self.inference_mixed(model,x0,k=k,mask=mask,t_split=t_split)
import math
def compute_logV(log_xt, beta, t, device=None, dtype=None):
    """
    Compute log V_i efficiently.
    
    Inputs ----------------------------------------------------
      log_xt : [..., N]           # log x_t with arbitrary leading dimensions
      beta : scalar
      t : [B, ...] or scalar      # Time parameter, can be batched with different values per sample
      device, dtype      (optional)
    -----------------------------------------------------------
    Returns:
      logV : [..., N]
    """
    # Get the shape and extract N (the last dimension)
    shape = log_xt.shape
    N = shape[-1]
    
    # Reshape t to broadcast properly with log_xt if it's batched
    if isinstance(t, torch.Tensor) and t.dim() > 0:
        # Add necessary dimensions to match log_xt's trailing dimensions
        t_shape = list(t.shape) + [1] * (log_xt.dim() - t.dim())
        r = 1.0 - t.view(*t_shape)
        s = t.view(*t_shape)
    else:
        # Scalar case
        r, s = 1.0 - t, t
    logA = math.log(beta / N)
    logB = math.log(1 - beta + beta / N)

    # --------  log Y_k,   pivot m  -----------------------------------
    logY = (log_xt - s * logA) / r             # [B, N]
    m     = logY.max(dim=-1, keepdim=True).values

    # --------  Ybar_k,    base  --------------------------------------
    Ybar  = torch.exp(logY - m)                # [B, N]
    base  = Ybar.sum(dim=-1, keepdim=True)     # [B, 1]
    log_base = torch.log(base)                 # [B, 1]

    # --------  helper logs  ------------------------------------------
    u        = Ybar / base                     # in (0,1]
    log_u    = torch.log(u)                    # [B, N]
    log1minus_u = torch.log1p(-u)              # log(1-u)  [B, N]

    # --------  log γ  (64-bit prevents under-flow) -------------------
    log_gamma = ((s / r) * (logA - logB))  # scalar, very negative when t→1
    # keep it in fp64 up to here
    log_gamma = log_gamma.float()                   # back to fp32 for the mix

    # --------  log V_i  via logaddexp path ---------------------------
    log_term = torch.logaddexp(
                log1minus_u,               # log(1-u)
                log_gamma + log_u)         # log γ + log u

    logV = m + log_base + log_term           # [B, N]
    return logV
def compute_logV(log_xt, beta, t, device=None, dtype=None):
    """
    Compute log V efficiently.
    """
    N = log_xt.shape[-1]
    t_reshaped = t
    if t.dim() < log_xt.dim():
        t_reshaped = t.view(*t.shape, *([1] * (log_xt.dim() - t.dim())))
    logV = -N * torch.log(t_reshaped)
    return logV
def compute_log_posterior(log_xt, beta, t, log_p, device=None, dtype=None):
    """
    Compute log p(x | xt) efficiently.
    
    Inputs ----------------------------------------------------
      log_xt : [..., N]         # log x_t with arbitrary leading dimensions
      beta : scalar
      t : [B, ...]             # Time parameter, can be batched with different values per sample
      log_p : [..., N]          # log prior probabilities (logits) with same shape as log_xt
      device, dtype      (optional)
    -----------------------------------------------------------
    Returns:
      log_posterior : [..., N]  # log p(x | xt)
    """

    N = log_xt.shape[-1]
    
    # Reshape t to broadcast properly with log_xt
    t_reshaped = t
    if t.dim() < log_xt.dim():
        # Add necessary dimensions to match log_xt's leading dimensions
        t_reshaped = t.view(*t.shape, *([1] * (log_xt.dim() - t.dim())))
    
    # Compute log V using the existing function with batched t
    logV = compute_logV(log_xt, beta, t_reshaped, device, dtype)
    
    # Normalize the log prior probabilities using log_softmax
    log_p_normalized = torch.nn.functional.log_softmax(log_p, dim=-1)
    
    # Compute log p(x | xt) = -N * log V + log p
    log_posterior = -N * logV + log_p_normalized
    
    return log_posterior
class Logit(object):
    def __init__(self, config, tokenizer=None):
        self.beta = config.get("beta", 0.01)
        self.max_t = config.get("max_t", 0.3)
        self.max_that = -np.log(1-self.max_t)
        self.N = config.get("N", 100)
        self.step_size = self.max_that / self.N
        self.method = config.get("method", "sample1")
        self.dtype = config.get("dtype", torch.float)
        self.t_split = config.get("t_split", 0.)
        
        # Initialize vocab_size from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)
    
    def update_N(self,N):
        self.N = N
        self.step_size = self.max_that / N
    
    def sampler_0(self,x):
        dst = torch.distributions.Dirichlet(torch.ones(self.vocab_size,device=x.device))
        res = dst.sample(sample_shape=x.shape)
        return res
    def interpolate(self,x0,x1,t=None,mask=None,return_logits=False):
        if t is None:
            t = torch.rand(x0.size(0),device=x0.device) * self.max_t
        t = t.to(x0.dtype)
        
        beta, N = self.beta, x0.size(-1)
        x = (1-t[:,None,None]) * torch.log(x0)
        x += t[:,None,None] * np.log(beta/N)
        x[torch.arange(x1.size(0),device=x1.device)[:,None],torch.arange(x1.size(1),device=x1.device)[None,:],x1] += t[:,None] * np.log(N/beta - N + 1)
        xt = x.softmax(dim=-1)
        if mask is not None:
            xt[mask] = torch.nn.functional.one_hot(x1[mask],x0.size(-1)).to(xt.dtype)
        if return_logits:
            return t,xt,x.log_softmax(dim=-1)
        else:
            return t,xt
    def rhs(self,model,that,xt,k=1):
        t = 1-torch.exp(-that)
        x1 = model(t,xt).softmax(dim=-1)
        l1 = x1 * np.log(1-self.beta+self.beta/xt.size(-1)) + (1-x1) * np.log(self.beta/xt.size(-1))
        return l1
    def rhs_sample(self,model,that,xt,k=1):
        t = 1-torch.exp(-that)
        l1 = model(t,xt) # p(x1|xt), model = l1, x1 = softmax(l1)
        if k == 1:
            x1 = l1.argmax(dim=-1)
        else:
            vals,ids = l1.topk(k,dim=-1)
            
            x1 = ids[torch.arange(ids.size(0))[:,None],
                    torch.arange(ids.size(1))[None],
                    torch.distributions.Categorical(logits=vals).sample()
                   ]
        return x1
            
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=None):
        if t_split is None:
            t_split = self.t_split
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_local = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),torch.log(x0))
        xt = x0.clone()
        for that in tqdm(thats[:-1]):
            t = 1- torch.exp(-that)
            if t < t_split:
                l1 = self.rhs(model,that[None].repeat(x0.size(0)), xt, k=-1)
                lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),(1-step_size_local)*lt + step_size_local*l1)
                xt =  lt.softmax(dim=-1)*(1-mask[...,None]) + x0*mask[...,None]
            else:
                x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),xt,k=k)
                x0_ = self.sampler_0(x1)
                tdt = 1 - torch.exp(-that-step_size_local)[None].repeat(x1.size(0))
                xt = torch.where(mask[...,None]>0.5, x0, self.interpolate(x0_,x1,t=tdt)[1])
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = x0 * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
        return x1_hat
    def inference_prefix(self,model,x0, prefix, k=1,mask=None,t_split=None):
        if t_split is None:
            t_split = self.t_split
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_local = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),torch.log(x0))
        xt = x0.clone()
        for that in tqdm(thats[:-1]):
            t = 1- torch.exp(-that)
            if t < t_split:
                l1 = self.rhs(model,that[None].repeat(x0.size(0)), xt, k=-1)
                prefix_logits = prefix * np.log(1-self.beta+self.beta/self.vocab_size) + (1-prefix) * np.log(self.beta/self.vocab_size)
                l1 = torch.where(mask[...,None]>0.5,prefix_logits,l1)
                lt = (1-step_size_local)*lt + step_size_local*l1
                xt =  lt.softmax(dim=-1)
            else:
                x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),xt,k=k)
                prefix_probs = prefix.argmax(dim=-1)
                x1 = torch.where(mask>0.5,prefix_probs,x1)
                x0_ = self.sampler_0(x1)
                tdt = 1 - torch.exp(-that-step_size_local)[None].repeat(x1.size(0))
                xt = self.interpolate(x0_,x1,t=tdt)[1]
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = prefix * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
        return x1_hat
    def inference(self,model,x0,k=1,mask=None,t_split=None):
        return self.inference_mixed(model,x0,k=k,mask=mask,t_split=t_split)
    
class LogitEntropy(object):
    def __init__(self, config, tokenizer=None):
        self.beta = config.get("beta", 0.01)
        self.max_t = config.get("max_t", 0.3)
        self.max_that = -np.log(1-self.max_t)
        self.N = config.get("N", 100)
        self.step_size = self.max_that / self.N
        self.method = config.get("method", "sample1")
        self.dtype = config.get("dtype", torch.float)
        self.t_split = config.get("t_split", 0.)
        
        # Initialize vocab_size from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)
    
    def update_N(self,N):
        self.N = N
        self.step_size = self.max_that / N
    
    def sampler_0(self,x):
        dst = torch.distributions.Dirichlet(torch.ones(self.vocab_size,device=x.device))
        res = dst.sample(sample_shape=x.shape)
        return res
    def interpolate(self,x0,x1,t=None,mask=None,return_logits=False):
        if t is None:
            t = torch.rand(x0.size(0),device=x0.device) * self.max_t
        t = t.to(x0.dtype)
        
        beta, N = self.beta, x0.size(-1)
        x = (1-t[:,None,None]) * torch.log(x0)
        x += t[:,None,None] * np.log(beta/N)
        x[torch.arange(x1.size(0),device=x1.device)[:,None],torch.arange(x1.size(1),device=x1.device)[None,:],x1] += t[:,None] * np.log(N/beta - N + 1)
        xt = x.softmax(dim=-1)
        if mask is not None:
            xt[mask] = torch.nn.functional.one_hot(x1[mask],x0.size(-1)).to(xt.dtype)
        if return_logits:
            return t,xt,x.log_softmax(dim=-1)
        else:
            return t,xt
    def rhs(self,model,that,xt,k=1):
        t = 1-torch.exp(-that)
        x1 = model(t,xt).softmax(dim=-1)
        l1 = x1 * np.log(1-self.beta+self.beta/xt.size(-1)) + (1-x1) * np.log(self.beta/xt.size(-1))
        return l1
    def rhs_sample(self,model,that,xt,k=1):
        t = 1-torch.exp(-that)
        l1 = model(t,xt) # p(x1|xt), model = l1, x1 = softmax(l1)
        if k == 1:
            x1 = l1.argmax(dim=-1)
        else:
            vals,ids = l1.topk(k,dim=-1)
            
            x1 = ids[torch.arange(ids.size(0))[:,None],
                    torch.arange(ids.size(1))[None],
                    torch.distributions.Categorical(logits=vals).sample()
                   ]
        return x1
            
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=None):
        def calc_entropy(logits):
            return (-logits.log_softmax(dim=-1) * logits.softmax(dim=-1)).sum(dim=-1)
        if t_split is None:
            t_split = self.t_split
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_local = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),torch.log(x0))
        xt = x0.clone()
        for that in tqdm(thats[:-1]):
            t = 1- torch.exp(-that)
            if t < t_split:
                l1 = self.rhs(model,that[None].repeat(x0.size(0)), xt, k=-1)
                lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),(1-step_size_local)*lt + step_size_local*l1)
                xt =  lt.softmax(dim=-1)*(1-mask[...,None]) + x0*mask[...,None]
            else:
                x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),xt,k=k)
                x0_ = self.sampler_0(x1)
                tdt = 1 - torch.exp(-that-step_size_local)[None].repeat(x1.size(0))
                xt = torch.where(mask[...,None]>0.5, x0, self.interpolate(x0_,x1,t=tdt)[1])
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = x0 * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
        return x1_hat
    def inference_prefix(self,model,x0, prefix, k=1,mask=None,t_split=None):
        if t_split is None:
            t_split = self.t_split
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_local = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),torch.log(x0))
        xt = x0.clone()
        for that in tqdm(thats[:-1]):
            t = 1- torch.exp(-that)
            if t < t_split:
                l1 = self.rhs(model,that[None].repeat(x0.size(0)), xt, k=-1)
                prefix_logits = prefix * np.log(1-self.beta+self.beta/self.vocab_size) + (1-prefix) * np.log(self.beta/self.vocab_size)
                l1 = torch.where(mask[...,None]>0.5,prefix_logits,l1)
                lt = (1-step_size_local)*lt + step_size_local*l1
                xt =  lt.softmax(dim=-1)
            else:
                x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),xt,k=k)
                prefix_probs = prefix.argmax(dim=-1)
                x1 = torch.where(mask>0.5,prefix_probs,x1)
                x0_ = self.sampler_0(x1)
                tdt = 1 - torch.exp(-that-step_size_local)[None].repeat(x1.size(0))
                xt = self.interpolate(x0_,x1,t=tdt)[1]
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = prefix * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
        return x1_hat

def compute_logV(log_xt, beta, t, device=None, dtype=None):
    """
    Compute log V_i efficiently.
    
    Inputs ----------------------------------------------------
      log_xt : [..., N]           # log x_t with arbitrary leading dimensions
      beta : scalar
      t : [B, ...] or scalar      # Time parameter, can be batched with different values per sample
      device, dtype      (optional)
    -----------------------------------------------------------
    Returns:
      logV : [..., N]
    """
    # Get the shape and extract N (the last dimension)
    shape = log_xt.shape
    N = shape[-1]
    
    # Reshape t to broadcast properly with log_xt if it's batched
    if isinstance(t, torch.Tensor) and t.dim() > 0:
        # Add necessary dimensions to match log_xt's trailing dimensions
        t_shape = list(t.shape) + [1] * (log_xt.dim() - t.dim())
        r = 1.0 - t.view(*t_shape)
        s = t.view(*t_shape)
    else:
        # Scalar case
        r, s = 1.0 - t, t
    logA = math.log(beta / N)
    logB = math.log(1 - beta + beta / N)

    # --------  log Y_k,   pivot m  -----------------------------------
    logY = (log_xt - s * logA) / r             # [B, N]
    m     = logY.max(dim=-1, keepdim=True).values

    # --------  Ybar_k,    base  --------------------------------------
    Ybar  = torch.exp(logY - m)                # [B, N]
    base  = Ybar.sum(dim=-1, keepdim=True)     # [B, 1]
    log_base = torch.log(base)                 # [B, 1]

    # --------  helper logs  ------------------------------------------
    u        = Ybar / base                     # in (0,1]
    log_u    = torch.log(u)                    # [B, N]
    log1minus_u = torch.log1p(-u)              # log(1-u)  [B, N]

    # --------  log γ  (64-bit prevents under-flow) -------------------
    log_gamma = ((s / r) * (logA - logB))  # scalar, very negative when t→1
    # keep it in fp64 up to here
    log_gamma = log_gamma.float()                   # back to fp32 for the mix

    # --------  log V_i  via logaddexp path ---------------------------
    log_term = torch.logaddexp(
                log1minus_u,               # log(1-u)
                log_gamma + log_u)         # log γ + log u

    logV = m + log_base + log_term           # [B, N]
    return logV.to(log_xt.dtype)


def compute_log_posterior(log_xt, beta, t, log_p, device=None, dtype=None):
    """
    Compute log p(x | xt) efficiently.
    
    Inputs ----------------------------------------------------
      log_xt : [..., N]         # log x_t with arbitrary leading dimensions
      beta : scalar
      t : [B, ...]             # Time parameter, can be batched with different values per sample
      log_p : [..., N]          # log prior probabilities (logits) with same shape as log_xt
      device, dtype      (optional)
    -----------------------------------------------------------
    Returns:
      log_posterior : [..., N]  # log p(x | xt)
    """

    N = log_xt.shape[-1]
    
    # Reshape t to broadcast properly with log_xt
    t_reshaped = t
    if t.dim() < log_xt.dim():
        # Add necessary dimensions to match log_xt's leading dimensions
        t_reshaped = t.view(*t.shape, *([1] * (log_xt.dim() - t.dim())))
    
    # Compute log V using the existing function with batched t
    logV = compute_logV(log_xt, beta, t_reshaped, device, dtype)
    
    # Normalize the log prior probabilities using log_softmax
    log_p_normalized = torch.nn.functional.log_softmax(log_p, dim=-1)
    
    # Compute log p(x | xt) = -N * log V + log p
    log_posterior = -N * logV + log_p_normalized
    
    return log_posterior

class LogitUpdated(object):
    def __init__(self, config, tokenizer=None):
        self.beta = config.get("beta", 0.01)
        self.max_t = config.get("max_t", 0.3)
        self.max_that = -np.log(1-self.max_t)
        self.N = config.get("N", 100)
        self.step_size = self.max_that / self.N
        self.method = config.get("method", "sample1")
        self.dtype = config.get("dtype", torch.float)
        self.t_split = config.get("t_split", 0.)
        
        # Initialize vocab_size from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)
    
    def update_N(self,N):
        self.N = N
        self.step_size = self.max_that / N
    
    def sampler_0(self,x):
        dst = torch.distributions.Dirichlet(torch.ones(self.vocab_size,device=x.device))
        res = dst.sample(sample_shape=x.shape)
        return res
    def interpolate(self,x0,x1,t=None,mask=None):
        if t is None:
            t = torch.rand(x0.size(0),device=x0.device) * self.max_t
        t = t.to(x0.dtype)
        
        beta, N = self.beta, x0.size(-1)
        x = (1-t[:,None,None]) * torch.log(x0)
        x += t[:,None,None] * np.log(beta/N)
        x[torch.arange(x1.size(0),device=x1.device)[:,None],torch.arange(x1.size(1),device=x1.device)[None,:],x1] += t[:,None] * np.log(N/beta - N + 1)
        xt = x.softmax(dim=-1)
        if mask is not None:
            xt[mask] = torch.nn.functional.one_hot(x1[mask],x0.size(-1)).to(xt.dtype)
        lt = x.log_softmax(dim=-1)
        return t,lt,xt
    
    def rhs(self,model,that,lt,xt):
        t = 1-torch.exp(-that)
        l1 = model(t,lt,xt)
        return l1.softmax(dim=-1)
    def rhs_sample(self,model,that,lt,xt,k=1):
        t = 1-torch.exp(-that)
        l1 = model(t,lt,xt)
        if k == 1:
            x1 = l1.argmax(dim=-1)
        else:
            vals,ids = l1.topk(k,dim=-1)
            
            x1 = ids[
                torch.arange(ids.size(0))[:,None],
                torch.arange(ids.size(1))[None],
                torch.distributions.Categorical(logits=vals).sample()
            ]
        return x1
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=None):
        if t_split is None:
            t_split = self.t_split
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_local = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),torch.log(x0))
        xt = x0.clone()
        for that in tqdm(thats[:-1]):
            t = 1- torch.exp(-that)
            if t < t_split:
                l1 = self.rhs(model,that[None].repeat(x0.size(0)), lt, xt)    
                lt = ((1-step_size_local)*lt + step_size_local*l1).log_softmax(dim=-1)
                xt_ =  lt.softmax(dim=-1)
                xt = torch.where(mask[...,None]>0.5, x0, xt_)
            else:
                x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),lt,xt,k=k)
                x0_ = self.sampler_0(x1)
                tdt = 1 - torch.exp(-that-step_size_local)[None].repeat(x1.size(0))
                _, lt_, xt_ = self.interpolate(x0_,x1,t=tdt)
                xt = torch.where(mask[...,None]>0.5, x0, xt_)
                lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),lt_)
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = x0 * mask[...,None] + model(t_max,lt,xt).softmax(dim=-1) * (1-mask[...,None])
        return x1_hat
    def inference_prefix(self,model,x0, prefix, k=1,mask=None,t_split=None):
        if t_split is None:
            t_split = self.t_split
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_local = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),torch.log(x0))
        xt = x0.clone()
        for that in tqdm(thats[:-1]):
            t = 1- torch.exp(-that)
            if t < t_split:
                l1 = self.rhs(model,that[None].repeat(x0.size(0)), lt, xt)
                prefix_logits = prefix * np.log(1-self.beta+self.beta/self.vocab_size) + (1-prefix) * np.log(self.beta/self.vocab_size)
                l1 = torch.where(mask[...,None]>0.5,prefix_logits,l1)
                lt = (1-step_size_local)*lt + step_size_local*l1
                xt =  lt.softmax(dim=-1)
            else:
                x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),lt,xt,k=k)
                prefix_probs = prefix.argmax(dim=-1)
                x1 = torch.where(mask>0.5,prefix_probs,x1)
                x0_ = self.sampler_0(x1)
                tdt = 1 - torch.exp(-that-step_size_local)[None].repeat(x1.size(0))
                _, lt_, xt_ = self.interpolate(x0_,x1,t=tdt)
                xt = torch.where(mask[...,None]>0.5, x0, xt_)
                lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),lt_)
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = prefix * mask[...,None] + model(t_max,lt,xt).softmax(dim=-1) * (1-mask[...,None])
        return x1_hat
    def inference(self,model,x0,k=1,mask=None,t_split=None):
        return self.inference_mixed(model,x0,k=k,mask=mask,t_split=t_split)
    

class GPT(object):
    def __init__(self, config, tokenizer=None):
        self.t_split = config.get("t_split", 0.)
        
        # Initialize vocab_size from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)
    
    def sampler_0(self,x):
        x0 = x.clone()
        return x0
    def update_N(self,N):
        pass
    def interpolate(self,x0,x1,t=None,mask=None,return_logits=False):
        t = torch.zeros(x0.size(0),device=x0.device)
        return t,x0
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=None):
        x1 = x0.clone()
        t = torch.zeros(x1.size(0),device=x1.device)
        zero_indices = torch.nonzero((mask < 0.5).sum(dim=0), as_tuple=True)[0]
        start_idx, end_idx = zero_indices[0].item(), zero_indices[-1].item()
        for i in tqdm(range(start_idx-1, end_idx)):
            new_token = torch.distributions.Categorical(logits=model(t,x1)[:,i,:self.vocab_size]).sample()
            x1[:,i+1] = torch.where(mask[:,i+1]>0.5,x0[:,i+1],new_token)
        return x1
    def inference(self,model,x0,k=1,mask=None,t_split=None):
        return self.inference_mixed(model,x0,k=k,mask=mask,t_split=t_split)


class OneShot(object):
    def __init__(self, config, tokenizer=None):
        self.beta = config.get("beta", 0.01)
        self.max_t = config.get("max_t", 0.3)
        self.max_that = -np.log(1-self.max_t)
        self.N = config.get("N", 100)
        self.step_size = self.max_that / self.N
        self.method = config.get("method", "sample1")
        self.dtype = config.get("dtype", torch.float)
        self.t_split = config.get("t_split", 0.)
        
        # Initialize vocab_size from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)
    
    def sampler_0(self,x):
        dst = torch.distributions.Dirichlet(torch.ones(self.vocab_size,device=x.device))
        res = dst.sample(sample_shape=x.shape)
        return res
    def interpolate(self,x0,x1,t=None,mask=None,return_logits=False):
        # if t is None:
        t = torch.ones(x0.size(0),device=x0.device) * self.max_t
        # t = t.to(x0.dtype)
        
        beta, N = self.beta, x0.size(-1)
        x = (1-t[:,None,None]) * torch.log(x0)
        x += t[:,None,None] * np.log(beta/N)
        x[torch.arange(x1.size(0),device=x1.device)[:,None],torch.arange(x1.size(1),device=x1.device)[None,:],x1] += t[:,None] * np.log(N/beta - N + 1)
        xt = x.softmax(dim=-1)
        if mask is not None:
            xt[mask] = torch.nn.functional.one_hot(x1[mask],x0.size(-1)).to(xt.dtype)
        if return_logits:
            return t,xt,x.log_softmax(dim=-1)
        else:
            return t,xt
    def rhs(self,model,that,xt,k=1):
        t = 1-torch.exp(-that)
        x1 = model(t,xt).softmax(dim=-1)
        l1 = x1 * np.log(1-self.beta+self.beta/xt.size(-1)) + (1-x1) * np.log(self.beta/xt.size(-1))
        return l1
    def rhs_sample(self,model,that,xt,k=1):
        t = 1-torch.exp(-that)
        l1 = model(t,xt) # p(x1|xt), model = l1, x1 = softmax(l1)
        if k == 1:
            x1 = l1.argmax(dim=-1)
        else:
            vals,ids = l1.topk(k,dim=-1)
            
            x1 = ids[torch.arange(ids.size(0))[:,None],
                    torch.arange(ids.size(1))[None],
                    torch.distributions.Categorical(logits=vals).sample()
                   ]
        return x1
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=None):
        if t_split is None:
            t_split = self.t_split
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_local = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),torch.log(x0))
        xt = x0.clone()
        for that in tqdm(thats[:-1]):
            t = 1- torch.exp(-that)
            if t < t_split:
                l1 = self.rhs(model,that[None].repeat(x0.size(0)), xt, k=-1)
                lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),(1-step_size_local)*lt + step_size_local*l1)
                xt =  lt.softmax(dim=-1)*(1-mask[...,None]) + x0*mask[...,None]
            else:
                x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),xt,k=k)
                x0_ = self.sampler_0(x1)
                tdt = 1 - torch.exp(-that-step_size_local)[None].repeat(x1.size(0))
                xt = torch.where(mask[...,None]>0.5, x0, self.interpolate(x0_,x1,t=tdt)[1])
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = x0 * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
        return x1_hat
    def inference(self,model,x0,k=1,mask=None,t_split=None):
        return self.inference_mixed(model,x0,k=k,mask=mask,t_split=t_split)


class LogitMask(object):
    def __init__(self, config, tokenizer=None):
        self.beta = config.get("beta", 0.01)
        self.max_t = config.get("max_t", 0.3)
        self.max_that = -np.log(1-self.max_t)
        self.N = config.get("N", 100)
        self.step_size = self.max_that / self.N
        self.method = config.get("method", "sample1")
        self.dtype = config.get("dtype", torch.float)
        self.t_split = config.get("t_split", 0.)
        
        # Initialize vocab_size and mask_token_id from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
            self.mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None else self.vocab_size - 1
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)
            self.mask_token_id = self.vocab_size - 1
    
    def sampler_0(self,x):
        dst = torch.distributions.Dirichlet(torch.ones(self.vocab_size,device=x.device))
        res = dst.sample(sample_shape=x.shape) * 0.1
        res[...,self.mask_token_id] += 0.9
        return res
    def interpolate(self,x0,x1,t=None,mask=None,return_logits=False):
        if t is None:
            t = torch.rand(x0.size(0),device=x0.device) * self.max_t
        t = t.to(x0.dtype)
        
        beta, N = self.beta, x0.size(-1)
        x = (1-t[:,None,None]) * torch.log(x0)
        x += t[:,None,None] * np.log(beta/N)
        x[torch.arange(x1.size(0),device=x1.device)[:,None],torch.arange(x1.size(1),device=x1.device)[None,:],x1] += t[:,None] * np.log(N/beta - N + 1)
        xt = x.softmax(dim=-1)
        if mask is not None:
            xt[mask] = torch.nn.functional.one_hot(x1[mask],x0.size(-1)).to(xt.dtype)
        if return_logits:
            return t,xt,x.log_softmax(dim=-1)
        else:
            return t,xt
    def rhs(self,model,that,xt,k=1):
        t = 1-torch.exp(-that)
        x1 = model(t,xt).softmax(dim=-1)
        l1 = x1 * np.log(1-self.beta+self.beta/xt.size(-1)) + (1-x1) * np.log(self.beta/xt.size(-1))
        return l1
    def rhs_sample(self,model,that,xt,k=1):
        t = 1-torch.exp(-that)
        l1 = model(t,xt) # p(x1|xt), model = l1, x1 = softmax(l1)
        if k == 1:
            x1 = l1.argmax(dim=-1)
        else:
            vals,ids = l1.topk(k,dim=-1)
            
            x1 = ids[torch.arange(ids.size(0))[:,None],
                    torch.arange(ids.size(1))[None],
                    torch.distributions.Categorical(logits=vals).sample()
                   ]
        return x1
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=None):
        if t_split is None:
            t_split = self.t_split
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_local = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),torch.log(x0))
        xt = x0.clone()
        for that in tqdm(thats[:-1]):
            t = 1- torch.exp(-that)
            if t < t_split:
                l1 = self.rhs(model,that[None].repeat(x0.size(0)), xt, k=-1)
                lt = torch.where(mask[...,None]>0.5,torch.zeros_like(x0),(1-step_size_local)*lt + step_size_local*l1)
                xt =  lt.softmax(dim=-1)*(1-mask[...,None]) + x0*mask[...,None]
            else:
                x1 = self.rhs_sample(model,that[None].repeat(x0.size(0)),xt,k=k)
                x0_ = self.sampler_0(x1)
                tdt = 1 - torch.exp(-that-step_size_local)[None].repeat(x1.size(0))
                xt = torch.where(mask[...,None]>0.5, x0, self.interpolate(x0_,x1,t=tdt)[1])
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = x0 * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
        return x1_hat
    def inference(self,model,x0,k=1,mask=None,t_split=None):
        return self.inference_mixed(model,x0,k=k,mask=mask,t_split=t_split)

# %%
class DFM(object):
    def __init__(self, config, tokenizer=None):
        self.max_t = config.get("max_t", 1.)
        self.N = config.get("N", 100)
        self.step_size = self.max_t / self.N
        self.mode0 = config.get("mode0", "mask")
        self.beta = config.get("beta", 0.01)
        self.t_split = config.get("t_split", None)
        assert self.mode0 == "uniform" or self.mode0 == "mask", "invalid mode for zero sampler"
        
        # Initialize vocab_size and mask_token_id from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
            self.mask_token_id = tokenizer.mask_token_id 
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)
            self.mask_token_id = self.vocab_size - 1
    
    def update_N(self,N):
        self.N = N
        self.step_size = self.max_t / N
    
    def sampler_0(self,x):
        if self.mode0 == "uniform":
            dst = torch.distributions.Categorical(probs=torch.ones(self.vocab_size,device=x.device))
            return torch.nn.functional.one_hot(dst.sample(sample_shape=x.shape),self.vocab_size).float()
        elif self.mode0 == "mask":
            r = torch.zeros(list(x.shape)+[self.vocab_size],device=x.device,dtype=torch.float)
            r[:,:,self.mask_token_id] = 1.
            return r
    def scheduler(self,t):
        return 2*t - t.pow(2), 2 - 2 * t
    
    def interpolate(self,x0,x1,t=None,mask=None):
        if t is None:
            t = torch.rand(x0.size(0),device=x0.device) * self.max_t
        kt,dkt = self.scheduler(t)
        pt = torch.einsum("b...,b->b...",x0,1-kt)
        pt[torch.arange(x1.size(0),device=x1.device)[:,None],
           torch.arange(x1.size(1),device=x1.device)[None],
           x1] += kt[:,None]
        pt = pt / pt.sum(dim=-1,keepdim=True)
        xt = torch.distributions.Categorical(probs=pt).sample()
        del(pt)
        xt = torch.nn.functional.one_hot(xt,x0.size(-1)).float()
        if mask is not None:
            xt[mask] = torch.nn.functional.one_hot(x1[mask],x0.size(-1)).to(xt.dtype)
        return t,xt
    def rhs_base(self,model,t,x0,xt,mask=None):
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        l1 = model(t,xt).log_softmax(dim=-1)
        kt,dkt = self.scheduler(t)
        C = (dkt / (1 - kt) * self.step_size).clamp(0,1)
        lt = torch.where(mask[...,None] > 0.5,torch.zeros_like(xt),torch.log(xt))
        
        ltdt = torch.cat([
            lt[...,None] + torch.log(1-C)[...,None,None,None],
            l1[...,None] + torch.log(C)[...,None,None,None]
        ],dim=-1).logsumexp(dim=-1)
        
        xtdt = torch.where(mask[...,None]>0.5,x0,torch.nn.functional.one_hot(torch.distributions.Categorical(logits=ltdt).sample(),xt.size(-1)).float())
        return xtdt
    @torch.no_grad()
    def inference(self,model,x0,k=1,mask=None):
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_save = self.step_size
        self.step_size = self.max_t / n_steps
        ts = torch.linspace(0,self.max_t,n_steps+1,device=x0.device)
        xt = x0.clone()
        for t in tqdm(ts[:-1]):
            xt = self.rhs_base(model,t[None].repeat(x0.size(0)),x0,xt,mask=mask)
        self.step_size = step_size_save
        return xt
    def rhs_sample(self,model,t,x0,xt,mask=None,k=1):
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        l1 = model(t,xt) # p(x1|xt), model = l1, x1 = softmax(l1)
        vals,ids = l1.topk(k,dim=-1)
        
        x1 = ids[
            torch.arange(ids.size(0))[:,None],
            torch.arange(ids.size(1))[None],
            torch.distributions.Categorical(logits=vals).sample()
        ]
        x0_ = self.sampler_0(x1)
        tdt = t + self.step_size
        xtdt = torch.where(mask[...,None]>0.5, x0, self.interpolate(x0_,x1,t=tdt)[1])
        return xtdt
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=0.25):
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_save = self.step_size
        self.step_size = self.max_t / n_steps
        ts = torch.linspace(0,self.max_t,n_steps+1,device=x0.device)
        xt = x0.clone()
        for t in tqdm(ts[:-1]):
            xt = self.rhs_base(model, t[None].repeat(x0.size(0)), x0, xt)
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = x0 * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
        self.step_size = step_size_save
        return x1_hat
    def inference_prefix(self,model,x0, prefix, k=1,mask=None,t_split=None):
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_save = self.step_size
        self.step_size = self.max_t / n_steps
        ts = torch.linspace(0,self.max_t,n_steps+1,device=x0.device)
        xt = x0.clone()
        for t in tqdm(ts[:-1]):
            xt = self.rhs_base(model, t[None].repeat(x0.size(0)), x0, xt)
            xt = torch.where(mask[...,None]>0.5,prefix,xt)
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = prefix * mask[...,None] + model(t_max,xt).softmax(dim=-1) * (1-mask[...,None])
        self.step_size = step_size_save
        return x1_hat


# %%
        
class Sphere(object):
    def __init__(self, config, tokenizer=None):
        self.max_t = config.get("max_t", 0.99)
        self.max_that = -np.log(1-self.max_t)
        self.N = config.get("N", 100)
        self.step_size = self.max_that / self.N
        
        # Initialize vocab_size from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)
    
    def sampler_0(self,x):
        dst = torch.distributions.Dirichlet(torch.ones(self.vocab_size,device=x.device))
        res = dst.sample(sample_shape=x.shape)
        return res
    def interpolate(self,x0,x1,t=None):
        if t is None:
            t = torch.rand(x0.size(0),device=x0.device) * self.max_t
        q0 = torch.sqrt(x0)
        q1 = torch.nn.functional.one_hot(x1,x0.size(-1)).float()
        qt = int_quat(q0,q1,t)
        return t,qt
    def rhs(self,model,t,q0,qt,mask=None):
        xt = qt.pow(2)
        q1 = torch.nn.functional.normalize(model(t,xt).abs(),dim=-1)
        qtdt = qt + (q1 - qt * torch.einsum("...i,...i->...",qt,q1)[...,None]) * self.step_size
        qtdt = torch.nn.functional.normalize(qtdt,dim=-1)
        return qtdt
    def rhs_sample(self,model,t,q0,qt,mask=None,k=1):
        tdt = 1 - torch.exp(torch.log(1-t) - self.step_size)
        xt = qt.pow(2)
        q1 = torch.nn.functional.normalize(model(t,xt).abs(),dim=-1)
        vals,ids = q1.topk(k,dim=-1)
        
        x1 = ids[
            torch.arange(ids.size(0))[:,None],
            torch.arange(ids.size(1))[None],
            torch.distributions.Categorical(logits=vals).sample()
        ]
        q1 = torch.nn.functional.one_hot(x1,qt.size(-1)).sqrt()
        q0_ = self.sampler_0(qt[...,0]).sqrt()
        qtdt = int_quat(q0_,q1,tdt)
        return qtdt
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=0.25):
        if mask is None:
            mask = torch.zeros_like(x0[...,0])
        n_steps = _inference_steps_from_mask(mask, self.N)
        step_size_save = self.step_size
        self.step_size = self.max_that / n_steps
        thats = torch.linspace(0,self.max_that,n_steps+1,device=x0.device)
        q0 = x0.sqrt()
        qt = q0.clone()
        for that in tqdm(thats[:-1]):
            t = 1 - torch.exp(-that)
            if t < t_split:
                qt = self.rhs(model, t[None].repeat(x0.size(0)), q0, qt)
            else:
                qt = self.rhs_sample(model, t[None].repeat(x0.size(0)), q0, qt, k=k)
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = torch.nn.functional.normalize(model(t_max,qt.pow(2)).abs(),dim=-1)
        self.step_size = step_size_save
        return x1_hat
import scipy
import scipy.special
class Dirichlet(object):
    def __init__(self, config, tokenizer=None):
        self.max_t = config.get("max_t", 20.)
        self.N = config.get("N", 100)
        self.K = config.get("K", 20)
        self.step_size = (self.max_t - 1) / self.N
        self.alphas = np.arange(1, self.max_t + 0.01, 0.01)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, self.K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / 0.01
        
        # Initialize vocab_size from tokenizer
        if tokenizer is not None:
            self.vocab_size = len(tokenizer)
        else:
            # Fallback to config if tokenizer not provided
            self.vocab_size = config.get("vocab_size", 50257)

    def c_factor(self, bs, alpha):
        bs = bs.cpu().numpy()
        out1 = scipy.special.beta(alpha, self.K - 1)
        out2 = np.where(bs < 1, out1 / ((1 - bs) ** (self.K - 1)), 0)
        out = np.where((bs ** (alpha - 1)) > 0, out2 / (bs ** (alpha - 1)), 0)
        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)
        final = interp * out
        return (torch.from_numpy(final).to(torch.float))

    def sampler_0(self,x):
        dst = torch.distributions.Dirichlet(torch.ones(self.vocab_size,device=x.device))
        res = dst.sample(sample_shape=x.shape)
        return res
    def interpolate(self,x0,x1,t=None):
        if t is None:
            t = 1-torch.log(1-torch.rand(x0.size(0),device=x0.device)*0.9999) * 5
        alphas = t
        alphas_ = torch.ones(x0.size(0), x0.size(1), x0.size(2), device=x0.device)
        alphas_ = alphas_ + torch.nn.functional.one_hot(x1,x0.size(-1)) * (alphas[:,None,None] - 1)
        xt = torch.distributions.Dirichlet(alphas_).sample()
        return alphas,xt
    def rhs(self,model,t,xt):
        flow_probs = model(t,xt).softmax(dim=-1)
        c_factor = self.c_factor(xt.cpu(), t[0].cpu().item()).to(xt.device)
        eye = torch.eye(xt.size(-1))
        cond_flows = (eye - xt.cpu().unsqueeze(-1)) * c_factor.unsqueeze(-2)
        flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1).to(xt.device)
        xtdt = xt + self.step_size * flow
        return xtdt / (xtdt.sum(dim=-1,keepdim=True) + 1e-10)
    def rhs_sample(self,model,t,xt,k=1):
        l1 = model(t,xt)
        vals,ids = l1.topk(k,dim=-1)
        
        x1 = ids[
            torch.arange(ids.size(0))[:,None],
            torch.arange(ids.size(1))[None],
            torch.distributions.Categorical(logits=vals).sample()
        ]
        x0 = self.sampler_0(x1)
        xtdt = self.interpolate(x0,x1,t=t+self.step_size)
        return xtdt
    @torch.no_grad()
    def inference_mixed(self,model,x0, mask=None, k=1, t_split=0.25):
        n_steps = _inference_steps_from_mask(mask, self.N) if mask is not None else self.N
        step_size_save = self.step_size
        self.step_size = (self.max_t - 1) / n_steps
        ts = torch.linspace(1,self.max_t,n_steps+1,device=x0.device)
        xt = x0.clone()
        for t in tqdm(ts[:-1]):
            if t < t_split:
                qt = self.rhs(model, t[None].repeat(x0.size(0)), xt)
            else:
                qt = self.rhs_sample(model, t[None].repeat(x0.size(0)), xt, k=k)
            
        t_max = torch.tensor([self.max_t],device=x0.device).repeat(x0.size(0))
        x1_hat = model(t_max,xt).softmax(dim=-1)
        self.step_size = step_size_save
        return x1_hat