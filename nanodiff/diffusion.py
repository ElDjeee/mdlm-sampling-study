"""Finite-time MDLM/SUBS objective and explicit sampling alternatives.

With m_k = probability of masking at step k, the exact finite-chain NELBO
is sum_k E[(m_k-m_{k-1})/m_k * sum_i 1[z_i=MASK] CE_i].
Sample k uniformly from 1..T and multiply by T for an unbiased estimator.
Normalize by ALL valid characters, never by the realized masked count.
The k=1 term includes final reconstruction; terminal prior KL is zero.
"""
import jax
import jax.numpy as jnp
from .model import forward, decode_step, empty_cache


def mask_probability(t, schedule='linear'):
    if schedule == 'linear': return t
    if schedule == 'cosine': return jnp.sin(jnp.pi*t/2)**2
    if schedule == 'log': return jnp.log1p(9*t)/jnp.log(10.)
    raise ValueError(f'Unknown schedule: {schedule}')


def corruption(key, tokens, probability, cfg):
    valid = tokens < cfg.vocab_size
    p = jnp.asarray(probability)
    if p.ndim == 1: p = p[:,None]
    masked = (jax.random.uniform(key,tokens.shape) < p) & valid
    return jnp.where(masked,cfg.mask_id,tokens),masked


def cross_entropy(logits, targets):
    safe = jnp.clip(targets,0,logits.shape[-1]-1)
    return -jnp.take_along_axis(jax.nn.log_softmax(logits),safe[...,None],-1)[...,0]


def weighted_loss(logits, targets, masked, weights, cfg):
    valid = targets < cfg.vocab_size
    per_sequence = jnp.sum(cross_entropy(logits,targets)*masked*valid,axis=-1)/jnp.maximum(valid.sum(-1),1)
    return jnp.mean(per_sequence*weights)


def ar_inputs(targets, cfg):
    return jnp.concatenate([jnp.full((targets.shape[0],1),cfg.bos_id,jnp.int32),targets[:,:-1]],-1)


def objective(params, targets, key, cfg, kind='mdlm', schedule='linear', steps=64):
    if kind == 'ar':
        logits = forward(params,ar_inputs(targets,cfg),cfg,causal=True)
        return weighted_loss(logits,targets,targets < cfg.vocab_size,jnp.ones(targets.shape[0]),cfg)
    kt,kc = jax.random.split(key)
    # Fixed time strata across exchangeable IID data rows. The aggregate time
    # marginal is uniform; unbiasedness is over both data and time sampling.
    # A particular fixed row does NOT have a uniform time marginal. Final NELBO
    # evaluation therefore sums every time step for every held-out window.
    u = (jnp.arange(targets.shape[0])+jax.random.uniform(kt,()))/targets.shape[0]
    k = jnp.minimum((u*steps).astype(jnp.int32)+1,steps)
    m = mask_probability(k/steps,schedule)
    previous = mask_probability((k-1)/steps,schedule)
    weights = steps*(m-previous)/jnp.maximum(m,1e-8)
    if kind == 'mlm':
        m = jnp.full_like(m,.15)
    noisy,masked = corruption(kc,targets,m,cfg)
    logits = forward(params,noisy,cfg)
    if kind == 'mlm':
        ce = cross_entropy(logits,targets)
        return jnp.mean(jnp.sum(ce*masked,-1)/jnp.maximum(masked.sum(-1),1))
    if kind == 'unweighted': weights = jnp.ones_like(weights)
    return weighted_loss(logits,targets,masked,weights,cfg)


def draw_tokens(key, logits, temperature=1., top_k=0):
    if temperature == 0: return jnp.argmax(logits,-1).astype(jnp.int32)
    scores = logits/temperature
    if top_k:
        cutoff = jax.lax.top_k(scores,min(top_k,scores.shape[-1]))[0][...,-1,None]
        scores = jnp.where(scores>=cutoff,scores,-jnp.inf)
    return jax.random.categorical(key,scores).astype(jnp.int32)


def diffusion_sample(params,key,initial,cfg,steps=16,strategy='ancestral',schedule='linear',temperature=1.,top_k=0):
    """Return final tokens and trajectory. Never change observed/unmasked tokens.

    Ancestral uses the actual reverse transition. Confidence/random use exact
    quotas and are labeled heuristic; their output likelihood is not the ELBO.
    """
    if strategy not in ('ancestral','random','confidence'): raise ValueError(strategy)
    if steps < 1: raise ValueError('steps must be positive')
    original_count = (initial == cfg.mask_id).sum(-1)
    def body(carry,index):
        tokens,key = carry
        key,kvalue,kposition = jax.random.split(key,3)
        logits = forward(params,tokens,cfg)
        candidates = draw_tokens(kvalue,logits,temperature,top_k)
        masked = tokens == cfg.mask_id
        t = 1.-index/steps
        s = jnp.maximum(0.,1.-(index+1)/steps)
        mt,ms = mask_probability(t,schedule),mask_probability(s,schedule)
        if strategy == 'ancestral':
            reveal = jax.random.uniform(kposition,tokens.shape) < (mt-ms)/jnp.maximum(mt,1e-8)
        else:
            if strategy == 'confidence':
                probs = jax.nn.softmax(logits)
                scores = jnp.take_along_axis(probs,candidates[...,None],-1)[...,0]
            else:
                scores = jax.random.uniform(kposition,tokens.shape)
            scores = jnp.where(masked,scores,-jnp.inf)
            ranks = jnp.argsort(jnp.argsort(-scores,axis=-1),axis=-1)
            remain = jnp.floor(original_count*ms).astype(jnp.int32)
            count = jnp.maximum(masked.sum(-1)-remain,0)
            reveal = ranks < count[:,None]
        reveal = (reveal | (index==steps-1)) & masked
        tokens = jnp.where(reveal,candidates,tokens)
        return (tokens,key),tokens
    (tokens,_),trajectory = jax.lax.scan(body,(initial,key),jnp.arange(steps))
    return tokens,trajectory


def ar_sample(params,key,cfg,batch=1,length=128,temperature=1.,top_k=0):
    cache = empty_cache(cfg,batch,length)
    def body(carry,index):
        previous,cache,key = carry
        logits,cache = decode_step(params,previous,index,cache,cfg)
        key,sub = jax.random.split(key)
        token = draw_tokens(sub,logits,temperature,top_k)
        return (token,cache,key),token
    _,tokens = jax.lax.scan(body,(jnp.full((batch,),cfg.bos_id,jnp.int32),cache,key),jnp.arange(length))
    return tokens.T
