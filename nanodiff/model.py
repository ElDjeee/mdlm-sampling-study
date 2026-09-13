"""Pre-norm Transformer, including a numerically equivalent AR KV cache."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    width: int = 96
    layers: int = 2
    heads: int = 4
    position_scale: float = 1.0
    local_bias: bool = False

    def __post_init__(self):
        if self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")

    @property
    def mask_id(self): return self.vocab_size
    @property
    def bos_id(self): return self.vocab_size + 1
    @property
    def pad_id(self): return self.vocab_size + 2


def init_params(key, cfg):
    def normal(shape, scale):
        nonlocal key
        key, sub = jax.random.split(key)
        return jax.random.normal(sub, shape) * scale
    d = cfg.width
    blocks = []
    for _ in range(cfg.layers):
        blocks.append(dict(
            ln1_g=jnp.ones(d), ln1_b=jnp.zeros(d),
            ln2_g=jnp.ones(d), ln2_b=jnp.zeros(d),
            qkv=normal((d, 3*d), d**-.5), qkv_b=jnp.zeros(3*d),
            out=normal((d, d), (d*2*cfg.layers)**-.5), out_b=jnp.zeros(d),
            fc1=normal((d, 4*d), d**-.5), fc1_b=jnp.zeros(4*d),
            fc2=normal((4*d, d), (4*d*2*cfg.layers)**-.5), fc2_b=jnp.zeros(d)))
    return dict(embedding=normal((cfg.vocab_size+3, d), .1), blocks=blocks,
                final_g=jnp.ones(d), final_b=jnp.zeros(d),
                head=normal((d, cfg.vocab_size), .02), head_b=jnp.zeros(cfg.vocab_size))


def layer_norm(x, gain, bias):
    mean = jnp.mean(x, -1, keepdims=True)
    var = jnp.mean((x-mean)**2, -1, keepdims=True)
    return (x-mean)*jax.lax.rsqrt(var+1e-5)*gain+bias


def positions(indices, width):
    phase = indices[..., None] * jnp.exp(-jnp.log(10000.)*jnp.arange(width//2)/(width//2))
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], -1)


def attention(q, k, v, allowed, query_start=0, local_bias=False):
    scores = jnp.einsum('bthd,bshd->bhts', q, k) / jnp.sqrt(q.shape[-1])
    if local_bias:
        # Fixed symmetric distance prior, shared by AR and diffusion. At this
        # tiny data/compute scale it provides locality without learned params.
        distance = jnp.abs((jnp.arange(q.shape[1])+query_start)[:,None]-jnp.arange(k.shape[1])[None,:])
        slopes = 2.**(-2.*(jnp.arange(q.shape[2])+1))
        scores = scores-slopes[None,:,None,None]*distance[None,None]
    scores = jnp.where(allowed[:, None, :, :], scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum('bhts,bshd->bthd', weights, v).reshape(q.shape[0], q.shape[1], -1)


def forward(params, tokens, cfg, causal=False):
    b, length = tokens.shape
    x = params['embedding'][tokens] + cfg.position_scale*positions(jnp.arange(length), cfg.width)[None]
    valid = tokens != cfg.pad_id
    allowed = jnp.broadcast_to(valid[:, None, :], (b, length, length))
    if causal:
        allowed = allowed & (jnp.arange(length)[:, None] >= jnp.arange(length)[None, :])[None]
    for p in params['blocks']:
        z = layer_norm(x, p['ln1_g'], p['ln1_b'])
        qkv = (z @ p['qkv']+p['qkv_b']).reshape(b, length, 3, cfg.heads, cfg.width//cfg.heads)
        q,k,v = qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
        x = x + attention(q,k,v,allowed,local_bias=cfg.local_bias) @ p['out'] + p['out_b']
        z = layer_norm(x,p['ln2_g'],p['ln2_b'])
        x = x + jax.nn.gelu(z @ p['fc1']+p['fc1_b']) @ p['fc2']+p['fc2_b']
    return layer_norm(x,params['final_g'],params['final_b']) @ params['head']+params['head_b']


def empty_cache(cfg, batch, length):
    return jnp.zeros((cfg.layers,2,batch,length,cfg.heads,cfg.width//cfg.heads), jnp.float32)


def decode_step(params, token, index, cache, cfg):
    """One causal token; fixed-size cache, no recomputation of past keys/values."""
    b = token.shape[0]
    x = params['embedding'][token][:,None] + cfg.position_scale*positions(index,cfg.width)[None,None]
    allowed = jnp.broadcast_to(jnp.arange(cache.shape[3]) <= index,(b,1,cache.shape[3]))
    for layer,p in enumerate(params['blocks']):
        z = layer_norm(x,p['ln1_g'],p['ln1_b'])
        qkv = (z @ p['qkv']+p['qkv_b']).reshape(b,1,3,cfg.heads,cfg.width//cfg.heads)
        q,k,v = qkv[:,:,0],qkv[:,:,1],qkv[:,:,2]
        cache = cache.at[layer,0,:,index,:,:].set(k[:,0])
        cache = cache.at[layer,1,:,index,:,:].set(v[:,0])
        x = x + attention(q,cache[layer,0],cache[layer,1],allowed,index,cfg.local_bias) @ p['out']+p['out_b']
        z = layer_norm(x,p['ln2_g'],p['ln2_b'])
        x = x + jax.nn.gelu(z @ p['fc1']+p['fc1_b']) @ p['fc2']+p['fc2_b']
    logits = layer_norm(x,params['final_g'],params['final_b']) @ params['head']+params['head_b']
    return logits[:,0],cache


def parameter_count(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
