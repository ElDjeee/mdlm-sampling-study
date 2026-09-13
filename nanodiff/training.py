"""Pure JAX AdamW and resumable checkpoints without pickle."""
import json
from dataclasses import asdict
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .model import init_params,ModelConfig
from .diffusion import objective


def init_state(key,cfg):
    key,pk = jax.random.split(key)
    params = init_params(pk,cfg)
    zeros = jax.tree_util.tree_map(jnp.zeros_like,params)
    return dict(params=params,m=zeros,v=zeros,step=jnp.array(0,jnp.int32),key=key)


def update(state,grads,lr,weight_decay=.01):
    norm = jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree_util.tree_leaves(grads)))
    grads = jax.tree_util.tree_map(lambda g:g*jnp.minimum(1.,1./(norm+1e-8)),grads)
    step = state['step']+1
    m = jax.tree_util.tree_map(lambda a,g:.9*a+.1*g,state['m'],grads)
    v = jax.tree_util.tree_map(lambda a,g:.999*a+.001*g*g,state['v'],grads)
    params = jax.tree_util.tree_map(
        lambda p,a,b:p-lr*(a/(1-.9**step)/(jnp.sqrt(b/(1-.999**step))+1e-8)+weight_decay*p*(p.ndim>1)),
        state['params'],m,v)
    return dict(state,params=params,m=m,v=v,step=step),norm


def make_train_step(cfg,kind,schedule,steps,learning_rate,total_steps):
    def train_step(state,batch):
        key,sub = jax.random.split(state['key'])
        loss,grads = jax.value_and_grad(objective)(state['params'],batch,sub,cfg,kind,schedule,steps)
        warmup = jnp.minimum((state['step']+1)/50.,1.)
        progress = jnp.minimum(state['step']/max(total_steps,1),1.)
        lr = learning_rate*warmup*(.2+.8*.5*(1+jnp.cos(jnp.pi*progress)))
        state,norm = update(dict(state,key=key),grads,lr)
        return state,dict(loss=loss,gradient_norm=norm,lr=lr)
    return jax.jit(train_step)


def batch_at(data,seed,step,batch_size,length):
    # Stateless per-step sampling makes interrupted/restarted runs identical.
    rng = np.random.default_rng(np.random.SeedSequence([seed,step,1701]))
    starts = rng.integers(0,len(data)-length+1,size=batch_size)
    return jnp.asarray(data[starts[:,None]+np.arange(length)[None]],jnp.int32)


def save_checkpoint(path,state,cfg,metadata):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    leaves = jax.tree_util.tree_leaves(state)
    arrays = {f'a{i}':np.asarray(x) for i,x in enumerate(leaves)}
    arrays['metadata'] = np.array(json.dumps(dict(config=asdict(cfg),run=metadata)))
    with path.open('wb') as handle: np.savez_compressed(handle,**arrays)


def load_checkpoint(path):
    with np.load(path,allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata']))
        cfg = ModelConfig(**meta['config'])
        template = init_state(jax.random.PRNGKey(0),cfg)
        tree = jax.tree_util.tree_structure(template)
        leaves = [jnp.asarray(archive[f'a{i}']) for i in range(tree.num_leaves)]
        state = jax.tree_util.tree_unflatten(tree,leaves)
    return state,cfg,meta['run']
