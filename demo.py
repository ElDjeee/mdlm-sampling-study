#!/usr/bin/env python3
"""Display actual unmasking steps from a saved model."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from nanodiff.data import decode
from nanodiff.training import load_checkpoint
from nanodiff.diffusion import diffusion_sample


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='results/study/mdlm_s0')
    parser.add_argument('--length',type=int,default=128)
    parser.add_argument('--steps',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--strategy',choices=['ancestral','confidence','random'],default='ancestral')
    parser.add_argument('--temperature',type=float,default=1.)
    parser.add_argument('--text',help='Optional infilling context. Use █ for masked characters.')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    directory=root/args.run
    state,cfg,metadata=load_checkpoint(directory/'checkpoint.npz')
    manifest=json.loads((directory.parent/'manifest.json').read_text())
    chars=manifest['characters'];lookup={c:i for i,c in enumerate(chars)}
    if args.text:
        initial=jnp.array([[cfg.mask_id if c=='█' else lookup.get(c,manifest['unk_id']) for c in args.text]],jnp.int32)
    else: initial=jnp.full((1,args.length),cfg.mask_id,jnp.int32)
    fn=jax.jit(lambda p,k:diffusion_sample(p,k,initial,cfg,args.steps,args.strategy,metadata['schedule'],args.temperature))
    result,trace=fn(state['params'],jax.random.PRNGKey(args.seed))
    print('INITIAL\n'+decode(np.array(initial)[0],chars))
    for i,row in enumerate(np.array(trace)):
        print(f'\nSTEP {i+1}/{args.steps}\n'+decode(row[0],chars))


if __name__=='__main__':main()
