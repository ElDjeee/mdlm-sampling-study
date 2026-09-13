#!/usr/bin/env python3
"""An exactly evaluable, pre-specified proof-of-function for the existing MDLM."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
import argparse
import hashlib
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from nanodiff.exact_copy import ExactCopy
from nanodiff.model import ModelConfig,forward,parameter_count
from nanodiff.training import init_state,make_train_step,save_checkpoint,load_checkpoint
from nanodiff.diffusion import diffusion_sample


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def model_probabilities(params,cfg,experiment):
    fn=jax.jit(lambda p,x:forward(p,x,cfg))
    chunks=[]
    for start in range(0,len(experiment.states),256):
        x=experiment.states[start:start+256]
        padded=np.pad(x,((0,256-len(x)),(0,0)),constant_values=2)
        chunks.append(np.asarray(fn(params,jnp.asarray(padded)))[:len(x)])
    logits=np.concatenate(chunks).astype(np.float64)
    probs=np.exp(logits-logits.max(-1,keepdims=True))
    return probs/probs.sum(-1,keepdims=True)


def evaluate(params,cfg,experiment,steps):
    probs=model_probabilities(params,cfg,experiment)
    distribution=experiment.output_distribution(probs,steps)
    return experiment.metrics(distribution),probs,distribution


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['run','report'],nargs='?',default='run')
    parser.add_argument('--config',default='configs/proof.json')
    parser.add_argument('--output',default='results/proof')
    args=parser.parse_args()
    config_path=ROOT/args.config
    config=json.loads(config_path.read_text())
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    if args.action=='report':
        from nanodiff.proof_reporting import build_proof_report
        build_proof_report(out)
        return
    # Preserve the original pre-run protocol; refuse incompatible reuse.
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=config:raise ValueError('Protocol changed: use a new output folder')
    else:write_json(out/'protocol.json',config)
    sources=['nanodiff/model.py','nanodiff/diffusion.py','nanodiff/training.py','nanodiff/exact_copy.py','proof.py',args.config]
    write_json(out/'source_manifest.json',dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),jax=jax.__version__,
               backend=str(jax.devices()),sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}))
    experiment=ExactCopy(config['pairs'])
    cfg=ModelConfig(2,config['width'],config['layers'],config['heads'],config['position_scale'],config['local_bias'])
    baselines={'independent':experiment.metrics(np.full(2**experiment.length,1/2**experiment.length)),
               'target':experiment.metrics(experiment.target),'oracle':{}}
    for steps in config['sampling_steps']:
        p=experiment.output_distribution(experiment.oracle_probabilities(),steps)
        baselines['oracle'][str(steps)]=experiment.metrics(p)
    write_json(out/'baselines.json',baselines)
    all_results=[]
    for seed in config['seeds']:
        folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
        path=folder/'checkpoint.npz'
        metadata=dict(seed=seed,config=config,history=[],train_seconds=0.,compile_and_first_step_seconds=0.)
        state=init_state(jax.random.PRNGKey(seed),cfg)
        if path.exists():
            state,saved_cfg,metadata=load_checkpoint(path)
            if saved_cfg!=cfg or metadata['config']!=config:raise ValueError('Incompatible checkpoint')
        start_step=int(state['step'])
        if not metadata['history']:
            metrics,_,_=evaluate(state['params'],cfg,experiment,config['diffusion_steps'])
            metadata['history'].append(dict(step=0,loss=None,**metrics))
        train=make_train_step(cfg,'mdlm','linear',config['diffusion_steps'],config['learning_rate'],config['train_steps'])
        print(f'PROOF seed={seed} params={parameter_count(state["params"])} start={start_step}',flush=True)
        for step in range(start_step,config['train_steps']):
            rng=np.random.default_rng(np.random.SeedSequence([seed,step,2026]))
            bits=rng.integers(0,2,(config['batch_size'],config['pairs']),dtype=np.int32)
            # Training data only: fresh fair binary words and their copies.
            batch=jnp.asarray(np.concatenate([bits,bits],axis=1))
            before=time.perf_counter()
            state,train_metrics=train(state,batch)
            train_metrics['loss'].block_until_ready()
            elapsed=time.perf_counter()-before
            if step==start_step:metadata['compile_and_first_step_seconds']+=elapsed
            else:metadata['train_seconds']+=elapsed
            if not np.isfinite(float(train_metrics['loss'])):raise FloatingPointError('Training diverged')
            if step+1 in config['checkpoints']:
                metrics,_,_=evaluate(state['params'],cfg,experiment,config['diffusion_steps'])
                metadata['history'].append(dict(step=step+1,loss=float(train_metrics['loss']),**metrics))
                save_checkpoint(path,state,cfg,metadata)
                write_json(folder/'training.json',metadata)
                print(f'  step={step+1} valid={metrics["valid_probability"]:.4f} TV={metrics["total_variation"]:.4f} NLL={metrics["exact_nll_bits_per_character"]:.4f}',flush=True)
        metrics,probs,distribution=evaluate(state['params'],cfg,experiment,config['diffusion_steps'])
        step_curve={}
        for steps in config['sampling_steps']:
            step_curve[str(steps)]=experiment.metrics(experiment.output_distribution(probs,steps))
        initial=jnp.full((config['sample_batch'],experiment.length),cfg.mask_id,jnp.int32)
        sample_fn=jax.jit(lambda p,k:diffusion_sample(p,k,initial,cfg,config['diffusion_steps'],'ancestral')[0])
        samples=[]
        for i in range((config['sample_count']+config['sample_batch']-1)//config['sample_batch']):
            samples.append(np.asarray(sample_fn(state['params'],jax.random.PRNGKey(90000+seed*1000+i))))
        samples=np.concatenate(samples)[:config['sample_count']]
        empirical=experiment.sample_distribution(samples)
        empirical_tv=float(.5*np.abs(empirical-distribution).sum())
        limits=config['acceptance']
        gates={
            'valid_probability':metrics['valid_probability']>=limits['valid_probability_min'],
            'total_variation':metrics['total_variation']<=limits['total_variation_max'],
            'exact_nll':metrics['exact_nll_bits_per_character']<=limits['exact_nll_bits_per_character_max'],
            'all_modes_noncollapsed':metrics['min_valid_mode_probability']>=limits['min_valid_mode_probability_min'],
            'sampler_agrees_with_exact_chain':empirical_tv<=limits['empirical_vs_exact_tv_max']}
        result=dict(seed=seed,parameters=parameter_count(state['params']),metrics=metrics,acceptance=gates,
                    all_passed=all(gates.values()),step_curve=step_curve,conditionals=experiment.conditional_metrics(probs),
                    sampler_check=dict(count=len(samples),empirical_vs_exact_tv=empirical_tv,
                                       empirical_valid_fraction=float(empirical[experiment.valid].sum()),
                                       observed_valid_modes=int(np.count_nonzero(empirical[experiment.valid])),
                                       first_16_samples=[''.join(map(str,row)) for row in samples[:16]]),
                    train_seconds=metadata['train_seconds'],compile_and_first_step_seconds=metadata['compile_and_first_step_seconds'])
        np.savez_compressed(folder/'distributions.npz',target=experiment.target,model=distribution,empirical=empirical,states=experiment.binary)
        np.save(folder/'samples.npy',samples,allow_pickle=False)
        write_json(folder/'evaluation.json',result)
        all_results.append(result)
        print(f'  RESULT {json.dumps(result["acceptance"])}',flush=True)
        jax.clear_caches()
    write_json(out/'summary.json',dict(all_passed=all(r['all_passed'] for r in all_results),runs=all_results))
    from nanodiff.proof_reporting import build_proof_report
    build_proof_report(out)
    if not all(r['all_passed'] for r in all_results):raise SystemExit(2)


if __name__=='__main__':main()
