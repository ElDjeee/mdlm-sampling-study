#!/usr/bin/env python3
"""One CLI for training, evaluation, benchmarks and the generated report."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).parent/'.mpl-cache'))
import argparse
import json
import platform
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from nanodiff.data import load_data,evaluation_windows
from nanodiff.model import ModelConfig,parameter_count
from nanodiff.training import init_state,make_train_step,batch_at,save_checkpoint,load_checkpoint
from nanodiff.diffusion import objective

ROOT = Path(__file__).resolve().parent


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def specs(config):
    runs = []
    for kind in ['ar','mdlm','mlm']:
        for seed in config['seeds']:
            runs.append(dict(name=f'{kind}_s{seed}',kind=kind,schedule='linear',seed=seed))
    for schedule in ['cosine','log']:
        runs.append(dict(name=f'mdlm_{schedule}_s0',kind='mdlm',schedule=schedule,seed=0))
    runs.append(dict(name='unweighted_s0',kind='unweighted',schedule='linear',seed=0))
    return runs


def train_run(spec,config,data,cfg,out):
    directory = out/spec['name']
    directory.mkdir(parents=True,exist_ok=True)
    checkpoint = directory/'checkpoint.npz'
    metadata = dict(spec,experiment=config,history=[],training_seconds=0.,compilation_seconds=[])
    if checkpoint.exists():
        state,old_cfg,metadata = load_checkpoint(checkpoint)
        if old_cfg != cfg or metadata['experiment'] != config:
            raise ValueError(f'Configuration differs from saved run: {directory}')
    else:
        state = init_state(jax.random.PRNGKey(spec['seed']),cfg)
    start_step = int(state['step'])
    if start_step >= config['train_steps']:
        print(f"{spec['name']}: already trained",flush=True)
        return
    step_fn = make_train_step(cfg,spec['kind'],spec['schedule'],config['diffusion_steps'],
                              config['learning_rate'],config['train_steps'])
    val = jnp.asarray(evaluation_windows(data['val'],32,config['length']))
    val_fn = jax.jit(lambda p:objective(p,val,jax.random.PRNGKey(100),cfg,spec['kind'],spec['schedule'],config['diffusion_steps']))
    if not metadata['history']:
        initial_val = float(val_fn(state['params']))
        metadata['history'].append(dict(step=0,val_objective=initial_val,loss=None))
    print(f"TRAIN {spec['name']} | {parameter_count(state['params']):,} parameters | step {start_step}/{config['train_steps']}",flush=True)
    last_loss = None
    for step in range(start_step,config['train_steps']):
        batch = batch_at(data['train'],spec['seed'],step,config['batch_size'],config['length'])
        before = time.perf_counter()
        state,metrics = step_fn(state,batch)
        # Synchronize so timings include execution, not only JAX dispatch.
        metrics['loss'].block_until_ready()
        elapsed = time.perf_counter()-before
        if step == start_step: metadata['compilation_seconds'].append(elapsed)
        else: metadata['training_seconds'] += elapsed
        last_loss = float(metrics['loss'])
        if not np.isfinite(last_loss): raise FloatingPointError(f'Nonfinite loss: {spec}, step={step}')
        if (step+1)%config['log_every']==0 or step+1==config['train_steps']:
            row = dict(step=step+1,loss=last_loss,val_objective=float(val_fn(state['params'])),
                       gradient_norm=float(metrics['gradient_norm']),lr=float(metrics['lr']))
            metadata['history'].append(row)
            print(f"  {step+1:5d} loss={last_loss:.3f} val={row['val_objective']:.3f} elapsed_train={metadata['training_seconds']:.1f}s",flush=True)
            save_checkpoint(checkpoint,state,cfg,metadata)
            write_json(directory/'training.json',dict(metadata,parameters=parameter_count(state['params']),
                       tokens_seen=int(state['step'])*config['batch_size']*config['length']))
    jax.clear_caches()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','evaluate','benchmark','report','all'])
    parser.add_argument('--config',default='configs/quick.json')
    parser.add_argument('--output',default='results/quick')
    parser.add_argument('--runs',nargs='*',help='Optional exact run names')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    out = ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    data,chars,manifest = load_data(ROOT/'data/tinyshakespeare.txt',config['train_chars'])
    cfg = ModelConfig(len(chars)+1,config['width'],config['layers'],config['heads'],config.get('position_scale',1.),config.get('local_bias',False))
    manifest['environment'] = dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,
                                   platform=platform.platform(),machine=platform.machine(),
                                   devices=[str(d) for d in jax.devices()],jax_platforms=os.environ.get('JAX_PLATFORMS'))
    write_json(out/'manifest.json',manifest)
    write_json(out/'config.json',config)
    runs = [s for s in specs(config) if not args.runs or s['name'] in args.runs]
    if args.runs and len(runs)!=len(set(args.runs)): raise ValueError('Unknown run name')
    if args.action in ('train','all'):
        for spec in runs: train_run(spec,config,data,cfg,out)
    if args.action in ('evaluate','all'):
        from nanodiff.evaluation import evaluate_all
        evaluate_all(runs,config,data,chars,cfg,out)
    if args.action in ('benchmark','all'):
        from nanodiff.evaluation import benchmark_all
        benchmark_all(config,data,chars,cfg,out)
    if args.action in ('report','all'):
        from nanodiff.reporting import build_report
        build_report(out)


if __name__ == '__main__': main()
