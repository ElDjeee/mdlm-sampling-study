#!/usr/bin/env python3
"""Fresh replicated ancestral generations from the three existing MDLM checkpoints."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from nanodiff.data import load_data, evaluation_windows
from nanodiff.diffusion import diffusion_sample
from nanodiff.training import load_checkpoint
from nanodiff.sampling_statistics import counts, js_counts, bootstrap_fixed_checkpoints, paired_sign_test

ROOT = Path(__file__).resolve().parent


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='results/study')
    parser.add_argument('--config', default='configs/sampling_study.json')
    args = parser.parse_args()
    study = ROOT/args.study
    protocol = json.loads((ROOT/args.config).read_text())
    base = json.loads((study/'config.json').read_text())
    out = study/'sampling_statistics'
    out.mkdir(exist_ok=True)
    sources = [Path('sampling_study.py'), Path('nanodiff/sampling_statistics.py'),
               Path('nanodiff/diffusion.py'), Path('nanodiff/model.py'),
               Path('nanodiff/training.py'), Path('nanodiff/data.py'), Path(args.config)]
    hashes = {str(p): hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    data, chars, metadata = load_data(ROOT/'data/tinyshakespeare.txt', base['train_chars'])
    checkpoints = {name:hashlib.sha256((study/name/'checkpoint.npz').read_bytes()).hexdigest()
                   for name in protocol['checkpoints']}
    frozen = dict(protocol=protocol, source_sha256=hashes, checkpoint_sha256=checkpoints,
                  dataset_sha256=metadata['sha256'], jax_version=jax.__version__, backend=jax.default_backend())
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text()) == frozen, 'Protocol or source changed; use a new output study'
    else:
        save(out/'protocol.json', frozen)
    reference = evaluation_windows(data[protocol['reference_split']], protocol['reference_windows'], protocol['length'])
    assert len(reference) == protocol['reference_windows']
    np.save(out/'reference_tokens.npy', reference)
    vocab = len(chars)+1
    specs = dict(js1=(1,1), js3=(3,1))
    specs.update({f'pair_js_lag{d}':(2,d) for d in protocol['pair_lags']})
    ref_counts = {name:counts(reference, vocab, *spec) for name,spec in specs.items()}
    state, cfg, _ = load_checkpoint(study/protocol['checkpoints'][0]/'checkpoint.npz')
    initial = jnp.full((protocol['samples_per_replicate'], protocol['length']), cfg.mask_id, jnp.int32)
    fns = {steps:jax.jit(lambda p,k,steps=steps: diffusion_sample(
        p,k,initial,cfg,steps,protocol['strategy'],protocol['schedule'],protocol['temperature'],protocol['top_k'])[0])
        for steps in protocol['steps']}
    rows = []
    before = time.perf_counter()
    for ci, name in enumerate(protocol['checkpoints']):
        state, current_cfg, _ = load_checkpoint(study/name/'checkpoint.npz')
        assert current_cfg == cfg and int(state['step']) == base['train_steps']
        for rep in range(protocol['replicates']):
            key_seed = protocol['sampling_seed'] + 1000*ci + rep
            for steps, fn in fns.items():
                path = out/f'{name}_rep{rep}_steps{steps}.npy'
                if path.exists():
                    tokens = np.load(path, allow_pickle=False)
                else:
                    tokens = np.asarray(fn(state['params'], jax.random.PRNGKey(key_seed)))
                    np.save(path, tokens)
                assert tokens.shape == initial.shape and ((tokens>=0)&(tokens<vocab)).all()
                metrics = {metric:js_counts(counts(tokens,vocab,*spec),ref_counts[metric])
                           for metric,spec in specs.items()}
                rows.append(dict(checkpoint=name, replicate=rep, steps=steps, rng_seed=key_seed, **metrics))
            save(out/'batch_metrics.json', rows)
            print(f'{name} replicate {rep+1}/{protocol["replicates"]} complete; elapsed {time.perf_counter()-before:.1f}s', flush=True)
    def matrix(steps, metric):
        return np.array([[next(r[metric] for r in rows if r['checkpoint']==name and r['replicate']==rep and r['steps']==steps)
                          for rep in range(protocol['replicates'])] for name in protocol['checkpoints']])
    def estimate(x):
        return bootstrap_fixed_checkpoints(x,protocol['bootstrap_draws'],protocol['bootstrap_seed'])
    curves = {metric:{str(steps):estimate(matrix(steps,metric)) for steps in protocol['steps']} for metric in specs}
    a,b = protocol['primary_contrast']
    differences = {metric:matrix(a,metric)-matrix(b,metric) for metric in specs}
    contrasts = {metric:estimate(delta) for metric,delta in differences.items()}
    primary = dict(metric=protocol['primary_metric'], steps_minus=[a,b],
                   **contrasts[protocol['primary_metric']],
                   sign_test=paired_sign_test(differences[protocol['primary_metric']]))
    result = dict(protocol=protocol, primary=primary, curves=curves, contrasts=contrasts,
                  total_sequences=len(rows)*protocol['samples_per_replicate'],
                  reference_sequences=len(reference), reference_chars=reference.size,
                  conditional_uncertainty='95% percentile bootstrap over generation batches within each fixed checkpoint; not over tokens, training seeds or reference text. Secondary intervals are pointwise and exploratory.',
                  total_elapsed_this_invocation_seconds=time.perf_counter()-before,
                  output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.npy'))})
    save(out/'summary.json',result)
    print(json.dumps(primary,indent=2),flush=True)


if __name__ == '__main__':
    main()
