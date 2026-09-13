import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .data import evaluation_windows,decode
from .model import forward
from .training import load_checkpoint
from .diffusion import cross_entropy,ar_inputs,mask_probability,corruption,diffusion_sample,ar_sample
from .metrics import sample_metrics,NGram


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)


def stats(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),se=float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,n=len(values))


def likelihood(params,windows,cfg,kind,schedule,steps,repeats):
    x=jnp.asarray(windows)
    if kind=='ar':
        fn=jax.jit(lambda p:cross_entropy(forward(p,ar_inputs(x,cfg),cfg,True),x).mean(-1)/jnp.log(2.))
        return dict(metric='exact_chunk_nll_bpc',**stats(fn(params)))
    @jax.jit
    def estimate(p,key):
        def body(carry,k):
            total,key=carry
            key,sub=jax.random.split(key)
            m=mask_probability(k/steps,schedule)
            previous=mask_probability((k-1)/steps,schedule)
            noisy,mask=corruption(sub,x,m,cfg)
            ce=cross_entropy(forward(p,noisy,cfg),x)
            contribution=(ce*mask).mean(-1)*(m-previous)/m
            return (total+contribution,key),None
        (total,_),_=jax.lax.scan(body,(jnp.zeros(len(x)),key),jnp.arange(1,steps+1))
        return total/jnp.log(2.)
    values=np.stack([np.asarray(estimate(params,jax.random.PRNGKey(570+r))) for r in range(repeats)])
    return dict(metric='finite_chain_nelbo_bpc_mc',**stats(values.mean(0)),
                repeat_means=values.mean(1).tolist(),steps=steps,repeats=repeats,
                note='Monte Carlo estimate of an upper bound on NLL; SE across windows; not exact perplexity or confidence-heuristic likelihood')


def reconstruction(params,windows,cfg):
    x=jnp.asarray(windows)
    @jax.jit
    def compute(p,probability,key):
        noisy,mask=corruption(key,x,probability,cfg)
        logits=forward(p,noisy,cfg)
        count=mask.sum()
        accuracy=((jnp.argmax(logits,-1)==x)*mask).sum()/jnp.maximum(count,1)
        ce=(cross_entropy(logits,x)*mask).sum()/jnp.maximum(count,1)
        return accuracy,ce,count
    rows=[]
    for i,rate in enumerate([.1,.25,.5,.75,.9,1.]):
        a,ce,n=compute(params,jnp.array(rate),jax.random.PRNGKey(700+i))
        rows.append(dict(mask_rate=rate,accuracy=float(a),masked_ce_nats=float(ce),masked_tokens=int(n)))
    return rows


def generated(params,key,cfg,kind,count,length,steps=16,strategy='ancestral',schedule='linear',temperature=1.,top_k=0):
    if kind=='ar':
        fn=jax.jit(lambda p,k:ar_sample(p,k,cfg,count,length,temperature,top_k))
    else:
        initial=jnp.full((count,length),cfg.mask_id,jnp.int32)
        fn=jax.jit(lambda p,k:diffusion_sample(p,k,initial,cfg,steps,strategy,schedule,temperature,top_k)[0])
    return np.asarray(fn(params,key))


def evaluate_all(runs,config,data,chars,cfg,out):
    training=decode(data['train'],chars)
    ngram=NGram(data['train'],cfg.vocab_size,config['length'])
    baselines={}
    for name,is_unigram in [('unigram',True),('trigram',False)]:
        result={}
        for split in ['val','test']:
            windows=evaluation_windows(data[split],config['eval_windows'],config['length'])
            result[split]=dict(metric='exact_chunk_nll_bpc',**stats(ngram.bpc(windows,is_unigram)))
        samples=[decode(x,chars) for x in ngram.sample(900,config['sample_count'],config['length'],is_unigram)]
        result['samples']=samples
        result['generation']=sample_metrics(samples,decode(data['test'],chars),training)
        if is_unigram:
            held=evaluation_windows(data['test'],min(config['eval_windows'],32),config['length'])
            left,right=config['length']//3,2*config['length']//3
            result['infilling_majority_accuracy']=float((held[:,left:right]==np.argmax(ngram.unigram)).mean())
        baselines[name]=result
    write_json(out/'baselines.json',baselines)
    for spec in runs:
        directory=out/spec['name']
        checkpoint=directory/'checkpoint.npz'
        state,_,meta=load_checkpoint(checkpoint)
        if int(state['step'])!=config['train_steps']: raise ValueError(f'Incomplete training: {spec}')
        p=state['params']
        result=dict(spec,steps_trained=int(state['step']),evaluated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        print(f"EVALUATE {spec['name']}",flush=True)
        for split in ['val','test']:
            windows=evaluation_windows(data[split],config['eval_windows'],config['length'])
            result[split]=likelihood(p,windows,cfg,spec['kind'],spec['schedule'],config['diffusion_steps'],config['eval_repeats'])
            if spec['kind']!='ar': result[f'{split}_reconstruction']=reconstruction(p,windows,cfg)
        strategy='confidence' if spec['kind']=='mlm' else 'ancestral'
        tokens=generated(p,jax.random.PRNGKey(800+spec['seed']),cfg,spec['kind'],config['sample_count'],config['length'],16,strategy,spec['schedule'])
        samples=[decode(x,chars) for x in tokens]
        result['samples']=samples
        result['generation_protocol']=dict(steps=config['length'] if spec['kind']=='ar' else 16,strategy='cached_causal' if spec['kind']=='ar' else strategy,temperature=1.,top_k=0)
        result['generation']=sample_metrics(samples,decode(data['test'],chars),training)
        if spec['kind']!='ar':
            windows=evaluation_windows(data['test'],min(config['eval_windows'],32),config['length'])
            left,right=config['length']//3,2*config['length']//3
            initial=jnp.asarray(windows).at[:,left:right].set(cfg.mask_id)
            fn=jax.jit(lambda p,k:diffusion_sample(p,k,initial,cfg,16,'confidence',spec['schedule'],0.))
            filled,trace=fn(p,jax.random.PRNGKey(1234))
            filled=np.asarray(filled)
            result['infilling']=dict(masked_span=[left,right],accuracy=float((filled[:,left:right]==windows[:,left:right]).mean()),
                unchanged_context=bool(np.all(filled[:,:left]==windows[:,:left]) and np.all(filled[:,right:]==windows[:,right:])),
                reference=decode(windows[0],chars),input=decode(np.asarray(initial)[0],chars),output=decode(filled[0],chars),
                note='Exact-match characters in held-out middle third, greedy confidence sampler; not a semantic metric; AR right-context comparison not claimed')
            result['trajectory']=[decode(np.asarray(initial)[0],chars)]+[decode(row[0],chars) for row in np.asarray(trace)]
        write_json(directory/'evaluation.json',result)
        print(f"  test {result['test']['metric']}={result['test']['mean']:.4f}",flush=True)
        jax.clear_caches()
    rows=[]
    for path in sorted(out.glob('*/evaluation.json')):
        result=json.loads(path.read_text())
        rows.append(dict(name=result['name'],kind=result['kind'],seed=result['seed'],schedule=result['schedule'],
                         metric=result['test']['metric'],test_bpc=result['test']['mean'],val_bpc=result['val']['mean'],**result['generation']))
    write_csv(out/'model_metrics.csv',rows)


def benchmark_all(config,data,chars,cfg,out):
    """Timing in fresh subprocesses: independent peak RSS and JIT warmups."""
    root=Path(__file__).resolve().parents[1]
    timing_rows=[]
    for length,batch in config['benchmark_shapes']:
        cases=[dict(kind='ar',name='ar_s0',steps=length,strategy='cached_causal')]
        for strategy in ['ancestral','confidence']:
            cases.extend(dict(kind='mdlm',name='mdlm_s0',steps=s,strategy=strategy) for s in config['sampling_steps'])
        for case in cases:
            args=dict(checkpoint=str(out/case['name']/'checkpoint.npz'),length=length,batch=batch,
                      steps=case['steps'],strategy=case['strategy'],kind=case['kind'],repeats=config['timing_repeats'])
            completed=subprocess.run([sys.executable,'-m','nanodiff.bench_worker',json.dumps(args)],cwd=root,capture_output=True,text=True,check=True)
            row=dict(case,**json.loads(completed.stdout),length=length,batch=batch,
                     extrapolates_training_length=length>config['length'])
            timing_rows.append(row)
            print(f"BENCH {case['kind']} {case['strategy']} S={case['steps']} L={length} B={batch}: {row['median_ms']:.2f} ms",flush=True)
            write_json(out/'timings.json',timing_rows)
            write_csv(out/'timings.csv',timing_rows)
    state,_,_=load_checkpoint(out/'mdlm_s0/checkpoint.npz')
    p=state['params']
    training=decode(data['train'],chars)
    reference=decode(data['val'],chars)
    ar_state,_,_=load_checkpoint(out/'ar_s0/checkpoint.npz')
    ar_tokens=generated(ar_state['params'],jax.random.PRNGKey(990),cfg,'ar',config['sample_count'],config['length'])
    ar_samples=[decode(x,chars) for x in ar_tokens]
    reference_rows={'ar':sample_metrics(ar_samples,reference,training)}
    ngram=NGram(data['train'],cfg.vocab_size,config['length'])
    for name,is_unigram in [('unigram',True),('trigram',False)]:
        texts=[decode(x,chars) for x in ngram.sample(990,config['sample_count'],config['length'],is_unigram)]
        reference_rows[name]=sample_metrics(texts,reference,training)
    write_json(out/'sampling_reference.json',dict(split='val',models=reference_rows))
    ablations=[]
    cases=[]
    for strategy in ['ancestral','random','confidence']:
        for steps in config['sampling_steps']:
            cases.append(dict(strategy=strategy,steps=steps,schedule='linear',temperature=1.,top_k=0))
    for schedule in ['cosine','log']:
        cases.append(dict(strategy='ancestral',steps=16,schedule=schedule,temperature=1.,top_k=0))
    for temperature,top_k in [(0.,0),(.7,0),(1.,5)]:
        cases.append(dict(strategy='confidence',steps=16,schedule='linear',temperature=temperature,top_k=top_k))
    for case in cases:
        tokens=generated(p,jax.random.PRNGKey(990),cfg,'mdlm',config['sample_count'],config['length'],**case)
        samples=[decode(x,chars) for x in tokens]
        row=dict(case,**sample_metrics(samples,reference,training),samples=samples)
        ablations.append(row)
        print(f"ABLATION {case}: JS3={row['js_3']:.4f}",flush=True)
        write_json(out/'sampling_ablations.json',ablations)
        jax.clear_caches()
    write_csv(out/'sampling_ablations.csv',[{k:v for k,v in r.items() if k!='samples'} for r in ablations])
