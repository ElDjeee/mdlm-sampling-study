"""One isolated benchmark process per shape/sampler; stdout is machine-readable."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json
import resource
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from .training import load_checkpoint
from .diffusion import diffusion_sample,ar_sample


def main():
    args=json.loads(sys.argv[1])
    state,cfg,_=load_checkpoint(args['checkpoint'])
    p=state['params']
    key=jax.random.PRNGKey(77)
    length,batch=args['length'],args['batch']
    if args['kind']=='ar':
        fn=jax.jit(lambda p,k:ar_sample(p,k,cfg,batch,length))
    else:
        initial=jnp.full((batch,length),cfg.mask_id,jnp.int32)
        fn=jax.jit(lambda p,k:diffusion_sample(p,k,initial,cfg,args['steps'],args['strategy'])[0])
    start=time.perf_counter()
    compiled=fn.lower(p,key).compile()
    compilation_seconds=time.perf_counter()-start
    compiled(p,key).block_until_ready()
    timings=[]
    for i in range(args['repeats']):
        sample_key=jax.random.fold_in(key,i)
        sample_key.block_until_ready()
        start=time.perf_counter()
        compiled(p,sample_key).block_until_ready()
        timings.append((time.perf_counter()-start)*1000)
    mem=compiled.memory_analysis()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin': rss*=1024
    median=float(np.median(timings))
    result=dict(median_ms=median,p10_ms=float(np.percentile(timings,10)),p90_ms=float(np.percentile(timings,90)),
                tokens_per_second=batch*length/(median/1000),timings_ms=timings,compilation_seconds=compilation_seconds,
                peak_process_rss_mib=rss/2**20,
                compiled_temp_bytes=int(mem.temp_size_in_bytes) if mem else None,
                compiled_argument_bytes=int(mem.argument_size_in_bytes) if mem else None,
                compiled_output_bytes=int(mem.output_size_in_bytes) if mem else None,
                memory_note='Fresh-process CPU RSS high-water mark includes Python, JAX and compilation; not GPU VRAM or inference-only memory',
                backend=jax.default_backend(),devices=[str(d) for d in jax.devices()])
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__': main()
