#!/usr/bin/env python3
"""Check completeness and consistency of a finished experiment bundle."""
import argparse
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/study')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    out=root/args.output
    config=json.loads((out/'config.json').read_text())
    expected=[f'{kind}_s{seed}' for kind in ['ar','mdlm','mlm'] for seed in config['seeds']]
    expected+=['mdlm_cosine_s0','mdlm_log_s0','unweighted_s0']
    checks=[]
    for name in expected:
        training=json.loads((out/name/'training.json').read_text())
        evaluation=json.loads((out/name/'evaluation.json').read_text())
        assert training['history'][-1]['step']==config['train_steps'],name
        assert evaluation['steps_trained']==config['train_steps'],name
        assert np.isfinite(evaluation['test']['mean']),name
        assert len(evaluation['samples'])==config['sample_count'],name
        if evaluation['kind']!='ar': assert evaluation['infilling']['unchanged_context'],name
        assert (out/name/'checkpoint.npz').is_file(),name
        checks.append(name+' complete')
    timings=json.loads((out/'timings.json').read_text())
    expected_timings=len(config['benchmark_shapes'])*(1+2*len(config['sampling_steps']))
    assert len(timings)==expected_timings,(len(timings),expected_timings)
    for row in timings:
        assert row['median_ms']>0 and row['tokens_per_second']>0
        assert len(row['timings_ms'])==config['timing_repeats']
        assert abs(row['tokens_per_second']-row['length']*row['batch']/row['median_ms']*1000)<1e-5
    checks.append(f'{len(timings)} benchmark cases consistent')
    ablations=json.loads((out/'sampling_ablations.json').read_text())
    assert len(ablations)==3*len(config['sampling_steps'])+5
    checks.append(f'{len(ablations)} sampling ablations complete')
    for path in ['REPORT.md','REPORT.html','figures/learning.png','figures/likelihood.png','figures/reconstruction.png','figures/tradeoff.png']:
        assert (out/path).stat().st_size>0,path
    tests=json.loads((root/'results/tests.json').read_text())
    assert tests['successful'] and all(r['status']=='passed' for r in tests['outcomes'])
    for name,digest in tests['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,f'Source changed after tests: {name}'
    checks.append(f'{tests["tests_run"]} tests passed')
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src') and not value.startswith(('https://','http://','#','data:')):
                    assert (out/value).is_file(),f'Missing report asset: {value}'
    Links().feed((out/'REPORT.html').read_text())
    checks.append('Report links and image files exist; tested source digests match')
    # Hash numerical records, configs and checkpoints; do not hash this output
    # recursively or incorporate transient Python/matplotlib caches.
    paths=sorted(set(out.glob('*.json'))|set(out.glob('*.csv'))|set(out.glob('*/*.json'))|set(out.glob('*/*.npz')))
    digests={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!='audit.json'}
    result=dict(successful=True,checks=checks,artifact_sha256=digests)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(successful=True,checks=checks),indent=2))


if __name__=='__main__':main()
