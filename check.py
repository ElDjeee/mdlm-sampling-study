#!/usr/bin/env python3
"""Execute tests and save auditable outcomes with a source digest."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
import hashlib
import json
import time
import unittest
from pathlib import Path


class Result(unittest.TextTestResult):
    def startTestRun(self):
        super().startTestRun()
        self.outcomes=[]
    def addSuccess(self,test):
        super().addSuccess(test);self.outcomes.append(dict(test=test.id(),status='passed'))
    def addFailure(self,test,err):
        super().addFailure(test,err);self.outcomes.append(dict(test=test.id(),status='failed'))
    def addError(self,test,err):
        super().addError(test,err);self.outcomes.append(dict(test=test.id(),status='error'))
    def addSkip(self,test,reason):
        super().addSkip(test,reason);self.outcomes.append(dict(test=test.id(),status='skipped',reason=reason))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/tests.json')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    suite=unittest.defaultTestLoader.discover(str(root/'tests'))
    before=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    digests={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ['nanodiff','tests'] for p in sorted((root/folder).glob('*.py'))}
    output=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),duration_seconds=time.perf_counter()-before,
                tests_run=result.testsRun,successful=result.wasSuccessful(),outcomes=result.outcomes,source_sha256=digests,
                backend=os.environ['JAX_PLATFORMS'],gpu_tested=False,
                gpu_note='JAX Metal initialization aborts on this host; CPU tested. CUDA support is unverified.')
    path=root/args.output;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
