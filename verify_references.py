#!/usr/bin/env python3
"""Replay static source observations, without importing or executing third-party code."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkouts', type=Path, required=True,
                        help='Directory with vadd-audit, mdm-prime-audit, lomdm-audit, tiny-diffusion-audit')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    audit = json.loads((root/'references/independent_audit.json').read_text())
    names = ['vadd-audit', 'mdm-prime-audit', 'lomdm-audit', 'tiny-diffusion-audit']
    inspected = [
        ['README.md', 'configs/config.yaml', 'diffusion.py', 'models/dit.py',
         'scripts/train_owt_vadd.sh', 'scripts/eval_owt_vadd.sh'],
        ['README.md', 'text/README.md', 'text/diffusion.py', 'text/configs/config.yaml',
         'text/configs/data/openwebtext-split.yaml', 'text/assets/curves.png'],
        ['README.md', 'scripts/train_owt_mdlm.sh', 'scripts/zero_shot_eval_mdlm_example.sh'],
        ['README.md', 'diffusion.py']]
    records = []
    for entry, name, files in zip(audit['repositories'], names, inspected):
        path = args.checkouts/name
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        assert revision == entry['commit'], f'Revision mismatch: {name}'
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
        assert not dirty, f'Checkout modified: {name}'
        tracked = subprocess.check_output(['git', '-C', str(path), 'ls-files'], text=True).splitlines()
        observations = []
        if name == 'vadd-audit':
            assert not any(p.startswith('configs/data/') for p in tracked)
            assert '/data: openwebtext-split' in (path/'configs/config.yaml').read_text()
            observations.append('Required data config directory absent from tracked files.')
        if name == 'mdm-prime-audit':
            assert (path/'text/configs/data/openwebtext-split.yaml').is_file()
            source = (path/'text/diffusion.py').read_text()
            assert 'self.config.prime.target_length == 1' in source
            assert 'def _subs_parameterization' in source
            observations.append('Standard branch and OWT config present; no numerical equivalence asserted.')
        if name == 'tiny-diffusion-audit':
            assert '(loss * mask_flat).sum() / mask_flat.sum()' in (path/'diffusion.py').read_text()
            observations.append('Masked-count loss normalization found.')
        records.append(dict(repository=entry['id'], commit=revision,
                            checks=observations,
                            source_sha256={p: hashlib.sha256((path/p).read_bytes()).hexdigest() for p in files}))
    result = dict(static_checks_passed=True, numerical_validation_performed=False,
                  scope='Pinned clean source checkouts and selected static observations only; no network or model execution.',
                  repositories=records)
    (root/'references/source_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Four pinned source checkouts verified; no third-party numerical evaluation performed.')


if __name__ == '__main__':
    main()
