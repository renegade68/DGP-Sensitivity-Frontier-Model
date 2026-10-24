"""Seal independent SFA validation outputs for the prediction comparison.

Checks recorded input hashes and writes a new manifest; no fitting or reporting.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def seal(directory):
    directory = directory.resolve()
    for name in ['SOURCE_HASHES.json', 'SUMMARY.json', 'INPUTS.csv', 'R_RESULTS.json',
                 'R_RESULTS.rds', 'R_PARAMS_INDEX.json', 'R_PARAMS.f64', 'R_BEST_BY_CASE.csv']:
        if not (directory / name).is_file():
            raise ValueError('Missing validation output: ' + name)
    sources = json.loads((directory / 'SOURCE_HASHES.json').read_text())
    for name, digest in sources.items():
        if sha(Path(name)) != digest:
            raise ValueError('Validation input changed: ' + name)
    files = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'MANIFEST.json' and '__pycache__' not in p.parts}
    with (directory / 'MANIFEST.json').open('x') as stream:
        json.dump({'files': files, 'recorded_inputs_unchanged': True,
                   'scope': 'Integrity manifest only; not a scientific pass/fail certificate'}, stream, indent=2)
        stream.write('\n')
    return {'sealed_files': len(files), 'checked_inputs': len(sources), 'estimator_fits': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(seal(args.directory), indent=2))
