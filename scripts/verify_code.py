"""Check tracked-file scope, source preservation, syntax, and frozen schedules."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from prepare_workspace import schedules

ROOT = Path(__file__).resolve().parents[1]


def verify():
    allowed = set((ROOT / 'TRACKED_FILES.txt').read_text().splitlines())
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(ROOT).parts}
    if actual != allowed:
        raise ValueError({'extra_files': sorted(actual - allowed), 'missing_files': sorted(allowed - actual)})
    if (ROOT / '.git').exists():
        tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().strip('\0').split('\0')) - {''}
        if tracked - allowed:
            raise ValueError('Out-of-scope files tracked: ' + repr(sorted(tracked - allowed)))
    provenance = json.loads((ROOT / 'SOURCE_PROVENANCE.json').read_text())
    for name, record in provenance['copied_files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Preserved source changed; version and validate deliberately: ' + name)
    python_files = [ROOT / n for n in allowed if n.endswith('.py')]
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    values = schedules()
    tasks = json.loads(values['PRODUCTION_TASKS.json'])
    cells = json.loads((ROOT / 'design/EFFECTIVE_DGP.json').read_text())['cells']
    if len(cells) != 117 or len(tasks) != 27000 or len({t['task'] for t in tasks}) != 27000:
        raise ValueError('Design scope or task uniqueness differs')
    if sum(c['target_replications'] == 500 for c in cells) != 12:
        raise ValueError('Precision-extension scope differs')
    return {'status': 'PASS', 'files': len(allowed), 'python_syntax_files': len(python_files),
            'unchanged_source_files': len(provenance['copied_files']),
            'conditions': len(cells), 'tasks': len(tasks), 'generated_schedules_exact': True,
            'estimator_fits': 0}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
