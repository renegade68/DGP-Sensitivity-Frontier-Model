"""Prepare source/design in a fresh workspace; never launch estimators."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def schedules():
    cells = json.loads((ROOT / 'design/EFFECTIVE_DGP.json').read_text())['cells']
    cells = sorted(cells, key=lambda c: c['cell_id'])
    seeds, used, collisions = {}, set(), []
    for cell in cells:
        for rep in range(1, cell['target_replications'] + 1):
            key = f"{cell['cell_id']}_r{rep:04d}"
            attempt = 0
            while True:
                payload = [20261009, key, 'EAT', 'numStop5_fold5', 'algorithm', attempt]
                digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                value = int(digest[:16], 16) % 2147483646 + 1
                if value not in used:
                    break
                attempt += 1
            used.add(value)
            seeds[key] = value
            if attempt:
                collisions.append({'task': key, 'rehashed': attempt})
    rng = {'eat_seeds': seeds, 'seed_collision_resolutions': collisions}
    rng.update(json.loads((ROOT / 'design/RNG_TEMPLATE.json').read_text()))
    tasks = [dict(task=f"{c['cell_id']}_r{rep:04d}", cell_id=c['cell_id'], replication_id=rep, cohort='production')
             for rep in range(1, max(c['target_replications'] for c in cells) + 1)
             for c in cells if rep <= c['target_replications']]
    values = {'RNG.json': encoded(rng), 'PRODUCTION_TASKS.json': encoded(tasks)}
    expected = json.loads((ROOT / 'design/GENERATED_HASHES.json').read_text())
    for name, data in values.items():
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError('Frozen schedule mismatch: ' + name)
    return values


def prepare(output):
    output = output.resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        raise ValueError('Use a new directory outside the checkout')
    if not output.parent.is_dir():
        raise ValueError('Output parent directory must already exist')
    values = schedules()  # Fail before writing if any frozen schedule differs.
    with tempfile.TemporaryDirectory(prefix='.frontier-prepare-', dir=output.parent) as tmp:
        stage = Path(tmp) / 'workspace'
        stage.mkdir()
        for folder in ['tools', 'design']:
            shutil.copytree(ROOT / folder, stage / folder)
        for name, data in values.items():
            (stage / 'design' / name).write_bytes(data)
        shutil.copytree(ROOT / 'environment', stage / 'environment')
        for folder in ['validation', 'evidence', 'vendor', 'runs']:
            (stage / folder).mkdir()
        status = {'workspace_prepared': True, 'production_enabled': False,
                  'runtime_configured': False, 'scientific_readiness_certified': False,
                  'estimator_fits': 0, 'tasks': 27000,
                  'next': 'Configure and qualify a fresh runtime; see repository docs/RUNNING.md.'}
        (stage / 'PREPARATION.json').write_bytes(encoded(status))
        stage.rename(output)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
