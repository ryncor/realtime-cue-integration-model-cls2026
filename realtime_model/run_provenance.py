from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def digest(path: Path, method: str = 'md5') -> str:
    h = hashlib.new(method)
    with Path(path).open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(dest: Path, values: dict[str, object]) -> None:
    dest = Path(dest)
    temp = dest.with_suffix(dest.suffix + '.tmp')
    with temp.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(('key', 'value'))
        for key, value in values.items():
            writer.writerow((key, value))
    temp.replace(dest)


def read_manifest(path: Path) -> dict[str, str]:
    if not Path(path).is_file():
        raise RuntimeError(f'Missing completed-run manifest: {path}')
    with Path(path).open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    if any(set(row) != {'key', 'value'} for row in rows):
        raise RuntimeError(f'Invalid completed-run manifest: {path}')
    values = {row['key']: row['value'] for row in rows}
    if len(values) != len(rows):
        raise RuntimeError(f'Duplicate manifest key: {path}')
    return values


def verify_results(root: Path) -> dict[str, str]:
    """Reject stale or mixed controller/model/audit outputs before visualization."""
    root = Path(root)
    basedir = root / 'demos/output/learning'
    learned = read_manifest(basedir / 'completed_learning_provenance.csv')
    audited = read_manifest(basedir / 'boundary_locking_audit/completed_audit_provenance.csv')
    fields = {'controller_md5', 'model_md5', 'seed', 'n_test'}
    if not fields <= learned.keys() or not fields <= audited.keys():
        raise RuntimeError('Missing controller/model/run metadata in manifests')
    for field in fields:
        if learned[field] != audited[field]:
            raise RuntimeError(f'Mixed controller/model/run provenance: {field}')
    paths = [
        (root / 'realtime_model/phonetic_competition_model.py', learned, 'controller_md5'),
        (basedir / 'phonetic_full_inventory_model.npz', learned, 'model_md5'),
        (basedir / 'commitment_timing_trials.csv', learned, 'timing_csv_md5'),
        (basedir / 'boundary_locking_audit/matched_duration_trials.csv', audited, 'matched_csv_md5'),
    ]
    for path, manifest, field in paths:
        if field not in manifest or not path.is_file() or digest(path) != manifest[field]:
            raise RuntimeError(f'Stale or missing result: {path}')
    return {'controller_md5': learned['controller_md5'],
            'model_md5': learned['model_md5'],
            'seed': learned['seed'], 'n_test': learned['n_test']}
