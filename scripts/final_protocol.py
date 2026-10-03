"""Hash-pinned protocol and complete-matrix held-out evaluation guard."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from src.data.eth_ucy_dataset import ROOT, SCENES

RESULTS = ROOT/'results/final_eth_ucy_benchmark'
CHECKPOINTS = ROOT/'checkpoints/final_eth_ucy_benchmark'
MODELS = ('ett', 'emt', 'ett_sr', 'emt_sr')
SEEDS = (42, 123, 2024)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def git_info():
    return {k: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip() for k, args in {'branch': ['branch', '--show-current'], 'commit': ['rev-parse', 'HEAD']}.items()}


def verify_protocol(root=ROOT, results=RESULTS):
    protocol = read(results/'protocol_frozen.json')
    expected = (results/'protocol_frozen.sha256').read_text().strip()
    if sha_file(results/'protocol_frozen.json') != expected:
        raise ValueError('Frozen protocol hash changed')
    for key in ('config_sha256', 'code_sha256', 'dataset_sha256'):
        for path, digest in protocol[key].items():
            if sha_file(root/path) != digest:
                raise ValueError(f'Frozen {key} changed: {path}')
    return protocol, expected


def verify_history(root=ROOT, results=RESULTS):
    inventory = read(results/'data_audit/historical_frozen_inventory.json')
    for path, digest in inventory.items():
        if sha_file(root/path) != digest:
            raise ValueError(f'Historical artifact changed: {path}')
    return len(inventory)


def verify_training_complete(root=ROOT, results=RESULTS, checkpoints=CHECKPOINTS):
    protocol, psha = verify_protocol(root, results)
    complete = read(results/'FINAL_TRAINING_COMPLETE.json')
    if complete.get('completed_runs') != 60 or complete.get('protocol_sha256') != psha or complete.get('test_metrics_accessed_before_freeze') is not False:
        raise ValueError('All 60 runs must be frozen before test evaluation')
    keys = set()
    for q in complete['runs']:
        key = (q['fold'], q['model'], q['seed'])
        if key in keys:
            raise ValueError('Duplicate final training run')
        keys.add(key)
        path = root/q['checkpoint_path']
        if sha_file(path) != q['checkpoint_sha256']:
            raise ValueError('Final checkpoint changed')
        report = read(results/'training'/f"heldout_{q['fold']}"/q['model']/f"seed_{q['seed']}"/'training_report.json')
        folder = results/'training'/f"heldout_{q['fold']}"/q['model']/f"seed_{q['seed']}"
        if sha_file(folder/'training_report.json') != q['training_report_sha256'] or sha_file(folder/'training_history.json') != q['training_history_sha256']:
            raise ValueError('Frozen training log changed')
        if report['protocol_sha256'] != psha or report['nan_inf'] or report['epochs_run'] != protocol['final_epochs'][q['fold']][q['model']]:
            raise ValueError('Training metadata violates frozen protocol')
        if report['checkpoint_sha256'] != q['checkpoint_sha256'] or report['heldout_test_evaluated']:
            raise ValueError('Invalid final training checkpoint record')
    expected = {(f, m, s) for f in SCENES for m in MODELS for s in SEEDS}
    if keys != expected:
        raise ValueError('Incomplete final training matrix')
    return protocol, psha, complete


def require_heldout_evaluation(enabled, **kwargs):
    if not enabled:
        raise PermissionError('Explicit --enable-heldout-evaluation is required')
    return verify_training_complete(**kwargs)
