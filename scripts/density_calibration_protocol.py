"""Pinned source-only density calibration protocol and formal-test path guards."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.final_protocol import ROOT, SCENES, dump, git_info, utc_now

PREVIOUS_COMMIT = 'eba8b4e000de29ccbd4f8a66d26344f9669e2c1a'
RESULTS = ROOT/'results/density_aware_social_calibration'
CHECKPOINTS = ROOT/'checkpoints/density_aware_social_calibration'
DIAGNOSTIC = ROOT/'results/training_regime_diagnostic'
DIAGNOSTIC_CKPTS = ROOT/'checkpoints/training_regime_diagnostic'
MODELS = ('emt', 'emt_sr', 'emt_density_sr')
SEEDS = (42, 123, 2024)
FROZEN_RESULTS = (
    'results/eth_ucy_mamba_baseline', 'results/social_residual',
    'results/clean_social_validation', 'results/final_eth_ucy_benchmark',
    'results/training_regime_diagnostic', 'results/social_distribution_shift',
)
FROZEN_CODE = (
    'src/models/mamba_trajectory.py', 'src/models/trajectory_transformer.py',
    'src/models/social_attention.py', 'src/models/social_residual.py',
    'src/data/eth_ucy_training_regime_dataset.py', 'src/data/eth_ucy_social_clean_dataset.py',
    'scripts/eth_ucy_utils.py', 'scripts/training_regime_utils.py',
        'configs/final_training_epochs.json', 'configs/training_regime_diagnostic.json',
)


def guard_path(path):
    text = os.path.normpath(str(path).replace('\\', '/'))
    if 'final_eth_ucy_benchmark' in text.split('/'):
        raise RuntimeError('Formal held-out artifacts are inaccessible in this stage')
    resolved = Path(path).resolve()
    if 'final_eth_ucy_benchmark' in resolved.parts:
        raise RuntimeError('Formal held-out artifacts are inaccessible, including symlinks')
    return resolved


def require_source_mode(mode):
    if mode not in ('train', 'val', 'source_validation'):
        raise PermissionError('Only source train/validation modes are allowed')


def sha_file(path):
    path = guard_path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(guard_path(path).read_text())


def verify_history():
    subprocess.run(
        ['git', 'diff', '--exit-code', PREVIOUS_COMMIT, '--', *FROZEN_RESULTS, *FROZEN_CODE],
        cwd=ROOT, check=True, capture_output=True,
    )
    raw = subprocess.check_output(
        ['git', 'ls-tree', '-r', PREVIOUS_COMMIT, '--', *FROZEN_RESULTS],
        cwd=ROOT, text=True,
    )
    return {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2] for line in raw.splitlines()}


def source_input_paths():
    data_root = ROOT/'data/processed/eth_ucy_social_clean'
    data = sorted(p for p in data_root.rglob('*.npz') if p.name in ('train.npz', 'val.npz'))
    if not data or any('test' in p.parts for p in data):
        raise RuntimeError('Source dataset allowlist is empty or contains a test path')
    artifacts = []
    for fold in SCENES:
        for seed in SEEDS:
            for model in ('emt', 'emt_sr'):
                folder = DIAGNOSTIC/f'heldout_{fold}'/model/f'seed_{seed}'
                artifacts.extend(folder/name for name in (
                    'metrics_validation.json', 'validation_history.json',
                    'validation_samples.npz', 'initialization_report.json',
                ))
                ckpt = DIAGNOSTIC_CKPTS/f'heldout_{fold}'/model/f'seed_{seed}.pt'
                artifacts.append(ckpt)
    return data, artifacts


def freeze_protocol():
    config_path = ROOT/'configs/density_aware_social_calibration.json'
    config = read_json(config_path)
    if config['parent_commit'] != PREVIOUS_COMMIT:
        raise ValueError('Unexpected frozen parent')
    existing = RESULTS/'protocol_frozen.json'
    if existing.exists():
        protocol = read_json(existing)
        if sha_file(existing) != (RESULTS/'protocol_frozen.sha256').read_text().strip():
            raise ValueError('Density calibration protocol hash mismatch')
        for name, expected in read_json(RESULTS/'data_audit/source_files.json')['sha256'].items():
            if sha_file(ROOT/name) != expected:
                raise ValueError(f'Frozen input changed: {name}')
        verify_history()
        return protocol
    data, artifacts = source_input_paths()
    code = [config_path, *(ROOT/p for p in FROZEN_CODE), ROOT/'src/models/density_aware_social.py', ROOT/'scripts/density_calibration_protocol.py', ROOT/'scripts/run_density_aware_social_calibration.py', ROOT/'scripts/summarize_density_aware_social_calibration.py', ROOT/'tests/test_density_aware_social_calibration.py']
    inputs = {str(p.relative_to(ROOT)): sha_file(p) for p in [*code, *data, *artifacts]}
    history = verify_history()
    dump(RESULTS/'data_audit/source_files.json', {
        'sha256': inputs, 'input_file_count': len(inputs), 'source_train_validation_archives_only': len(data),
        'baseline_emt_and_emt_sr_only': True, 'formal_test_artifacts_read': False,
        'heldout_test_accessed': False, 'historical_test_already_accessed': True,
    })
    final_epochs = read_json(ROOT/'configs/final_training_epochs.json')
    if set(final_epochs) != set(SCENES) or any('emt_sr' not in final_epochs[f] for f in SCENES):
        raise ValueError('Fixed EMT social epochs are incomplete')
    protocol = {
        **config, 'code_commit': git_info()['commit'], 'frozen_at_utc': utc_now(),
        'final_epochs': final_epochs, 'source_input_files': len(inputs),
        'historical_frozen_tracked_files': len(history),
    }
    dump(existing, protocol)
    (RESULTS/'protocol_frozen.sha256').write_text(sha_file(existing)+'\n')
    return protocol
