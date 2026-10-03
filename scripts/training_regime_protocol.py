"""Pinned source-only protocol; historical test values never loaded from artifacts."""
import subprocess
from pathlib import Path
from src.data.eth_ucy_dataset import ROOT, SCENES
from scripts.final_protocol import dump, read, sha_file, utc_now, git_info

RESULTS = ROOT/'results/training_regime_diagnostic'
CHECKPOINTS = ROOT/'checkpoints/training_regime_diagnostic'
MODELS = ('ett', 'emt', 'ett_sr', 'emt_sr')
SEEDS = (42, 123, 2024)
HISTORICAL_RESULTS = ('results/eth_ucy_mamba_baseline', 'results/social_residual', 'results/clean_social_validation', 'results/final_eth_ucy_benchmark')


def historical_git_inventory():
    # Git object identities only: do not load historical test metrics or predictions.
    raw = subprocess.check_output(['git', 'ls-tree', '-r', 'a69b8f8d2703212cc508c4dddd4a00b1e333dfcd', '--', *HISTORICAL_RESULTS], cwd=ROOT, text=True)
    return {line.split('\t', 1)[1]: line.split('\t', 1)[0].split()[2] for line in raw.splitlines()}


def verify_historical_results():
    subprocess.run(['git', 'diff', '--exit-code', 'a69b8f8d2703212cc508c4dddd4a00b1e333dfcd', '--', *HISTORICAL_RESULTS], cwd=ROOT, check=True, capture_output=True)
    return len(historical_git_inventory())


def verify_protocol():
    p = read(RESULTS/'protocol_frozen.json')
    psha = (RESULTS/'protocol_frozen.sha256').read_text().strip()
    if sha_file(RESULTS/'protocol_frozen.json') != psha:
        raise ValueError('Diagnostic protocol changed')
    for name in ('file_sha256', 'data_sha256'):
        for path, expected in p[name].items():
            if sha_file(ROOT/path) != expected:
                raise ValueError(f'Frozen diagnostic dependency changed: {path}')
    return p, psha


def freeze_protocol():
    if (RESULTS/'protocol_frozen.json').exists():
        return verify_protocol()
    config = read(ROOT/'configs/training_regime_diagnostic.json')
    epochs = read(ROOT/config['epoch_config'])
    assert set(epochs) == set(SCENES) and all(set(q) == set(MODELS) for q in epochs.values())
    configs = [ROOT/'configs/training_regime_diagnostic.json', ROOT/'configs/final_training_epochs.json', ROOT/'configs/eth_ucy_mamba_baseline.json', ROOT/'configs/clean_social_validation.json']
    sources = [ROOT/'src/data/eth_ucy_training_regime_dataset.py', ROOT/'src/data/eth_ucy_social_clean_dataset.py', ROOT/'scripts/training_regime_protocol.py', ROOT/'scripts/training_regime_metrics.py', ROOT/'scripts/training_regime_utils.py', ROOT/'scripts/run_training_regime_diagnostic.py', ROOT/'scripts/summarize_training_regime_diagnostic.py', ROOT/'scripts/eth_ucy_utils.py', ROOT/'scripts/social_residual_utils.py', *list((ROOT/'src/models').glob('*.py'))]
    manifests = [ROOT/'results/eth_ucy_mamba_baseline/data_audit'/f'manifest_{f}.json' for f in SCENES]
    p = {**config, 'final_epochs': epochs, 'code_commit': git_info()['commit'], 'frozen_at_utc': utc_now(), 'file_sha256': {str(path.relative_to(ROOT)): sha_file(path) for path in configs+sources+manifests}, 'data_sha256': {str(path.relative_to(ROOT)): sha_file(path) for path in (ROOT/'data/processed/eth_ucy_social_clean').rglob('*.npz')}, 'historical_formal_context': {'source': 'user-supplied aggregate values in stage request; no test artifacts loaded or recomputed', 'ett': {'ADE': .608966, 'FDE': 1.240611}, 'emt': {'ADE': .600249, 'FDE': 1.241118}, 'ett_sr': {'ADE': .596162, 'FDE': 1.219276}, 'emt_sr': {'ADE': .618485, 'FDE': 1.283079}}}
    dump(RESULTS/'data_audit/historical_results_git_inventory.json', historical_git_inventory())
    dump(RESULTS/'protocol_frozen.json', p)
    psha = sha_file(RESULTS/'protocol_frozen.json')
    (RESULTS/'protocol_frozen.sha256').write_text(psha+'\n')
    dump(RESULTS/'data_audit/access_guard.json', {'allowed_modes': ['train', 'val'], 'rejected_mode': 'heldout_test', 'formal_test_artifacts_loaded_for_analysis': False, 'heldout_test_accessed': False, 'historical_test_already_accessed': True, 'protocol_sha256': psha})
    return p, psha
