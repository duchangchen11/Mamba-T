"""Input allowlists, immutable-history checks, and source-only provenance."""
import json
import os
import subprocess
from pathlib import Path
from scripts.final_protocol import ROOT, SCENES, dump, utc_now, git_info, sha_file as raw_sha

RESULTS = ROOT/'results/social_distribution_shift'
PREVIOUS = ROOT/'results/training_regime_diagnostic'
CHECKPOINTS = ROOT/'checkpoints/training_regime_diagnostic'
BASE_COMMIT = '8672babecba5b25769e931862db26274e3cdf42a'
MODELS = ('emt_sr','ett_sr')
SEEDS = (42,123,2024)
HISTORY = ('results/eth_ucy_mamba_baseline','results/social_residual','results/clean_social_validation','results/final_eth_ucy_benchmark','results/training_regime_diagnostic')


def guard_path(path):
    text = os.path.normpath(str(path).replace('\\','/'))
    if 'final_eth_ucy_benchmark' in text.split('/'):
        raise RuntimeError('Formal test artifacts are inaccessible in the social shift audit')
    resolved = Path(path).resolve()
    if 'final_eth_ucy_benchmark' in resolved.parts:
        raise RuntimeError('Formal test artifacts are inaccessible, including symlinks')
    return resolved


def read(path):
    return json.loads(guard_path(path).read_text())


def sha_file(path):
    return raw_sha(guard_path(path))


def require_validation(mode):
    if mode not in ('val','source_validation'):
        raise PermissionError('Social shift audit only permits source validation')


def verify_history():
    subprocess.run(['git','diff','--exit-code',BASE_COMMIT,'--',*HISTORY,'src/models','configs/final_training_epochs.json'],cwd=ROOT,check=True,capture_output=True)
    raw = subprocess.check_output(['git','ls-tree','-r',BASE_COMMIT,'--',*HISTORY],cwd=ROOT,text=True)
    return {line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in raw.splitlines()}


def verify_inputs():
    p = read(RESULTS/'protocol_frozen.json')
    assert sha_file(RESULTS/'protocol_frozen.json') == (RESULTS/'protocol_frozen.sha256').read_text().strip()
    for name,expected in read(RESULTS/'data_audit/source_files.json')['sha256'].items():
        if sha_file(ROOT/name) != expected:
            raise ValueError(f'Frozen source dependency changed: {name}')
    verify_history()
    return p


def freeze_protocol():
    if (RESULTS/'protocol_frozen.json').exists():
        return verify_inputs()
    config = read(ROOT/'configs/social_distribution_shift.json')
    previous = read(PREVIOUS/'protocol_frozen.json')
    assert sha_file(PREVIOUS/'protocol_frozen.json') == (PREVIOUS/'protocol_frozen.sha256').read_text().strip()
    sources = {str(Path(p).relative_to(ROOT)):sha_file(p) for p in [*list((ROOT/'src/models').glob('*.py')),ROOT/'src/analysis/social_shift_features.py',*list((ROOT/'scripts').glob('social_shift_*.py')),ROOT/'scripts/run_social_distribution_shift.py',ROOT/'configs/social_distribution_shift.json',PREVIOUS/'protocol_frozen.json',PREVIOUS/'data_audit/checkpoint_inventory.json']}
    sources.update(previous['file_sha256'])
    # Only the 20 source-validation archives; each fold excludes its held-out scene.
    sources.update({p:s for p,s in previous['data_sha256'].items() if p.endswith('/val.npz')})
    for fold in SCENES:
        for seed in SEEDS:
            for model in ('emt','ett','emt_sr','ett_sr'):
                folder = PREVIOUS/f'heldout_{fold}'/model/f'seed_{seed}'
                q = read(folder/'metrics_validation.json')
                assert not q['heldout_test_accessed'] and q['historical_test_already_accessed']
                for name in ('metrics_validation.json','validation_samples.npz','validation_history.json'):
                    path=folder/name;sources[str(path.relative_to(ROOT))]=sha_file(path)
                assert sha_file(ROOT/q['checkpoint_path']) == q['checkpoint_sha256']
                sources[q['checkpoint_path']] = q['checkpoint_sha256']
    for path,expected in sources.items():
        assert sha_file(ROOT/path) == expected, path
    history = verify_history()
    dump(RESULTS/'data_audit/historical_git_inventory.json',history)
    dump(RESULTS/'data_audit/source_files.json',{'sha256':sources,'only_source_validation_inputs':True,'heldout_test_accessed':False,'historical_test_already_accessed':True})
    p = {**config,'code_commit':git_info()['commit'],'frozen_at_utc':utc_now(),'parent_results_commit':BASE_COMMIT,'source_input_files':len(sources)}
    dump(RESULTS/'protocol_frozen.json',p)
    (RESULTS/'protocol_frozen.sha256').write_text(sha_file(RESULTS/'protocol_frozen.json')+'\n')
    return p
