"""Derive epochs exclusively from frozen validation, then pin pre-test protocol."""
import argparse
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.final_protocol import ROOT, RESULTS, SCENES, MODELS, SEEDS, dump, read, sha_file, utc_now, git_info, verify_protocol


def derive_epochs():
    epochs, provenance = {}, {}
    for fold in SCENES:
        epochs[fold], provenance[fold] = {}, {}
        for model in MODELS:
            directory = 'clean_social_validation' if model.endswith('_sr') else 'eth_ucy_mamba_baseline'
            paths = [ROOT/'results'/directory/f'heldout_{fold}'/model/f'seed_{seed}'/'metrics_validation.json' for seed in SEEDS]
            best = [read(p)['best_epoch'] for p in paths]
            epochs[fold][model] = int(round(statistics.median(best)))
            provenance[fold][model] = {'seeds': list(SEEDS), 'validation_best_epochs': best, 'final_epoch': epochs[fold][model], 'sources': {str(p.relative_to(ROOT)): sha_file(p) for p in paths}}
    return epochs, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-only', action='store_true')
    args = parser.parse_args()
    epochs, provenance = derive_epochs()
    path = ROOT/'configs/final_training_epochs.json'
    if path.exists():
        if read(path) != epochs:
            raise ValueError('Existing final epochs disagree with frozen validation')
    else:
        dump(path, epochs)
    dump(RESULTS/'data_audit/epoch_determination.json', {'method': 'per fold/model median of validation best_epoch over seeds 42,123,2024; nearest integer if needed', 'epochs': epochs, 'source_validation': provenance, 'final_epoch_config_sha256': sha_file(path), 'test_metrics_accessed_before_freeze': False})
    if args.epochs_only:
        print(epochs)
        return
    if (RESULTS/'protocol_frozen.json').exists():
        _, psha = verify_protocol()
        print('Existing frozen protocol verified:', psha)
        return
    import subprocess
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src', 'scripts', 'configs'], cwd=ROOT, check=True)
    config = read(ROOT/'configs/final_eth_ucy_benchmark.json')
    temporal = read(ROOT/'configs/eth_ucy_mamba_baseline.json')
    inventory = {}
    for directory in ('results/eth_ucy_mamba_baseline', 'results/social_residual', 'results/clean_social_validation', 'checkpoints/eth_ucy_mamba_baseline', 'checkpoints/social_residual', 'checkpoints/clean_social_validation', 'data/processed/eth_ucy_social_clean'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file():
                inventory[str(p.relative_to(ROOT))] = sha_file(p)
    dump(RESULTS/'data_audit/historical_frozen_inventory.json', inventory)
    code = {str(p.relative_to(ROOT)): sha_file(p) for folder in ('src', 'scripts') for p in (ROOT/folder).rglob('*.py')}
    configs = {str(p.relative_to(ROOT)): sha_file(p) for p in (ROOT/'configs').glob('*.json')}
    datasets = {str(p.relative_to(ROOT)): sha_file(p) for directory in ('data/processed/eth_ucy', 'data/processed/eth_ucy_social', 'data/processed/eth_ucy_social_final') for p in (ROOT/directory).rglob('*.npz')}
    protocol = {**config, 'frozen_at_utc': utc_now(), 'code_commit': git_info()['commit'], 'branch': git_info()['branch'], 'final_epochs': epochs, 'epoch_determination': 'median of three prior validation best epochs per fold/model', 'fold_definitions': {f: {'train': [s for s in SCENES if s != f], 'test': [f]} for f in SCENES}, 'temporal_architectures': temporal, 'trajectory_decoder': ['LayerNorm(128)', 'Linear(128,128)', 'GELU', 'Dropout(0.1)', 'Linear(128,24)'], 'social_cross_attention': {'relation_embedding': ['Linear(3,32)', 'GELU', 'Linear(32,128)'], 'attention': 'MultiheadAttention(embed_dim=128,num_heads=4,dropout=0.1,batch_first=True)', 'all_masked_rows': 'zero context'}, 'residual_decoder': ['Linear(256,128)', 'GELU', 'Dropout(0.1)', 'Linear(128,24), last weight/bias zero'], 'social_backbone': 'corresponding final full-source fold/seed backbone, frozen/eval; old validation checkpoints prohibited', 'test_metrics_accessed_before_freeze': False, 'config_sha256': configs, 'code_sha256': code, 'dataset_sha256': datasets, 'historical_frozen_inventory_sha256': sha_file(RESULTS/'data_audit/historical_frozen_inventory.json')}
    dump(RESULTS/'protocol_frozen.json', protocol)
    psha = sha_file(RESULTS/'protocol_frozen.json')
    (RESULTS/'protocol_frozen.sha256').write_text(psha+'\n')
    dump(RESULTS/'data_audit/protocol_hashes.json', {'protocol_frozen_sha256': psha, 'config_sha256': configs, 'code_sha256': code, 'dataset_sha256': datasets, 'test_metrics_accessed_before_freeze': False})
    dump(RESULTS/'data_audit/test_access_log.json', {'test_metrics_accessed_before_freeze': False, 'protocol_frozen_sha256': psha, 'evaluations': []})
    print('Protocol frozen:', psha)


if __name__ == '__main__':
    main()
