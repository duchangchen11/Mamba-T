"""Exactly one held-out prediction pass, only after complete formal training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time
import numpy as np
import torch
from src.data.eth_ucy_final_dataset import ETHUCYFinalDataset
from src.data.eth_ucy_social_final_dataset import ETHUCYSocialFinalDataset
from src.data.eth_ucy_social_dataset import neighbor_statistics
from scripts.final_protocol import ROOT, RESULTS, SCENES, MODELS, SEEDS, dump, read, sha_file, utc_now, git_info, require_heldout_evaluation, verify_history
from scripts.final_training_utils import load_final_model
from scripts.final_prediction_export import predict_export, save_prediction_artifact


@torch.no_grad()
def benchmark_final(model, social):
    model.eval()
    result = {}
    for size in (1, 128):
        x = torch.zeros(size, 8, 4, device='cuda')
        if social:
            batch = {'target_history': x, 'neighbor_history': torch.zeros(size, 8, 8, 4, device='cuda'), 'neighbor_mask': torch.ones(size, 8, dtype=torch.bool, device='cuda'), 'neighbor_relation': torch.zeros(size, 8, 3, device='cuda')}
            forward = lambda: model(batch['target_history'], batch['neighbor_history'], batch['neighbor_mask'], batch['neighbor_relation'])
        else:
            forward = lambda: model(x)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(20):
            forward()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            forward()
        torch.cuda.synchronize()
        result[f'batch_{size}_latency_ms'] = (time.perf_counter()-start)*10
        result[f'batch_{size}_peak_memory_bytes'] = torch.cuda.max_memory_allocated()
    result['method'] = 'exclusive GPU, full uncached eval/no_grad, zeros, social N=8, 20 warmups +100 synchronized forwards; no transfers or attention diagnostics'
    return result


def begin_access(log_path, fold, name, seed, psha, checkpoint_sha, complete):
    log = read(log_path)
    if any((q['fold'], q['model'], q['seed']) == (fold, name, seed) for q in log['evaluations']):
        raise RuntimeError('Held-out evaluation already started; replay prohibited')
    entry = {'timestamp': utc_now(), **git_info(), 'protocol_sha256': psha, 'checkpoint_sha256': checkpoint_sha, 'fold': fold, 'heldout_scene': fold, 'model': name, 'seed': seed, 'status': 'started', 'training_completed_at_utc': complete['completed_at_utc'], 'training_complete_sha256': sha_file(RESULTS/'FINAL_TRAINING_COMPLETE.json')}
    log['evaluations'].append(entry)
    dump(log_path, log)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-heldout-evaluation', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol, psha, complete = require_heldout_evaluation(args.enable_heldout_evaluation)
    verify_history()
    log_path = RESULTS/'data_audit/test_access_log.json'
    for fold in SCENES:
        for name in MODELS:
            social = name.endswith('_sr')
            for seed in SEEDS:
                destination = RESULTS/'test'/f'heldout_{fold}'/name/f'seed_{seed}'
                metrics_path = destination/'metrics_test.json'
                if metrics_path.exists():
                    prior = read(metrics_path)
                    assert prior['protocol_sha256'] == psha and prior['heldout_test_evaluated']
                    assert sha_file(ROOT/prior['prediction_artifact_path']) == prior['prediction_artifact_sha256']
                    continue
                training = read(RESULTS/'training'/f'heldout_{fold}'/name/f'seed_{seed}'/'training_report.json')
                model = load_final_model(fold, name, seed, psha).eval()
                runtime = benchmark_final(model, social)
                dataset = ETHUCYSocialFinalDataset(fold, 'heldout_test') if social else ETHUCYFinalDataset(fold, 'heldout_test')
                assert tuple(dataset.scenes) == (fold,)
                entry = begin_access(log_path, fold, name, seed, psha, training['checkpoint_sha256'], complete)
                outputs = []
                with torch.no_grad():
                    for i in range(0, len(dataset), 128):
                        batch = {k: v[i:i+128].cuda() for k, v in dataset.arrays.items()}
                        output = predict_export(model, batch, social)
                        outputs.append({k: v.cpu().numpy() for k, v in output.items()})
                artifact = RESULTS/'predictions'/f'heldout_{fold}'/name/f'seed_{seed}.npz'
                measured = save_prediction_artifact(artifact, dataset, outputs, social)
                q = {'fold': fold, 'heldout_scene': fold, 'model': name, 'seed': seed, 'train_scene_list': training['train_scene_list'], 'train_sample_count': training['train_sample_count'], 'test_sample_count': len(dataset), 'training_epoch': training['training_epoch'], 'test_ADE': measured['ADE'], 'test_FDE': measured['FDE'], 'parameter_count': training['parameter_count'], 'checkpoint_sha256': training['checkpoint_sha256'], 'protocol_sha256': psha, 'nan_inf': False, 'heldout_test_evaluated': True, 'test_metrics_accessed_before_freeze': False, 'prediction_artifact_path': str(artifact.relative_to(ROOT)), 'prediction_artifact_sha256': sha_file(artifact), 'runtime': runtime, 'evaluated_at_utc': utc_now()}
                if social:
                    q['neighbor_statistics'] = neighbor_statistics(dataset.arrays['neighbor_mask'].numpy())
                dump(metrics_path, q)
                log = read(log_path)
                for record in log['evaluations']:
                    if (record['fold'], record['model'], record['seed']) == (fold, name, seed):
                        record.update({'status': 'complete', 'completed_at_utc': q['evaluated_at_utc'], 'metrics_sha256': sha_file(metrics_path), 'prediction_artifact_sha256': q['prediction_artifact_sha256']})
                dump(log_path, log)
                print(f'HELDOUT TEST COMPLETE {fold} {name} {seed}: ADE={measured["ADE"]:.6f} FDE={measured["FDE"]:.6f}', flush=True)
                del model, dataset, outputs
                torch.cuda.empty_cache()
    verify_history()
    print('All 60 one-pass held-out evaluations complete', flush=True)


if __name__ == '__main__':
    main()
