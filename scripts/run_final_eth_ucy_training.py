"""Smoke then all 60 formal source-only runs; freeze completion before test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from scripts.final_protocol import ROOT, RESULTS, CHECKPOINTS, SCENES, MODELS, SEEDS, dump, read, sha_file, utc_now, git_info, verify_protocol, verify_history
from scripts.final_training_utils import train_final


def smoke(protocol, psha):
    gate = RESULTS/'smoke/smoke_gate.json'
    if gate.exists():
        q = read(gate)
        assert q['passed'] and q['protocol_sha256'] == psha
        return
    for name in ('ett', 'emt', 'ett_sr', 'emt_sr'):
        q = train_final('eth', name, 42, protocol, psha, smoke=True)
        assert not q['nan_inf'] and not q['heldout_test_evaluated'] and q['source_prediction_export_passed']
    dump(gate, {'passed': True, 'protocol_sha256': psha, 'models': list(MODELS), 'epochs': 3, 'source_only_prediction_export': True, 'heldout_test_evaluated': False, 'test_metrics_accessed_before_freeze': False, 'completed_at_utc': utc_now()})


def worker(job):
    fold, seed = job
    torch.set_num_threads(1)
    protocol, psha = verify_protocol()
    for name in ('ett', 'ett_sr', 'emt', 'emt_sr'):
        train_final(fold, name, seed, protocol, psha)
    return job


def freeze_training_completion(protocol, psha):
    log = read(RESULTS/'data_audit/test_access_log.json')
    if log['evaluations']:
        raise ValueError('Cannot re-freeze training after test access')
    runs = []
    for fold in SCENES:
        for name in MODELS:
            for seed in SEEDS:
                folder = RESULTS/'training'/f'heldout_{fold}'/name/f'seed_{seed}'
                q = read(folder/'training_report.json')
                history = read(folder/'training_history.json')
                path = ROOT/q['checkpoint_path']
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                assert q['epochs_run'] == len(history) == protocol['final_epochs'][fold][name] == checkpoint['epoch']
                assert q['protocol_sha256'] == psha == checkpoint['protocol_sha256']
                assert q['checkpoint_sha256'] == sha_file(path)
                assert not q['nan_inf'] and not q['heldout_test_evaluated'] and not q['smoke']
                assert q['train_scene_list'] == protocol['fold_definitions'][fold]['train']
                assert q['train_sample_count'] == read(RESULTS/'data_audit/train_scene_counts.json')[fold]['total']
                assert all(torch.isfinite(v).all() for v in checkpoint.get('model', checkpoint.get('new_modules')).values())
                if name.endswith('_sr'):
                    assert q['backbone_unchanged'] and q['backbone_frozen'] and q['backbone_eval'] and q['initial_output_base_max_abs_diff'] < 1e-7
                    assert q['backbone_state_sha256_before'] == q['backbone_state_sha256_after']
                runs.append({'fold': fold, 'model': name, 'seed': seed, 'training_epoch': q['epochs_run'], 'checkpoint_path': q['checkpoint_path'], 'checkpoint_sha256': q['checkpoint_sha256'], 'protocol_sha256': psha, 'completed_at_utc': q['completed_at_utc'], 'training_history_sha256': sha_file(folder/'training_history.json'), 'training_report_sha256': sha_file(folder/'training_report.json')})
    assert len(runs) == 60
    verify_protocol()
    historical = verify_history()
    complete = {'completed_runs': 60, 'expected_runs': 60, 'completed_at_utc': utc_now(), **git_info(), 'protocol_sha256': psha, 'test_metrics_accessed_before_freeze': False, 'test_access_log_entries_at_freeze': 0, 'historical_files_verified': historical, 'runs': runs}
    dump(RESULTS/'data_audit/checkpoint_inventory.json', complete)
    dump(RESULTS/'FINAL_TRAINING_COMPLETE.json', complete)
    print('FINAL TRAINING COMPLETE: 60/60; held-out evaluation still unopened', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol, psha = verify_protocol()
    verify_history()
    smoke(protocol, psha)
    if args.smoke_only:
        return
    if (RESULTS/'FINAL_TRAINING_COMPLETE.json').exists():
        from scripts.final_protocol import verify_training_complete
        verify_training_complete()
        print('Existing 60/60 training freeze verified')
        return
    jobs = [(f, s) for f in SCENES for s in SEEDS]
    with ProcessPoolExecutor(max_workers=protocol['training_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        errors = []
        for future in as_completed(futures):
            try:
                print('FINAL PAIR COMPLETE', future.result(), flush=True)
            except Exception as error:
                errors.append(repr(error))
                print('FINAL PAIR FAILED', repr(error), flush=True)
        if errors:
            raise RuntimeError(errors)
    freeze_training_completion(protocol, psha)


if __name__ == '__main__':
    main()
