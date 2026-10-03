"""Source-only smoke and 60 fixed-regime runs; no formal test entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from scripts.training_regime_protocol import RESULTS, MODELS, SCENES, SEEDS, dump, read, sha_file, freeze_protocol, verify_protocol, verify_historical_results
from scripts.training_regime_utils import train_diagnostic


def smoke(protocol, psha):
    gate = RESULTS/'smoke/smoke_gate.json'
    if gate.exists():
        q = read(gate)
        assert q['passed'] and q['protocol_sha256'] == psha
        return
    for model in ('ett', 'emt', 'ett_sr', 'emt_sr'):
        q = train_diagnostic('eth', model, 42, protocol, psha, smoke=True)
        assert q['epochs_run'] == 2 and q['fixed_lr'] == .001 and not q['nan_inf'] and not q['heldout_test_accessed']
        if model.endswith('_sr'):
            assert q['backbone_unchanged'] and q['initial_output_base_max_abs_diff'] < 1e-7
    dump(gate, {'passed': True, 'models': list(MODELS), 'protocol_sha256': psha, 'epochs': 2, 'scheduler': None, 'early_stopping': False, 'checkpoint_rule': 'last epoch', 'validation_controls_training': False, 'heldout_test_accessed': False, 'historical_test_already_accessed': True})


def worker(job):
    torch.set_num_threads(1)
    protocol, psha = verify_protocol()
    fold, seed = job
    for model in ('ett', 'ett_sr', 'emt', 'emt_sr'):
        train_diagnostic(fold, model, seed, protocol, psha)
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--mode', default='source_validation')
    args = parser.parse_args()
    if args.mode != 'source_validation':
        raise PermissionError('Diagnostic entry point rejects heldout_test and final training modes')
    torch.set_num_threads(1)
    protocol, psha = freeze_protocol()
    verify_historical_results()
    smoke(protocol, psha)
    if args.smoke_only:
        return
    jobs = [(fold, seed) for fold in SCENES for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=protocol['training_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        errors = []
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            try:
                print('DIAGNOSTIC PAIR COMPLETE', future.result(), flush=True)
            except Exception as error:
                errors.append(repr(error))
                print('DIAGNOSTIC PAIR FAILED', repr(error), flush=True)
        if errors:
            raise RuntimeError(errors)
    verify_protocol()
    verify_historical_results()
    from scripts.summarize_training_regime_diagnostic import main as summarize
    summarize()


if __name__ == '__main__':
    main()
