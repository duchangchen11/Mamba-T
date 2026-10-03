"""Source-only fixed-epoch density calibration diagnostic (15 runs)."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn

from scripts.density_calibration_protocol import (
    CHECKPOINTS, DIAGNOSTIC, DIAGNOSTIC_CKPTS, RESULTS, SCENES, SEEDS,
    freeze_protocol, guard_path, read_json, require_source_mode, sha_file,
    verify_history,
)
from scripts.eth_ucy_utils import make_model, seed_all, state_hash
from scripts.social_residual_utils import cache_contexts, new_state
from scripts.training_regime_utils import run_fixed_epochs
from src.data.eth_ucy_training_regime_dataset import TrainingRegimeSourceDataset
from src.models.density_aware_social import DensityAwareSocialResidualPredictor, NEIGHBOR_LIMIT, density_scale_from_density


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def density_lut(model, device):
    with torch.no_grad():
        d = torch.arange(NEIGHBOR_LIMIT+1, device=device, dtype=torch.float32).reshape(-1, 1)/NEIGHBOR_LIMIT
        return density_scale_from_density(d, model.density_calibration).flatten().cpu().numpy()


@torch.no_grad()
def evaluate_density(model, data, dataset):
    require_source_mode(dataset.mode)
    model.eval()
    preds, scales, residuals = [], [], []
    for i in range(0, len(dataset), 128):
        sl = slice(i, i+128)
        out = model.forward_context(
            data['target_context'][sl], data['neighbor_context'][sl], data['neighbor_relation'][sl],
            data['neighbor_mask'][sl], data['base_prediction'][sl],
        )
        preds.append(out['future_pred'].cpu().numpy())
        scales.append(out['density_scale'][:, 0].cpu().numpy())
        residuals.append(out['residual'].cpu().numpy())
    prediction, scale, residual = map(np.concatenate, (preds, scales, residuals))
    last, future = dataset.arrays['last_obs_pos'].numpy(), dataset.arrays['future_abs'].numpy()
    d = np.linalg.norm((prediction+last[:, None]).astype(np.float64)-future.astype(np.float64), axis=-1)
    base_prediction = data['base_prediction'].cpu().numpy()
    bd = np.linalg.norm((base_prediction+last[:, None]).astype(np.float64)-future.astype(np.float64), axis=-1)
    residual_norm = np.linalg.norm(residual.astype(np.float64), axis=-1).mean(1)
    base_norm = np.linalg.norm(base_prediction.astype(np.float64), axis=-1).mean(1)
    count = dataset.arrays['neighbor_mask'].sum(1).numpy().astype(np.int64)
    if not np.isfinite(np.concatenate((d.ravel(), scale, residual_norm))).all():
        raise FloatingPointError('Nonfinite source validation output')
    if not ((scale > .5).all() and (scale < 1.5).all()):
        raise AssertionError('Density scale escaped its strict bounds')
    samples = {
        'scene_id': dataset.audit['scene_id'], 'target_ped_id': dataset.audit['target_ped_id'],
        'frame_ids': dataset.audit['frame_ids'], 'neighbor_count': count,
        'sample_ADE': d.mean(1), 'sample_FDE': d[:, -1],
        'base_ADE': bd.mean(1), 'base_FDE': bd[:, -1],
        'residual_norm': residual_norm, 'base_trajectory_norm': base_norm,
        'correction_ratio': residual_norm/(base_norm+1e-8), 'density_scale': scale,
    }
    count_scale = {}
    for n in range(NEIGHBOR_LIMIT+1):
        chosen = count == n
        count_scale[str(n)] = {'sample_count': int(chosen.sum()), 'mean_scale': float(scale[chosen].mean()) if chosen.any() else None}
    return {
        'validation_ADE': float(d.mean()), 'validation_FDE': float(d[:, -1].mean()),
        'base_validation_ADE': float(bd.mean()), 'base_validation_FDE': float(bd[:, -1].mean()),
        'residual_norm': float(residual_norm.mean()),
        'correction_ratio': float(samples['correction_ratio'].mean()),
        'density_scale': {'mean': float(scale.mean()), 'std': float(scale.std()), 'min': float(scale.min()), 'max': float(scale.max()), 'median': float(np.median(scale)), 'p10': float(np.quantile(scale, .1)), 'p90': float(np.quantile(scale, .9)), 'p95': float(np.quantile(scale, .95))},
        'observed_scale_by_neighbor_count': count_scale,
    }, samples


def run_one(fold, seed, protocol, protocol_sha, smoke=False):
    require_source_mode('train')
    torch.set_num_threads(1)
    result_name = 'smoke' if smoke else f'heldout_{fold}'
    folder = RESULTS/result_name/'emt_density_sr'/f'seed_{seed}'
    ckpt_path = CHECKPOINTS/result_name/'emt_density_sr'/f'seed_{seed}.pt'
    epochs = 2 if smoke else protocol['final_epochs'][fold]['emt_sr']
    metric_path = folder/'metrics_validation.json'
    if metric_path.exists():
        q = read_json(metric_path)
        if q.get('protocol_sha256') != protocol_sha or q.get('epochs_run') != epochs or q.get('checkpoint_sha256') != sha_file(ckpt_path):
            raise ValueError('Existing density run does not match frozen protocol')
        return q

    train = TrainingRegimeSourceDataset(fold, 'train', True)
    val = TrainingRegimeSourceDataset(fold, 'val', True)
    assert not train.tracks & val.tracks and fold not in train.scenes | val.scenes
    baseline_folder = DIAGNOSTIC/f'heldout_{fold}'/'emt_sr'/f'seed_{seed}'
    base_folder = DIAGNOSTIC/f'heldout_{fold}'/'emt'/f'seed_{seed}'
    baseline_metrics = read_json(baseline_folder/'metrics_validation.json')
    base_metrics = read_json(base_folder/'metrics_validation.json')
    baseline_init = read_json(baseline_folder/'initialization_report.json')
    if baseline_metrics['heldout_test_accessed'] or base_metrics['heldout_test_accessed']:
        raise PermissionError('A source baseline artifact reports held-out test access')
    if baseline_metrics['epochs_run'] != protocol['final_epochs'][fold]['emt_sr'] or baseline_metrics['checkpoint_rule'] != 'last epoch' or baseline_metrics['scheduler'] is not None or baseline_metrics['early_stopping']:
        raise ValueError('Frozen EMT-SR baseline does not match the fixed training regime')
    baseline_samples_path = guard_path(baseline_folder/'validation_samples.npz')
    with np.load(baseline_samples_path, allow_pickle=False) as archive:
        baseline_samples = {k: archive[k] for k in archive.files}
    base_ckpt_path = guard_path(DIAGNOSTIC_CKPTS/f'heldout_{fold}'/'emt'/f'seed_{seed}.pt')
    base_ckpt_sha = sha_file(base_ckpt_path)
    if base_metrics['checkpoint_sha256'] != base_ckpt_sha or baseline_metrics['base_checkpoint_sha256'] != base_ckpt_sha:
        raise ValueError('Frozen EMT checkpoint SHA mismatch')
    base_ckpt = torch.load(base_ckpt_path, map_location='cpu', weights_only=True)
    backbone, _ = make_model('emt', seed)
    backbone.load_state_dict(base_ckpt['model'])
    backbone = backbone.cuda().eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model = DensityAwareSocialResidualPredictor(backbone, seed).cuda()
    seed_all(seed)
    shared_init_hash = state_hash(new_state(model.shared))
    if shared_init_hash != baseline_init['new_modules_initialization_sha256']:
        raise AssertionError('Shared social/residual initialization differs from frozen EMT-SR')
    tr, va = cache_contexts(backbone, train), cache_contexts(backbone, val)
    model.eval()
    with torch.no_grad():
        initial = model.forward_context(va['target_context'][:128], va['neighbor_context'][:128], va['neighbor_relation'][:128], va['neighbor_mask'][:128], va['base_prediction'][:128])
        initial_diff = float((initial['future_pred']-va['base_prediction'][:128]).abs().max())
        assert torch.equal(initial['density_scale'], torch.ones_like(initial['density_scale']))
    if initial_diff >= 1e-7:
        raise AssertionError(f'Zero-initialized density model changed initial prediction: {initial_diff}')
    _, identity_samples = evaluate_density(model, va, val)
    np.testing.assert_allclose(identity_samples['sample_ADE'], baseline_samples['base_ADE'], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(identity_samples['sample_FDE'], baseline_samples['base_FDE'], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(identity_samples['base_ADE'], baseline_samples['base_ADE'], rtol=1e-6, atol=1e-7)
    for key, current, old in (
        ('scene_id', val.audit['scene_id'], baseline_samples['scene_id']),
        ('target_ped_id', val.audit['target_ped_id'], baseline_samples['target_ped_id']),
        ('frame_ids', val.audit['frame_ids'], baseline_samples['frame_ids']),
    ):
        np.testing.assert_array_equal(current, old)

    generator = torch.Generator().manual_seed(seed)
    def batches(epoch):
        order = torch.randperm(len(train), generator=generator).cuda()
        return (order[i:i+128] for i in range(0, len(train), 128))
    def train_prediction(index):
        output = model.forward_context(tr['target_context'][index], tr['neighbor_context'][index], tr['neighbor_relation'][index], tr['neighbor_mask'][index], tr['base_prediction'][index])
        return output['future_pred'], tr['future_target'][index]
    last = {}
    def validation():
        report, samples = evaluate_density(model, va, val)
        last.update(report=report, samples=samples)
        return {k: report[k] for k in ('validation_ADE', 'validation_FDE', 'residual_norm', 'correction_ratio')} | {
            'density_scale_mean': report['density_scale']['mean'], 'density_scale_std': report['density_scale']['std'],
            'density_scale_min': report['density_scale']['min'], 'density_scale_max': report['density_scale']['max'],
            'density_scale_by_neighbor_count': density_lut(model, next(model.parameters()).device).tolist(),
        }
    history = []
    max_grad = 0.
    def callback(rows):
        nonlocal max_grad
        max_grad = max(max_grad, rows[-1]['gradient_norm'])
        dump(folder/'validation_history.json', rows)
        if rows[-1]['epoch'] == 1 or rows[-1]['epoch'] == epochs:
            print(f'{fold} emt_density_sr seed={seed} epoch={rows[-1]["epoch"]}/{epochs}: ADE={rows[-1]["validation_ADE"]:.6f} scale={rows[-1]["density_scale_mean"]:.4f}', flush=True)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
    history = run_fixed_epochs(model, epochs, batches, train_prediction, validation, callback)
    torch.cuda.synchronize(); train_seconds = time.perf_counter()-start
    max_grad = max(row['gradient_norm'] for row in history)
    model.eval()
    backbone_hash = state_hash(backbone.state_dict())
    if model.training or backbone.training or any(p.requires_grad or p.grad is not None for p in backbone.parameters()) or backbone_hash != base_metrics['final_model_state_sha256']:
        raise AssertionError('Frozen EMT backbone state/mode changed')
    # Fixed-regime diagnostics save the last epoch. Validation never selects weights.
    final = history[-1]
    report, samples = last['report'], last['samples']
    if report['density_scale']['min'] <= .5 or report['density_scale']['max'] >= 1.5:
        raise AssertionError('Final scale violates strict bound')
    shared_state = new_state(model.shared)
    density_state = model.density_calibration.state_dict()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_path.with_suffix('.tmp.pt')
    torch.save({
        'shared_modules': shared_state, 'density_calibration': density_state,
        'epoch': epochs, 'protocol_sha256': protocol_sha,
        'base_checkpoint_sha256': base_ckpt_sha,
        'shared_initialization_sha256': shared_init_hash,
        'final_model_state_sha256': state_hash(model.state_dict()),
        'heldout_test_accessed': False, 'historical_test_already_accessed': True,
    }, tmp)
    tmp.replace(ckpt_path)
    np.savez_compressed(folder/'validation_samples.npz', **samples)
    (folder/'scale_by_neighbor_count.csv').parent.mkdir(parents=True, exist_ok=True)
    with (folder/'scale_by_neighbor_count.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['neighbor_count', 'density', 'scale', 'source_validation_samples'])
        writer.writeheader()
        lut = density_lut(model, next(model.parameters()).device)
        counts = np.bincount(samples['neighbor_count'], minlength=9)
        for n, value in enumerate(lut):
            writer.writerow({'neighbor_count': n, 'density': n/8, 'scale': value, 'source_validation_samples': int(counts[n])})
    metrics = {
        'fold': fold, 'seed': seed, 'model': 'emt_density_sr', 'model_chinese_name': 'Mamba密度自适应社会交互残差模型',
        'epochs_run': epochs, 'final_epoch': epochs, 'final_epoch_metrics': final,
        'validation_ADE': report['validation_ADE'], 'validation_FDE': report['validation_FDE'],
        'base_validation_ADE': report['base_validation_ADE'], 'base_validation_FDE': report['base_validation_FDE'],
        'residual_norm': report['residual_norm'], 'correction_ratio': report['correction_ratio'],
        'density_scale': report['density_scale'], 'observed_scale_by_neighbor_count': report['observed_scale_by_neighbor_count'],
        'scale_lut_by_neighbor_count': {str(n): float(v) for n, v in enumerate(density_lut(model, next(model.parameters()).device))},
        'initial_output_base_max_abs_diff': initial_diff,
        'shared_initialization_sha256': shared_init_hash,
        'expected_emt_sr_shared_initialization_sha256': baseline_init['new_modules_initialization_sha256'],
        'base_checkpoint_path': str(base_ckpt_path.relative_to(Path.cwd())), 'base_checkpoint_sha256': base_ckpt_sha,
        'checkpoint_path': str(ckpt_path.relative_to(Path.cwd())), 'checkpoint_sha256': sha_file(ckpt_path),
        'validation_samples_sha256': sha_file(folder/'validation_samples.npz'),
        'maximum_gradient_norm': max_grad, 'training_seconds': train_seconds,
        'gpu_peak_memory_bytes': torch.cuda.max_memory_allocated(),
        'backbone_state_sha256_before': base_metrics['final_model_state_sha256'],
        'backbone_state_sha256_after': backbone_hash, 'backbone_frozen': True, 'backbone_eval': True,
        'scheduler': None, 'early_stopping': False, 'checkpoint_rule': 'last fixed epoch',
        'optimizer': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'batch_size': 128,
        'loss': 'SmoothL1Loss(beta=1)', 'gradient_clip': 5.,
        'parameter_count': {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad), 'total': sum(p.numel() for p in model.parameters())},
        'nan_inf': False, 'heldout_test_accessed': False, 'historical_test_already_accessed': True,
        'trajectory_training': False, 'smoke': smoke, 'protocol_sha256': protocol_sha,
        'validation_sample_count': len(val), 'train_sample_count': len(train),
    }
    dump(folder/'metrics_validation.json', metrics)
    dump(folder/'validation_history.json', history)
    return metrics


def main(mode='source_validation', smoke_only=False):
    require_source_mode(mode)
    torch.set_num_threads(1)
    protocol = freeze_protocol()
    protocol_sha = sha_file(RESULTS/'protocol_frozen.json')
    rows = []
    if smoke_only:
        rows.append(run_one('eth', 42, protocol, protocol_sha, smoke=True))
        assert rows[0]['epochs_run'] == 2 and rows[0]['initial_output_base_max_abs_diff'] < 1e-7 and not rows[0]['nan_inf']
        dump(RESULTS/'data_audit/smoke_gate.json', {
            'passed': True, 'fold': 'eth', 'seed': 42, 'epochs': 2,
            'protocol_sha256': protocol_sha, 'heldout_test_accessed': False,
            'historical_test_already_accessed': True,
        })
        return
    verify_history()
    gate = read_json(RESULTS/'data_audit/smoke_gate.json')
    if not gate['passed'] or gate['protocol_sha256'] != protocol_sha:
        raise RuntimeError('Two-epoch source smoke gate missing or stale')
    for fold in SCENES:
        for seed in (42, 123, 2024):
            row = run_one(fold, seed, protocol, protocol_sha)
            rows.append(row)
            print(f'DENSITY RUN COMPLETE {fold} seed={seed}', flush=True)
    if len(rows) != 15:
        raise AssertionError(f'Expected 15 source runs; got {len(rows)}')
    from scripts.summarize_density_aware_social_calibration import summarize
    summarize(protocol, rows)
    verify_history()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='source_validation')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    main(args.mode, args.smoke_only)
