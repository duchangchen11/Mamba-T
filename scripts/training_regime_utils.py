"""Fixed training loop with passive validation, last-epoch state and source guard."""
import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.eth_ucy_training_regime_dataset import TrainingRegimeSourceDataset, require_source_mode
from src.models.social_residual import SocialResidualPredictor
from scripts.eth_ucy_utils import make_model, seed_all, state_hash
from scripts.social_residual_utils import cache_contexts, forward_cached, new_state, parameter_report
from scripts.training_regime_protocol import ROOT, RESULTS, CHECKPOINTS, dump, read, sha_file, utc_now
from scripts.training_regime_metrics import correction_metrics, report_samples


def run_fixed_epochs(model, epochs, batches, train_prediction, validation, callback=None):
    """Validation has no optimizer/control input and cannot change RNG or parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.0001)
    criterion = nn.SmoothL1Loss(beta=1.)
    history = []
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    for epoch in range(1, epochs+1):
        model.train()
        total = count = 0
        norms = []
        for batch in batches(epoch):
            optimizer.zero_grad(set_to_none=True)
            pred, target = train_prediction(batch)
            loss = criterion(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite source diagnostic loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            total += float(loss)*len(target)
            count += len(target)
        before = state_hash(model.state_dict())
        # Preserve CPU/CUDA dropout and loader RNG even if diagnostics consume randomness.
        with torch.random.fork_rng(devices=devices):
            model.eval()
            with torch.no_grad():
                recorded = validation()
        if state_hash(model.state_dict()) != before:
            raise RuntimeError('Passive validation mutated model state')
        model.train()
        row = {'epoch': epoch, 'train_loss': total/count, 'gradient_norm': max(norms), 'mean_gradient_norm': float(np.mean(norms)), 'lr': optimizer.param_groups[0]['lr'], **recorded}
        assert row['lr'] == .001
        history.append(row)
        if callback:
            callback(history)
    return history


def paths(fold, name, seed, smoke=False):
    suffix = Path('smoke')/f'heldout_{fold}'/name/f'seed_{seed}' if smoke else Path(f'heldout_{fold}')/name/f'seed_{seed}'
    return RESULTS/suffix, CHECKPOINTS/suffix.with_suffix('.pt')


def diagnostic_best(history):
    best = min(history, key=lambda q: q['validation_ADE'])
    best_fde = min(history, key=lambda q: q['validation_FDE'])
    final = history[-1]
    return {'diagnostic_best_validation_epoch': best['epoch'], 'diagnostic_best_validation_ADE': best['validation_ADE'], 'diagnostic_best_validation_FDE': best['validation_FDE'], 'diagnostic_minimum_FDE_epoch': best_fde['epoch'], 'diagnostic_minimum_FDE': best_fde['validation_FDE'], 'final_best_gap_ADE': final['validation_ADE']-best['validation_ADE'], 'final_best_gap_FDE': final['validation_FDE']-best['validation_FDE'], 'final_minimum_FDE_gap': final['validation_FDE']-best_fde['validation_FDE'], 'best_used_for_checkpoint_selection': False}


def save_last_checkpoint(path, model, epoch, metadata, social=False):
    state = {'new_modules': new_state(model)} if social else {'model': model.state_dict()}
    tensors = next(iter(state.values()))
    assert all(torch.isfinite(v).all() for v in tensors.values())
    state.update({'epoch': epoch, 'final_model_state_sha256': state_hash(model.state_dict()), **metadata})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(state, tmp)
    tmp.replace(path)
    return sha_file(path)


@torch.no_grad()
def evaluate_source(model, dataset, contexts=None):
    require_source_mode(dataset.mode)
    model.eval()
    social = contexts is not None
    predictions, bases, residuals = [], [], []
    for i in range(0, len(dataset), 128):
        if social:
            out = forward_cached(model, contexts, slice(i, i+128))
            bases.append(out['base_prediction'].cpu().numpy())
            residuals.append(out['residual'].cpu().numpy())
        else:
            out = model(dataset.arrays['target_history'][i:i+128].cuda())
        predictions.append(out['future_pred'].cpu().numpy())
    prediction = np.concatenate(predictions)
    last = dataset.arrays['last_obs_pos'].numpy()
    future = dataset.arrays['future_abs'].numpy()
    distance = np.linalg.norm((prediction+last[:, None]).astype(np.float64)-future.astype(np.float64), axis=-1)
    mask = dataset.arrays['neighbor_mask'].numpy()
    count = mask.sum(1)
    relation = dataset.arrays['neighbor_relation'].numpy()
    mean_distance = np.divide((relation[..., 2]*mask).sum(1), count, out=np.zeros(len(count), dtype=np.float64), where=count > 0)
    samples = {'scene_id': dataset.audit['scene_id'], 'target_ped_id': dataset.audit['target_ped_id'], 'frame_ids': dataset.audit['frame_ids'], 'sample_ADE': distance.mean(1), 'sample_FDE': distance[:, -1], 'neighbor_count': count, 'mean_neighbor_distance': mean_distance}
    if social:
        base, residual = np.concatenate(bases), np.concatenate(residuals)
        base_distance = np.linalg.norm((base+last[:, None]).astype(np.float64)-future.astype(np.float64), axis=-1)
        _, norms = correction_metrics(residual, base)
        samples.update({'base_ADE': base_distance.mean(1), 'base_FDE': base_distance[:, -1], **norms})
    if not all(np.isfinite(v).all() for v in samples.values() if v.dtype.kind in 'fbiu'):
        raise FloatingPointError('Nonfinite source diagnostic samples')
    return report_samples(samples, social), samples


def train_diagnostic(fold, name, seed, protocol, protocol_sha, smoke=False):
    torch.set_num_threads(1)
    social = name.endswith('_sr')
    folder, checkpoint = paths(fold, name, seed, smoke)
    epochs = 2 if smoke else protocol['final_epochs'][fold][name]
    metrics_path = folder/'metrics_validation.json'
    if metrics_path.exists():
        q = read(metrics_path)
        assert q['protocol_sha256'] == protocol_sha and q['epochs_run'] == epochs and q['checkpoint_sha256'] == sha_file(checkpoint) and not q['nan_inf']
        return q
    train = TrainingRegimeSourceDataset(fold, 'train', social)
    val = TrainingRegimeSourceDataset(fold, 'val', social)
    assert not train.tracks & val.tracks and fold not in train.scenes | val.scenes
    metadata = {'fold': fold, 'model': name, 'seed': seed, 'protocol_sha256': protocol_sha, 'train_scene_list': sorted(train.scenes), 'train_sample_count': len(train), 'validation_sample_count': len(val), 'train_pedestrian_count': len(train.tracks), 'validation_pedestrian_count': len(val.tracks), 'train_validation_pedestrian_overlap': 0, 'train_cross_split_neighbor_violations': 0, 'heldout_test_accessed': False, 'historical_test_already_accessed': True, 'smoke': smoke}
    if social:
        kind = name.split('_')[0]
        base_folder, base_path = paths(fold, kind, seed, smoke)
        base_report = read(base_folder/'metrics_validation.json')
        base_checkpoint = torch.load(base_path, map_location='cpu', weights_only=True)
        assert base_checkpoint['fold'] == fold and base_checkpoint['seed'] == seed and base_checkpoint['protocol_sha256'] == protocol_sha
        assert base_checkpoint['epoch'] == base_report['epochs_run'] and base_report['checkpoint_sha256'] == sha_file(base_path)
        backbone, _ = make_model(kind, seed)
        backbone.load_state_dict(base_checkpoint['model'])
        backbone = backbone.cuda().eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        metadata.update({'base_checkpoint_path': str(base_path.relative_to(ROOT)), 'base_checkpoint_sha256': sha_file(base_path), 'backbone_state_sha256_before': state_hash(backbone.state_dict())})
        model = SocialResidualPredictor(backbone, name, seed).cuda()
        seed_all(seed)
        cache_start = time.perf_counter()
        tr, va = cache_contexts(backbone, train), cache_contexts(backbone, val)
        metadata['context_cache_seconds'] = time.perf_counter()-cache_start
        model.eval()
        with torch.no_grad():
            initial = forward_cached(model, va, slice(0, 128))
            difference = float((initial['future_pred']-va['base_prediction'][:128]).abs().max())
        assert difference < 1e-7
        metadata['initial_output_base_max_abs_diff'] = difference
        generator = torch.Generator().manual_seed(seed)
        def batches(epoch):
            order = torch.randperm(len(train), generator=generator).cuda()
            return (order[i:i+128] for i in range(0, len(train), 128))
        def train_prediction(index):
            return forward_cached(model, tr, index)['future_pred'], tr['future_target'][index]
        initialization = {'new_modules_initialization_sha256': state_hash(new_state(model)), **metadata}
    else:
        model, decoder_hash = make_model(name, seed)
        model = model.cuda()
        loader = DataLoader(train, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
        batches = lambda epoch: loader
        train_prediction = lambda batch: (model(batch['obs_input'].cuda())['future_pred'], batch['future_target'].cuda())
        initialization = {'model_initialization_sha256': state_hash(model.state_dict()), 'decoder_initialization_sha256': decoder_hash, **metadata}
    dump(folder/'initialization_report.json', initialization)
    last = {}
    def validation():
        q, samples = evaluate_source(model, val, va if social else None)
        last.update({'report': q, 'samples': samples})
        row = {'validation_ADE': q['validation_ADE'], 'validation_FDE': q['validation_FDE']}
        if social:
            stats = q['correction_statistics']
            row.update({'base_validation_ADE': q['base_validation_ADE'], 'base_validation_FDE': q['base_validation_FDE'], 'mean_residual_norm': stats['residual_norm']['mean'], 'median_residual_norm': stats['residual_norm']['median'], 'p90_residual_norm': stats['residual_norm']['p90'], 'p95_residual_norm': stats['residual_norm']['p95'], 'mean_base_trajectory_norm': stats['base_trajectory_norm']['mean'], 'residual_base_ratio': stats['correction_ratio']['mean'], 'correction_ratio_statistics': stats['correction_ratio']})
        return row
    def callback(history):
        dump(folder/'validation_history.json', history)
        row = history[-1]
        if row['epoch'] == 1 or row['epoch'] % 10 == 0 or row['epoch'] == epochs:
            print(f'{fold} {name}-fixed {seed} epoch {row["epoch"]}/{epochs}: loss={row["train_loss"]:.6f} source_val_ADE={row["validation_ADE"]:.6f} source_val_FDE={row["validation_FDE"]:.6f}', flush=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    history = run_fixed_epochs(model, epochs, batches, train_prediction, validation, callback)
    torch.cuda.synchronize()
    metadata.update({'training_seconds': time.perf_counter()-start, 'gpu_peak_memory_bytes': torch.cuda.max_memory_allocated()})
    if social:
        assert not backbone.training and all(p.grad is None and not p.requires_grad for p in backbone.parameters())
        after = state_hash(backbone.state_dict())
        assert after == metadata['backbone_state_sha256_before']
        metadata.update({'backbone_state_sha256_after': after, 'backbone_unchanged': True, 'backbone_frozen': True, 'backbone_eval': True})
        assert all(abs(last['report'][f'base_validation_{m}']-base_report[f'validation_{m}']) < 1e-7 for m in ('ADE', 'FDE'))
    checkpoint_sha = save_last_checkpoint(checkpoint, model, epochs, metadata, social)
    sample_path = folder/'validation_samples.npz'
    np.savez_compressed(sample_path, **last['samples'])
    q = {**metadata, **last['report'], **diagnostic_best(history), 'epochs_run': epochs, 'final_epoch': epochs, 'final_epoch_metrics': history[-1], 'checkpoint_path': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': checkpoint_sha, 'final_model_state_sha256': state_hash(model.state_dict()), 'validation_samples_sha256': sha_file(sample_path), 'maximum_gradient_norm_before_clipping': max(h['gradient_norm'] for h in history), 'parameter_count': parameter_report(model), 'nan_inf': False, 'fixed_lr': .001, 'scheduler': None, 'early_stopping': False, 'checkpoint_rule': 'last epoch', 'validation_controls_training': False}
    dump(metrics_path, q)
    curve = RESULTS/'curves'/('smoke' if smoke else f'heldout_{fold}')/name/f'seed_{seed}'
    dump(curve.with_suffix('.json'), history)
    curve.parent.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in row.items() if isinstance(v, (str, int, float))} for row in history]
    with curve.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    del model, train, val
    if social:
        del tr, va, backbone
    torch.cuda.empty_cache()
    return q
