"""Fixed-epoch full-source training; no held-out loader or validation metrics."""
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.eth_ucy_final_dataset import ETHUCYFinalDataset
from src.data.eth_ucy_social_final_dataset import ETHUCYSocialFinalDataset
from src.models.social_residual import SocialResidualPredictor
from scripts.eth_ucy_utils import make_model, seed_all, state_hash
from scripts.social_residual_utils import cache_contexts, forward_cached, new_state, restore_new_state, parameter_report
from scripts.final_protocol import ROOT, RESULTS, CHECKPOINTS, dump, read, sha_file, utc_now
from scripts.final_prediction_export import predict_export, save_prediction_artifact


def run_paths(fold, name, seed, smoke=False):
    prefix = 'smoke' if smoke else 'training'
    folder = RESULTS/prefix/f'heldout_{fold}'/name/f'seed_{seed}'
    checkpoint = CHECKPOINTS/('smoke' if smoke else f'heldout_{fold}')
    if smoke:
        checkpoint = checkpoint/f'heldout_{fold}'
    return folder, checkpoint/name/f'seed_{seed}.pt'


def load_final_model(fold, name, seed, protocol_sha, smoke=False):
    kind = name.split('_')[0]
    _, base_path = run_paths(fold, kind, seed, smoke)
    base_ckpt = torch.load(base_path, map_location='cpu', weights_only=True)
    if base_ckpt['protocol_sha256'] != protocol_sha or base_ckpt['seed'] != seed or base_ckpt['fold'] != fold:
        raise ValueError('Wrong final full-source backbone')
    b, _ = make_model(kind, seed)
    b.load_state_dict(base_ckpt['model'])
    b = b.cuda().eval()
    if not name.endswith('_sr'):
        return b
    model = SocialResidualPredictor(b, name, seed).cuda().eval()
    _, path = run_paths(fold, name, seed, smoke)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt['protocol_sha256'] != protocol_sha or ckpt['base_checkpoint_sha256'] != sha_file(base_path):
        raise ValueError('Social checkpoint/base hash mismatch')
    restore_new_state(model, ckpt['new_modules'])
    return model


def train_final(fold, name, seed, protocol, protocol_sha, smoke=False):
    torch.set_num_threads(1)
    social = name.endswith('_sr')
    folder, checkpoint = run_paths(fold, name, seed, smoke)
    report_path = folder/'training_report.json'
    epochs = 3 if smoke else protocol['final_epochs'][fold][name]
    if report_path.exists():
        old = read(report_path)
        if old['protocol_sha256'] != protocol_sha or old['epochs_run'] != epochs or old['checkpoint_sha256'] != sha_file(checkpoint) or old['nan_inf']:
            raise ValueError('Existing final run violates frozen protocol')
        return old
    started = utc_now()
    metadata = {}
    dataset = ETHUCYSocialFinalDataset(fold, 'final_train') if social else ETHUCYFinalDataset(fold, 'final_train')
    assert fold not in dataset.scenes and set(dataset.scenes) == set(protocol['fold_definitions'][fold]['train'])
    if social:
        kind = name.split('_')[0]
        backbone = load_final_model(fold, kind, seed, protocol_sha, smoke)
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone_hash = state_hash(backbone.state_dict())
        _, base_path = run_paths(fold, kind, seed, smoke)
        model = SocialResidualPredictor(backbone, name, seed).cuda()
        seed_all(seed)
        cache_start = time.perf_counter()
        contexts = cache_contexts(backbone, dataset)
        metadata = {'base_checkpoint_path': str(base_path.relative_to(ROOT)), 'base_checkpoint_sha256': sha_file(base_path), 'backbone_state_sha256_before': backbone_hash, 'context_cache_seconds': time.perf_counter()-cache_start}
        model.eval()
        with torch.no_grad():
            initial = forward_cached(model, contexts, slice(0, 128))
            difference = float((initial['future_pred']-contexts['base_prediction'][:128]).abs().max())
        assert difference < 1e-7
        metadata['initial_output_base_max_abs_diff'] = difference
        initialization = {'new_module_initialization_sha256': state_hash(new_state(model)), 'residual_initialization_sha256': state_hash(model.residual.state_dict()), **metadata}
    else:
        model, decoder_hash = make_model(name, seed)
        model = model.cuda()
        initialization = {'model_initialization_sha256': state_hash(model.state_dict()), 'decoder_initialization_sha256': decoder_hash}
        loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=.0001)
    criterion = nn.SmoothL1Loss(beta=1.)
    generator = torch.Generator().manual_seed(seed)
    dump(folder/'initialization_report.json', {'seed': seed, 'model': name, 'protocol_sha256': protocol_sha, **initialization})
    history = []
    max_gradient = 0.
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.
        if social:
            order = torch.randperm(len(dataset), generator=generator).cuda()
            batches = (order[i:i+128] for i in range(0, len(dataset), 128))
        else:
            batches = loader
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            if social:
                pred = forward_cached(model, contexts, batch)['future_pred']
                target = contexts['future_target'][batch]
                count = len(batch)
            else:
                pred = model(batch['obs_input'].cuda())['future_pred']
                target = batch['future_target'].cuda()
                count = len(target)
            loss = criterion(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite final training loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
            max_gradient = max(max_gradient, float(norm))
            optimizer.step()
            total += float(loss)*count
        history.append({'epoch': epoch, 'train_loss': total/len(dataset), 'lr': .001})
        dump(folder/'training_history.json', history)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'{fold} {name} {seed} epoch {epoch}/{epochs}: train_loss={total/len(dataset):.6f}', flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()
    if social:
        assert all(not p.requires_grad and p.grad is None for p in backbone.parameters()) and not backbone.training
        after = state_hash(backbone.state_dict())
        assert after == backbone_hash
        metadata.update({'backbone_state_sha256_after': after, 'backbone_unchanged': True, 'backbone_frozen': True, 'backbone_eval': True})
        state = {'new_modules': new_state(model), 'base_checkpoint_sha256': metadata['base_checkpoint_sha256']}
    else:
        state = {'model': model.state_dict()}
    assert all(torch.isfinite(v).all() for v in (state['new_modules'] if social else state['model']).values())
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state.update({'seed': seed, 'fold': fold, 'model_name': name, 'epoch': epochs, 'protocol_sha256': protocol_sha, 'train_scene_list': list(dataset.scenes), 'train_sample_count': len(dataset), 'heldout_test_evaluated': False, 'smoke': smoke})
    tmp = checkpoint.with_suffix('.tmp.pt')
    torch.save(state, tmp)
    tmp.replace(checkpoint)
    report = {'fold': fold, 'heldout_scene': fold, 'model': name, 'seed': seed, 'train_scene_list': list(dataset.scenes), 'train_sample_count': len(dataset), 'epochs_run': epochs, 'training_epoch': epochs, 'epoch_rule': '3 smoke epochs' if smoke else 'frozen median of prior validation best_epoch', 'started_at_utc': started, 'completed_at_utc': utc_now(), 'protocol_sha256': protocol_sha, 'checkpoint_path': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha_file(checkpoint), 'parameter_count': parameter_report(model), 'maximum_gradient_norm_before_clipping': max_gradient, 'training_seconds': seconds, 'gpu_peak_memory_bytes': peak, 'nan_inf': False, 'heldout_test_evaluated': False, 'test_metrics_accessed_before_freeze': False, 'smoke': smoke, **metadata}
    if smoke:
        model.eval()
        # Source-only export exercises artifact fields; never instantiate a test loader.
        class SourceSubset:
            audit = {k: v[:4] for k, v in dataset.audit.items()}
            arrays = {k: v[:4] for k, v in dataset.arrays.items()}
        batch = {k: v.cuda() for k, v in SourceSubset.arrays.items()}
        output = predict_export(model, batch, social)
        outputs = [{k: v.cpu().numpy() for k, v in output.items()}]
        save_prediction_artifact(folder/'source_prediction_smoke.npz', SourceSubset, outputs, social)
        report['source_prediction_export_passed'] = True
    dump(report_path, report)
    del model, dataset
    if social:
        del contexts, backbone
    torch.cuda.empty_cache()
    return report
