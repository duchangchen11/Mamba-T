"""Diagnostic attention and prediction artifacts without changing model output."""
import numpy as np
import torch


@torch.no_grad()
def attention_weights(model, target, neighbors, mask, relation):
    if model.training or model.backbone.training:
        raise ValueError('Attention export requires eval mode')
    _, h = model.backbone.encode(target)
    valid = mask.bool()
    weights = torch.zeros(valid.shape, device=h.device, dtype=h.dtype)
    has = valid.any(1)
    if has.any():
        encoded = torch.zeros((*valid.shape, 128), device=h.device, dtype=h.dtype)
        _, n = model.backbone.encode(neighbors[valid])
        encoded[valid] = n
        m = valid[has]
        kv = encoded[has].masked_fill(~m[..., None], 0) + model.social.relation_embedding(relation[has].masked_fill(~m[..., None], 0))
        # Diagnostic call only: its attention output never replaces standard SDPA output.
        _, w = model.social.attention(h[has, None], kv, kv, key_padding_mask=~m, need_weights=True, average_attn_weights=True)
        weights[has] = w[:, 0]
    return weights


@torch.no_grad()
def predict_export(model, batch, social, return_attention_weights=True):
    if model.training:
        raise ValueError('Prediction export requires eval mode')
    if social:
        out = model(batch['target_history'], batch['neighbor_history'], batch['neighbor_mask'], batch['neighbor_relation'])
        if return_attention_weights:
            out['attention_weights'] = attention_weights(model, batch['target_history'], batch['neighbor_history'], batch['neighbor_mask'], batch['neighbor_relation'])
    else:
        out = model(batch['obs_input'])
    if not all(torch.isfinite(v).all() for v in out.values()):
        raise FloatingPointError('Nonfinite prediction export')
    return out


def displacement_metrics(prediction_abs, future_gt):
    distance = np.linalg.norm(prediction_abs.astype(np.float64)-future_gt.astype(np.float64), axis=-1)
    if not np.isfinite(distance).all():
        raise FloatingPointError('Nonfinite artifact metrics')
    return distance.mean(1), distance[:, -1]


def save_prediction_artifact(path, dataset, outputs, social):
    prediction_rel = np.concatenate([o['future_pred'] for o in outputs])
    last = dataset.arrays['last_obs_pos'].numpy()
    future = dataset.arrays['future_abs'].numpy()
    prediction_abs = prediction_rel + last[:, None]
    ade, fde = displacement_metrics(prediction_abs, future)
    artifact = {k: dataset.audit[k] for k in ('scene_id', 'ped_id', 'frame_ids', 'obs_abs')}
    artifact.update({'future_gt': future, 'prediction_abs': prediction_abs, 'prediction_rel': prediction_rel, 'last_obs_pos': last, 'sample_ADE': ade, 'sample_FDE': fde})
    if social:
        relation = dataset.arrays['neighbor_relation'].numpy()
        mask = dataset.arrays['neighbor_mask'].numpy()
        neighbor_abs = dataset.arrays['neighbor_history'].numpy()[..., :2] + (last[:, None]+relation[..., :2])[:, :, None]
        neighbor_abs[~mask] = 0
        artifact.update({'neighbor_abs': neighbor_abs, 'neighbor_mask': mask, 'neighbor_relation': relation, 'neighbor_ids': dataset.audit['neighbor_ids'], 'neighbor_frame_ids': dataset.audit['neighbor_frame_ids'], 'attention_weights': np.concatenate([o['attention_weights'] for o in outputs]), 'social_residual': np.concatenate([o['residual'] for o in outputs]), 'base_prediction': np.concatenate([o['base_prediction'] for o in outputs])})
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic artifact, retaining raw float32 predictions and float64 metric reductions.
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **artifact)
    tmp.replace(path)
    return {'ADE': float(ade.mean()), 'FDE': float(fde.mean()), 'sample_count': len(ade)}
