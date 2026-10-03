"""Per-sample source validation diagnostics and explicitly bounded strata."""
import numpy as np

COUNT_GROUPS = ('0', '1-2', '3-4', '5+')
DISTANCE_GROUPS = ('no-neighbor', '<2m', '2-4m', '>4m')
CORRECTION_GROUPS = ('0-5%', '5-10%', '10-20%', '>20%')


def distribution(values):
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return {k: None for k in ('mean', 'median', 'p90', 'p95', 'max')}
    if not np.isfinite(v).all():
        raise FloatingPointError('Nonfinite diagnostic distribution')
    return {'mean': float(v.mean()), 'median': float(np.median(v)), 'p90': float(np.quantile(v, .9)), 'p95': float(np.quantile(v, .95)), 'max': float(v.max())}


def correction_metrics(residual, base_prediction, epsilon=1e-8):
    residual_norm = np.linalg.norm(np.asarray(residual, dtype=np.float64), axis=-1).mean(1)
    base_norm = np.linalg.norm(np.asarray(base_prediction, dtype=np.float64), axis=-1).mean(1)
    ratio = residual_norm/(base_norm+epsilon)
    summary = {'residual_norm': distribution(residual_norm), 'base_trajectory_norm': distribution(base_norm), 'correction_ratio': distribution(ratio), 'correction_ratio_epsilon': epsilon}
    return summary, {'residual_norm': residual_norm, 'base_trajectory_norm': base_norm, 'correction_ratio': ratio}


def neighbor_groups(count, distance):
    count, distance = np.asarray(count), np.asarray(distance)
    return {'neighbor_count': {'0': count == 0, '1-2': (count >= 1)&(count <= 2), '3-4': (count >= 3)&(count <= 4), '5+': count >= 5}, 'neighbor_distance': {'no-neighbor': count == 0, '<2m': (count > 0)&(distance < 2), '2-4m': (count > 0)&(distance >= 2)&(distance <= 4), '>4m': (count > 0)&(distance > 4)}}


def correction_groups(ratio):
    r = np.asarray(ratio)
    return {'0-5%': (r >= 0)&(r <= .05), '5-10%': (r > .05)&(r <= .1), '10-20%': (r > .1)&(r <= .2), '>20%': r > .2}


def strata(samples, masks, social=False):
    result = {}
    for label, selected in masks.items():
        n = int(selected.sum())
        q = {'sample_count': n, 'ADE': float(samples['sample_ADE'][selected].mean()) if n else None, 'FDE': float(samples['sample_FDE'][selected].mean()) if n else None}
        if social:
            for m in ('ADE', 'FDE'):
                base = samples[f'base_{m}'][selected]
                sr = samples[f'sample_{m}'][selected]
                q[f'base_{m}'] = float(base.mean()) if n else None
                q[f'delta_{m}'] = float((sr-base).mean()) if n else None
                q[f'gain_{m}'] = float((base-sr).mean()) if n else None
            q['improvement_rate_ADE'] = float((samples['sample_ADE'][selected] < samples['base_ADE'][selected]).mean()) if n else None
            q['base_better_or_equal_rate_ADE'] = 1-q['improvement_rate_ADE'] if n else None
        result[label] = q
    return result


def report_samples(samples, social=False):
    masks = neighbor_groups(samples['neighbor_count'], samples['mean_neighbor_distance'])
    report = {'validation_ADE': float(samples['sample_ADE'].mean()), 'validation_FDE': float(samples['sample_FDE'].mean()), 'validation_sample_count': len(samples['sample_ADE']), **{key: strata(samples, value, social) for key, value in masks.items()}}
    if social:
        report.update({'base_validation_ADE': float(samples['base_ADE'].mean()), 'base_validation_FDE': float(samples['base_FDE'].mean()), 'correction_magnitude': strata(samples, correction_groups(samples['correction_ratio']), True), 'correction_statistics': {'residual_norm': distribution(samples['residual_norm']), 'base_trajectory_norm': distribution(samples['base_trajectory_norm']), 'correction_ratio': distribution(samples['correction_ratio'])}})
    return report
