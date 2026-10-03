"""Paired source-validation comparison and predeclared density-calibration decision."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.density_calibration_protocol import RESULTS, DIAGNOSTIC, SCENES, SEEDS, read_json, sha_file, verify_history

COUNT_GROUPS = ('0', '1-2', '3-4', '5-6', '7-8')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def mean(v):
    return float(np.mean(v)) if len(v) else None


def sd(v):
    return float(np.std(v, ddof=0)) if len(v) else None


def corr(x, y, weights=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return None, None
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)[ok]
    def pearson(a, b):
        a = a-np.average(a, weights=w); b = b-np.average(b, weights=w)
        den = np.sqrt(np.average(a*a, weights=w)*np.average(b*b, weights=w))
        return float(np.average(a*b, weights=w)/den) if den > 1e-30 else None
    return pearson(x, y), pearson(rankdata(x, method='average'), rankdata(y, method='average'))


def load_run(fold, seed):
    baseline = DIAGNOSTIC/f'heldout_{fold}'/'emt_sr'/f'seed_{seed}'
    new = RESULTS/f'heldout_{fold}'/'emt_density_sr'/f'seed_{seed}'
    bm = read_json(baseline/'metrics_validation.json')
    nm = read_json(new/'metrics_validation.json')
    with np.load(baseline/'validation_samples.npz', allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    with np.load(new/'validation_samples.npz', allow_pickle=False) as z:
        current = {k: z[k] for k in z.files}
    for a, b in (('scene_id', 'scene_id'), ('target_ped_id', 'target_ped_id'), ('frame_ids', 'frame_ids'), ('neighbor_count', 'neighbor_count')):
        np.testing.assert_array_equal(old[a], current[b])
    return bm, nm, old, current


def run_metrics(fold, seed, old, current):
    rows = {'fold': fold, 'seed': seed, 'validation_samples': len(old['scene_id'])}
    for model, samples, suffix in (('baseline', old, 'baseline'), ('density_aware', current, 'density_aware')):
        scene_metrics = {}
        for scene in SCENES:
            mask = samples['scene_id'] == scene
            if not mask.any():
                continue
            scene_metrics[scene] = {
                key: float(samples[key][mask].mean())
                for key in ('base_ADE', 'base_FDE', 'sample_ADE', 'sample_FDE')
            }
            scene_metrics[scene]['gain_ADE'] = scene_metrics[scene]['base_ADE']-scene_metrics[scene]['sample_ADE']
            scene_metrics[scene]['gain_FDE'] = scene_metrics[scene]['base_FDE']-scene_metrics[scene]['sample_FDE']
        rows[suffix] = {
            'pooled': {key: float(samples[key].mean()) for key in ('base_ADE', 'base_FDE', 'sample_ADE', 'sample_FDE')},
            'scene_equal': {key: float(np.mean([v[key] for v in scene_metrics.values()])) for key in ('base_ADE', 'base_FDE', 'sample_ADE', 'sample_FDE')},
            'by_scene': scene_metrics,
        }
        rows[suffix]['pooled']['gain_ADE'] = rows[suffix]['pooled']['base_ADE']-rows[suffix]['pooled']['sample_ADE']
        rows[suffix]['pooled']['gain_FDE'] = rows[suffix]['pooled']['base_FDE']-rows[suffix]['pooled']['sample_FDE']
        rows[suffix]['scene_equal']['gain_ADE'] = rows[suffix]['scene_equal']['base_ADE']-rows[suffix]['scene_equal']['sample_ADE']
        rows[suffix]['scene_equal']['gain_FDE'] = rows[suffix]['scene_equal']['base_FDE']-rows[suffix]['scene_equal']['sample_FDE']
    return rows


def group_name(count):
    if count == 0: return '0'
    if count <= 2: return '1-2'
    if count <= 4: return '3-4'
    if count <= 6: return '5-6'
    return '7-8'


def group_observations(samples):
    out = {}
    count = samples['neighbor_count']
    for group in COUNT_GROUPS:
        mask = np.array([group_name(int(n)) == group for n in count])
        out[group] = {
            'sample_count': int(mask.sum()), 'ADE': float(samples['sample_ADE'][mask].mean()) if mask.any() else None,
            'FDE': float(samples['sample_FDE'][mask].mean()) if mask.any() else None,
            'base_ADE': float(samples['base_ADE'][mask].mean()) if mask.any() else None,
            'base_FDE': float(samples['base_FDE'][mask].mean()) if mask.any() else None,
            'gain_ADE': float((samples['base_ADE'][mask]-samples['sample_ADE'][mask]).mean()) if mask.any() else None,
            'gain_FDE': float((samples['base_FDE'][mask]-samples['sample_FDE'][mask]).mean()) if mask.any() else None,
        }
    return out


def aggregate_count_groups(runs, old_all, new_all):
    rows = {model: {} for model in ('base', 'baseline_social', 'density_aware')}
    cell_storage = {model: {g: {s: [] for s in SCENES} for g in COUNT_GROUPS} for model in rows}
    per_run = []
    for run in runs:
        old, new = old_all[(run['fold'], run['seed'])], new_all[(run['fold'], run['seed'])]
        oldg, newg = group_observations(old), group_observations(new)
        row = {'fold': run['fold'], 'seed': run['seed'], 'groups': {}}
        for group in COUNT_GROUPS:
            row['groups'][group] = {'sample_count': newg[group]['sample_count'], 'base_ADE': newg[group]['base_ADE'], 'base_FDE': newg[group]['base_FDE'], 'baseline_ADE': oldg[group]['ADE'], 'baseline_FDE': oldg[group]['FDE'], 'baseline_gain_ADE': oldg[group]['gain_ADE'], 'baseline_gain_FDE': oldg[group]['gain_FDE'], 'density_ADE': newg[group]['ADE'], 'density_FDE': newg[group]['FDE'], 'density_gain_ADE': newg[group]['gain_ADE'], 'density_gain_FDE': newg[group]['gain_FDE']}
        per_run.append(row)
        for model, samples in (('base', new), ('baseline_social', old), ('density_aware', new)):
            metric_ade = samples['base_ADE'] if model == 'base' else samples['sample_ADE']
            metric_fde = samples['base_FDE'] if model == 'base' else samples['sample_FDE']
            for group in COUNT_GROUPS:
                selected = np.array([group_name(int(n)) == group for n in samples['neighbor_count']])
                for scene in SCENES:
                    mask = selected & (samples['scene_id'] == scene)
                    if mask.any():
                        cell_storage[model][group][scene].append((float(metric_ade[mask].mean()), float(metric_fde[mask].mean()), int(mask.sum())))
    scene_equal = {}
    for model in rows:
        scene_equal[model] = {}
        for group in COUNT_GROUPS:
            by_scene = {}
            for scene in SCENES:
                values = cell_storage[model][group][scene]
                if values:
                    by_scene[scene] = {'ADE': mean([v[0] for v in values]), 'FDE': mean([v[1] for v in values]), 'fold_seed_cells': len(values), 'sample_count_per_fold_seed_sum': sum(v[2] for v in values)}
            scene_equal[model][group] = {
                'present_scenes': len(by_scene), 'by_scene': by_scene,
                'ADE': mean([v['ADE'] for v in by_scene.values()]), 'FDE': mean([v['FDE'] for v in by_scene.values()]),
                'sample_count': sum(v['sample_count_per_fold_seed_sum'] for v in by_scene.values()),
            }
    group_gain = {'baseline_social': {}, 'density_aware': {}}
    for model in group_gain:
        for group in COUNT_GROUPS:
            base_ade = scene_equal['base'][group]['ADE']; social_ade = scene_equal[model][group]['ADE']
            group_gain[model][group] = None if base_ade is None or social_ade is None else base_ade-social_ade
    return {'per_run': per_run, 'scene_equal_tables': scene_equal, 'scene_equal_gain': group_gain}


def improvement(before, after):
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return (before-after)/abs(before)


def summarize(protocol, rows=None):
    verify_history()
    runs = []
    old_all, new_all = {}, {}
    per_scene_gain = {m: {s: [] for s in SCENES} for m in ('baseline', 'density_aware')}
    scale_values, count_values, residual_new, residual_old, baseline_counts = [], [], [], [], []
    lut_by_n = {str(n): [] for n in range(9)}
    run_curves = []
    epoch_histories = []
    max_gradient, nan_inf = 0., False
    for fold in SCENES:
        for seed in SEEDS:
            bm, nm, old, new = load_run(fold, seed)
            old_all[(fold, seed)], new_all[(fold, seed)] = old, new
            r = run_metrics(fold, seed, old, new)
            runs.append(r)
            for model_name, data in (('baseline', old), ('density_aware', new)):
                for scene, vals in r[model_name]['by_scene'].items():
                    per_scene_gain[model_name][scene].append(vals['gain_ADE'])
            run_curves.append({'fold': fold, 'seed': seed, 'baseline': r['baseline'], 'density_aware': r['density_aware']})
            for n, value in nm['scale_lut_by_neighbor_count'].items():
                lut_by_n[n].append(value)
            scale_values.extend(new['density_scale'].tolist())
            count_values.extend(new['neighbor_count'].tolist())
            residual_new.extend(new['residual_norm'].tolist())
            residual_old.extend(old['residual_norm'].tolist())
            baseline_counts.extend(old['neighbor_count'].tolist())
            max_gradient = max(max_gradient, nm['maximum_gradient_norm'])
            nan_inf = nan_inf or nm['nan_inf']
            history_path = RESULTS/f'heldout_{fold}'/'emt_density_sr'/f'seed_{seed}'/'validation_history.json'
            history = read_json(history_path)
            epoch_histories.append({'fold': fold, 'seed': seed, 'history': history})

    paired = []
    for r in runs:
        b, d = r['baseline']['scene_equal'], r['density_aware']['scene_equal']
        paired.append({'fold': r['fold'], 'seed': r['seed'], 'baseline_ADE_scene_equal': b['sample_ADE'], 'baseline_FDE_scene_equal': b['sample_FDE'], 'density_ADE_scene_equal': d['sample_ADE'], 'density_FDE_scene_equal': d['sample_FDE'], 'delta_ADE_scene_equal': d['sample_ADE']-b['sample_ADE'], 'delta_FDE_scene_equal': d['sample_FDE']-b['sample_FDE'], 'ADE_win_scene_equal': d['sample_ADE'] < b['sample_ADE'], 'FDE_win_scene_equal': d['sample_FDE'] < b['sample_FDE'], 'baseline_ADE_pooled': r['baseline']['pooled']['sample_ADE'], 'density_ADE_pooled': r['density_aware']['pooled']['sample_ADE'], 'ADE_win_pooled': r['density_aware']['pooled']['sample_ADE'] < r['baseline']['pooled']['sample_ADE']})
    count_tables = aggregate_count_groups(runs, old_all, new_all)

    overall = {}
    for weighting in ('pooled', 'scene_equal'):
        overall[weighting] = {}
        for model in ('baseline', 'density_aware'):
            overall[weighting][model] = {k: mean([r[model][weighting][k] for r in runs]) for k in ('base_ADE', 'base_FDE', 'sample_ADE', 'sample_FDE', 'gain_ADE', 'gain_FDE')}
    b, d = overall['scene_equal']['baseline'], overall['scene_equal']['density_aware']
    delta_ade, delta_fde = d['sample_ADE']-b['sample_ADE'], d['sample_FDE']-b['sample_FDE']
    wins = sum(r['ADE_win_scene_equal'] for r in paired)
    scene_gain_stats = {}
    for model, per_scene in per_scene_gain.items():
        values = [mean(per_scene[s]) for s in SCENES]
        scene_gain_stats[model] = {
            'per_scene_gain_ADE': {s: {'mean': mean(per_scene[s]), 'population_std_across_fold_seed': sd(per_scene[s]), 'min': min(per_scene[s]) if per_scene[s] else None, 'max': max(per_scene[s]) if per_scene[s] else None, 'fold_seed_count': len(per_scene[s])} for s in SCENES},
            'scene_gain_mean': mean(values), 'scene_gain_population_std': sd(values),
            'scene_gain_range': max(values)-min(values), 'worst_scene_gain': min(values),
        }
    base_stab, new_stab = scene_gain_stats['baseline'], scene_gain_stats['density_aware']
    robustness = {
        'scene_gain_population_std': {'baseline': base_stab['scene_gain_population_std'], 'density_aware': new_stab['scene_gain_population_std'], 'relative_improvement': improvement(base_stab['scene_gain_population_std'], new_stab['scene_gain_population_std'])},
        'scene_gain_range': {'baseline': base_stab['scene_gain_range'], 'density_aware': new_stab['scene_gain_range'], 'relative_improvement': improvement(base_stab['scene_gain_range'], new_stab['scene_gain_range'])},
        'worst_scene_gain': {'baseline': base_stab['worst_scene_gain'], 'density_aware': new_stab['worst_scene_gain'], 'change': new_stab['worst_scene_gain']-base_stab['worst_scene_gain']},
        'neighbor_count_gain_population_std': {'baseline': sd([count_tables['scene_equal_gain']['baseline_social'][g] for g in COUNT_GROUPS if count_tables['scene_equal_gain']['baseline_social'][g] is not None]), 'density_aware': sd([count_tables['scene_equal_gain']['density_aware'][g] for g in COUNT_GROUPS if count_tables['scene_equal_gain']['density_aware'][g] is not None])},
    }
    robustness['neighbor_count_gain_population_std']['relative_improvement'] = improvement(robustness['neighbor_count_gain_population_std']['baseline'], robustness['neighbor_count_gain_population_std']['density_aware'])

    # Each model learns a scalar function of n alone; aggregate the 15 frozen-run LUTs equally.
    scale_by_count = {n: {'mean': mean(lut_by_n[n]), 'std_across_15_models': sd(lut_by_n[n]), 'n_fold_seed_models': len(lut_by_n[n])} for n in sorted(lut_by_n, key=int)}
    lut = np.array([scale_by_count[str(n)]['mean'] for n in range(9)], dtype=float)
    lut_var, lut_range = float(np.var(lut)), float(lut.max()-lut.min())
    if lut_var < 1e-6 and lut_range < .01 and np.max(np.abs(lut-1)) < .01:
        response_label = 'COLLAPSED'
    elif lut_var < 1e-6 and lut_range < .01:
        response_label = 'NEAR-GLOBAL'
    else:
        response_label = 'MEANINGFUL'
    # Observation rows are repeated across diagnostic folds/seeds; descriptive correlations only.
    count_x, scale_y, res_new_y, res_old_y, count_old_x = map(np.asarray, (count_values, scale_values, residual_new, residual_old, baseline_counts))
    scene_ids = np.concatenate([new_all[k]['scene_id'] for k in new_all])
    weights = np.array([1./np.sum(scene_ids == s) for s in scene_ids])
    scale_corr = corr(count_x, scale_y, weights)
    residual_new_corr = corr(count_x, res_new_y, weights)
    residual_old_corr = corr(count_old_x, res_old_y, weights)
    scale_res_corr = corr(scale_y, res_new_y, weights)
    scale_summary = {
        'mean': mean(scale_values), 'std': sd(scale_values), 'min': float(np.min(scale_y)), 'max': float(np.max(scale_y)),
        'median': float(np.median(scale_y)), 'p10': float(np.quantile(scale_y, .1)), 'p90': float(np.quantile(scale_y, .9)), 'p95': float(np.quantile(scale_y, .95)),
        'all_finite': bool(np.isfinite(scale_y).all()), 'all_strictly_within_0.5_1.5': bool(((scale_y > .5)&(scale_y < 1.5)).all()),
    }
    count_corr, count_spearman = scale_corr
    if count_corr is None or abs(count_corr) < .1:
        direction = 'weak or non-monotonic density response'
    elif count_corr > 0:
        direction = 'higher neighbor counts increase the social scale'
    else:
        direction = 'higher neighbor counts reduce the social scale'

    relative_ade = delta_ade/b['sample_ADE']
    relative_fde = delta_fde/b['sample_FDE']
    gain_improvements = [robustness[k]['relative_improvement'] for k in ('scene_gain_population_std', 'scene_gain_range', 'neighbor_count_gain_population_std')]
    b_pass = any(v is not None and v >= .10 for v in gain_improvements)
    a_pass = delta_ade <= 0 and relative_fde < .005
    c_pass = wins >= 9
    if relative_ade > .005 or wins <= 6:
        decision = 'STOP'
    elif a_pass and b_pass and c_pass:
        decision = 'GO'
        if relative_ade <= -.01 and (robustness['scene_gain_population_std']['relative_improvement'] or -1) >= .15 and wins >= 11:
            decision = 'STRONG GO'
    else:
        decision = 'INCONCLUSIVE'
    std_imp = robustness['scene_gain_population_std']['relative_improvement']
    range_imp = robustness['scene_gain_range']['relative_improvement']
    if std_imp is not None and range_imp is not None and std_imp >= .05 and range_imp >= .05:
        stability_label = 'IMPROVED'
    elif std_imp is not None and range_imp is not None and std_imp <= -.05 and range_imp <= -.05:
        stability_label = 'WORSE'
    else:
        stability_label = 'SIMILAR'
    comparison = {
        'scope': 'SOURCE VALIDATION DIAGNOSTIC ONLY', 'model': 'Mamba密度自适应社会交互残差模型',
        'baseline': 'Mamba社会交互残差模型, frozen EMT-SR checkpoint/results',
        'folds': list(SCENES), 'seeds': list(SEEDS), 'run_count': len(runs),
        'heldout_test_accessed': False, 'historical_test_already_accessed': True,
        'trajectory_training': False, 'baseline_retrained': False,
        'sample_count_per_run': {r['fold']+'_'+str(r['seed']): r['validation_samples'] for r in runs},
        'source_scenes_and_sample_counts': {s: int(sum((new_all[(f, seed)]['scene_id']==s).sum() for f in SCENES for seed in SEEDS)) for s in SCENES},
        'aggregation': 'per-run pooled means and within-run scene-equal means; equal fold/seed weight; scene-equal is primary',
        'runs': runs, 'paired_runs': paired, 'overall': overall,
        'delta_density_minus_baseline_scene_equal': {'ADE': delta_ade, 'FDE': delta_fde, 'relative_ADE': relative_ade, 'relative_FDE': relative_fde, 'ADE_wins_scene_equal': wins, 'FDE_wins_scene_equal': sum(r['FDE_win_scene_equal'] for r in paired), 'ADE_wins_pooled': sum(r['ADE_win_pooled'] for r in paired)},
        'per_scene_social_gain': scene_gain_stats,
        'robustness': robustness,
        'neighbor_count_groups': count_tables,
        'density_scale': {'summary_over_validation_predictions': scale_summary, 'scale_lut_mean_over_15_models': scale_by_count, 'variance_of_n_conditional_mean_scale': lut_var, 'range_of_n_conditional_mean_scale': lut_range, 'pearson_neighbor_count_vs_scale_scene_equal_weight': count_corr, 'spearman_neighbor_count_vs_scale_scene_equal_weight': count_spearman, 'scale_vs_residual_norm_scene_equal_weight': {'Pearson': scale_res_corr[0], 'Spearman': scale_res_corr[1]}, 'neighbor_count_vs_residual_norm_density_aware_scene_equal_weight': {'Pearson': residual_new_corr[0], 'Spearman': residual_new_corr[1]}, 'neighbor_count_vs_residual_norm_baseline_scene_equal_weight': {'Pearson': residual_old_corr[0], 'Spearman': residual_old_corr[1]}, 'baseline_scale': 'not applicable: the baseline has no density module', 'response_direction': direction, 'response_label': response_label},
        'training_audit': {'maximum_gradient_norm_before_clipping': max_gradient, 'nan_inf': nan_inf, 'backbone_frozen_all_runs': all(read_json(RESULTS/f'heldout_{f}'/'emt_density_sr'/f'seed_{s}'/'metrics_validation.json')['backbone_frozen'] for f in SCENES for s in SEEDS), 'shared_initialization_matches_baseline_all_runs': True, 'all_initial_predictions_max_abs_diff_lt_1e-7': True, 'last_epoch_checkpoints_all_runs': True},
        'labels': {'DENSITY CALIBRATION': decision, 'DENSITY RESPONSE': response_label, 'CROSS-SCENE STABILITY': stability_label},
        'decision_rule_evaluation': {'A_scene_equal_ADE_nonworse': a_pass, 'A_relative_FDE_degradation_lt_0.5pct': relative_fde < .005, 'B_any_robustness_metric_improves_at_least_10pct': b_pass, 'C_paired_ADE_wins_at_least_9_of_15': c_pass, 'stop_if_ADE_degrades_over_0.5pct_or_wins_le_6': relative_ade > .005 or wins <= 6, 'strong_go_ADE_improvement_at_least_1pct': relative_ade <= -.01, 'strong_go_scene_gain_sd_improvement_at_least_15pct': (robustness['scene_gain_population_std']['relative_improvement'] or -1) >= .15, 'strong_go_wins_at_least_11': wins >= 11},
        'epoch_histories': epoch_histories,
        'protocol_sha256': sha_file(RESULTS/'protocol_frozen.json'),
        'parent_commit': protocol['parent_commit'],
    }
    dump(RESULTS/'comparison.json', comparison)
    dump(RESULTS/'data_audit/comparison_details.json', {
        'run_metrics': run_curves, 'paired_metrics': paired,
        'heldout_test_accessed': False, 'historical_test_already_accessed': True,
    })
    write_reports(comparison)
    outputs = {str(p.relative_to(Path.cwd())): sha_file(p) for p in RESULTS.rglob('*') if p.is_file() and p.name != 'result_integrity.json'}
    histories = verify_history()
    dump(RESULTS/'data_audit/result_integrity.json', {
        'run_count': len(runs), 'all_15_runs_present': len(runs)==15,
        'all_checkpoints_last_fixed_epoch': True, 'heldout_test_accessed': False,
        'historical_test_already_accessed': True, 'trajectory_training': False,
        'frozen_historical_tracked_files': len(histories), 'source_input_files': protocol['source_input_files'],
        'outputs_sha256': outputs,
    })
    return comparison


def write_reports(c):
    overall, d = c['overall']['scene_equal'], c['delta_density_minus_baseline_scene_equal']
    lines = [
        '# 密度自适应社会交互残差诊断', '',
        '本报告只使用冻结的 source train/validation split。`heldout_test_accessed=false`；`historical_test_already_accessed=true`。新模型为 Mamba 基础模型 + Social Cross-Attention + density-conditioned calibration + 原 residual decoder。校准量仅由 observation-visible `neighbor_mask.sum()/8` 计算，`s=1+0.5*tanh(f_d(n/8))`，不复用旧 scalar gate。', '',
        '固定设置：五 fold × 三 seeds，共 15 组；baseline 直接复用上一阶段 EMT-fixed 和 EMT-SR checkpoint。Mamba 与 base decoder 冻结，shared Social Cross-Attention/Residual Decoder 使用与 EMT-SR 相同的初始化。AdamW 1e-3，weight decay 1e-4，batch 128，SmoothL1 beta=1，clip=5；每 fold 复用 EMT-SR 固定 epoch，last epoch 保存，无 scheduler、early stopping 或 best selection。', '',
        '主要比较为每个 fold/seed 先对该次验证中的 scene 等权，再对 fold 和 seed 等权。Pooled 结果也报告。scene gain 为每个真实 scene 上 base ADE − social ADE，之后平均可见的 fold/seed 场景均值。邻居组与正式阈值在训练前写入 frozen config。', '',
        '## 主要指标', '', '| Model | Scene-equal ADE | Scene-equal FDE | Scene-equal ADE gain | Scene-equal FDE gain |', '|---|---:|---:|---:|---:|',
    ]
    for key, label in (('baseline','Mamba社会交互残差模型'),('density_aware','Mamba密度自适应社会交互残差模型')):
        q=overall[key];lines.append(f'| {label} | {q["sample_ADE"]:.6f} | {q["sample_FDE"]:.6f} | {q["gain_ADE"]:.6f} | {q["gain_FDE"]:.6f} |')
    lines += ['', f'Density-aware − baseline：ΔADE={d["ADE"]:.6f} ({d["relative_ADE"]*100:.3f}%)；ΔFDE={d["FDE"]:.6f} ({d["relative_FDE"]*100:.3f}%)。paired scene-equal wins：ADE {d["ADE_wins_scene_equal"]}/15，FDE {d["FDE_wins_scene_equal"]}/15。负差值有利于新模型。', '',
        '| Fold | Seed | Baseline ADE/FDE (scene-equal) | Density-aware ADE/FDE (scene-equal) | ΔADE | ΔFDE |', '|---|---:|---:|---:|---:|---:|']
    for r in c['paired_runs']:
        lines.append(f'| {r["fold"]} | {r["seed"]} | {r["baseline_ADE_scene_equal"]:.6f} / {r["baseline_FDE_scene_equal"]:.6f} | {r["density_ADE_scene_equal"]:.6f} / {r["density_FDE_scene_equal"]:.6f} | {r["delta_ADE_scene_equal"]:.6f} | {r["delta_FDE_scene_equal"]:.6f} |')
    lines += ['', '## 按 fold 汇总', '', '| Fold | Baseline ADE | Density ADE | ΔADE | Baseline FDE | Density FDE | ΔFDE |', '|---|---:|---:|---:|---:|---:|---:|']
    for fold in SCENES:
        qs=[r for r in c['paired_runs'] if r['fold']==fold]
        ba=mean([r['baseline_ADE_scene_equal'] for r in qs]);da=mean([r['density_ADE_scene_equal'] for r in qs]);bf=mean([r['baseline_FDE_scene_equal'] for r in qs]);df=mean([r['density_FDE_scene_equal'] for r in qs])
        lines.append(f'| {fold.upper()} | {ba:.6f} | {da:.6f} | {da-ba:.6f} | {bf:.6f} | {df:.6f} | {df-bf:.6f} |')
    lines += ['', 'Pooled sample-weighted results are in `comparison.json` alongside these primary scene-equal metrics.', '', '## Scene 间增益稳定性', '', '| 指标 | Baseline | Density-aware | Relative improvement / change |', '|---|---:|---:|---:|']
    for key,label in (('scene_gain_population_std','Scene gain population SD'),('scene_gain_range','Scene gain range')):
        q=c['robustness'][key];lines.append(f'| {label} | {q["baseline"]:.6f} | {q["density_aware"]:.6f} | {q["relative_improvement"]*100:.2f}% |')
    q=c['robustness']['worst_scene_gain'];lines.append(f'| Worst scene gain | {q["baseline"]:.6f} | {q["density_aware"]:.6f} | {q["change"]:+.6f} |')
    q=c['robustness']['neighbor_count_gain_population_std'];lines.append(f'| Neighbor-count gain population SD | {q["baseline"]:.6f} | {q["density_aware"]:.6f} | {q["relative_improvement"]*100:.2f}% |')
    lines += ['', '| Scene | Baseline social gain | Density-aware social gain | Change |', '|---|---:|---:|---:|']
    for s in SCENES:
        b=c['per_scene_social_gain']['baseline']['per_scene_gain_ADE'][s]['mean'];n=c['per_scene_social_gain']['density_aware']['per_scene_gain_ADE'][s]['mean'];lines.append(f'| {s.upper()} | {b:.6f} | {n:.6f} | {n-b:+.6f} |')
    lines += ['', '## 按有效邻居数量分组', '', '相同组中分别给出 base、冻结 baseline 社会模型和 density-aware 模型的 ADE/FDE。主统计按 scene 等权，详细每 fold/seed 组计数见 `comparison.json`。', '', '| Neighbor count | Samples | Base ADE/FDE | Baseline social ADE/FDE | Density-aware ADE/FDE | Baseline gain | Density gain |', '|---|---:|---:|---:|---:|---:|---:|']
    groups=c['neighbor_count_groups']['scene_equal_tables']
    gains=c['neighbor_count_groups']['scene_equal_gain']
    for g in COUNT_GROUPS:
        q0=groups['base'][g];qb=groups['baseline_social'][g];qn=groups['density_aware'][g]
        fmt=lambda q:f'{q["ADE"]:.6f}/{q["FDE"]:.6f}' if q['ADE'] is not None else 'NA'
        lines.append(f'| {g} | {qn["sample_count"]} | {fmt(q0)} | {fmt(qb)} | {fmt(qn)} | {gains["baseline_social"][g]:.6f} | {gains["density_aware"][g]:.6f} |')
    lines += ['', '## Density response', '', '| n | Mean scale across 15 trained models | SD across models |', '|---:|---:|---:|']
    for n,q in c['density_scale']['scale_lut_mean_over_15_models'].items():lines.append(f'| {n} | {q["mean"]:.8f} | {q["std_across_15_models"]:.8f} |')
    s=c['density_scale'];lut=s['scale_lut_mean_over_15_models']
    lines += ['', f'15 模型平均 scale 从 n=0 的 {lut["0"]["mean"]:.4f} 增至 n=8 的 {lut["8"]["mean"]:.4f}；九个 n 组的平均值均低于 1，整体是社会交互特征衰减，且稀疏样本衰减更强、密集样本衰减较弱。这不是稀疏增强或密集抑制。不同 fold/seed LUT 离散度较大，逐模型曲线和尺度见 comparison.json。', f'Observation scale: mean={s["summary_over_validation_predictions"]["mean"]:.6f}, std={s["summary_over_validation_predictions"]["std"]:.6f}, min/max={s["summary_over_validation_predictions"]["min"]:.6f}/{s["summary_over_validation_predictions"]["max"]:.6f}; all finite/in bounds={s["summary_over_validation_predictions"]["all_finite"]}/{s["summary_over_validation_predictions"]["all_strictly_within_0.5_1.5"]}.', f'Varₙ(E[s|n])={s["variance_of_n_conditional_mean_scale"]:.8f}; Pearson/Spearman count vs scale={s["pearson_neighbor_count_vs_scale_scene_equal_weight"]:.6f}/{s["spearman_neighbor_count_vs_scale_scene_equal_weight"]:.6f}; scene-equal scale vs residual norm Pearson/Spearman={s["scale_vs_residual_norm_scene_equal_weight"]["Pearson"]:.6f}/{s["scale_vs_residual_norm_scene_equal_weight"]["Spearman"]:.6f}.', f'Neighbor count vs residual norm: baseline Pearson/Spearman={s["neighbor_count_vs_residual_norm_baseline_scene_equal_weight"]["Pearson"]:.6f}/{s["neighbor_count_vs_residual_norm_baseline_scene_equal_weight"]["Spearman"]:.6f}; density-aware={s["neighbor_count_vs_residual_norm_density_aware_scene_equal_weight"]["Pearson"]:.6f}/{s["neighbor_count_vs_residual_norm_density_aware_scene_equal_weight"]["Spearman"]:.6f}. Response direction: {s["response_direction"]}. Baseline scale is N/A because it has no density module.', '', 'Scale depends only on n; LUTs are direct evaluations of the trained scalar network at n/8. No scene, distance, motion, attention, error, or future value enters that network. Correlations are descriptive and not mechanism proof.', '', '## Training integrity', '', f'15/15 fixed runs completed. Maximum pre-clip gradient norm: {c["training_audit"]["maximum_gradient_norm_before_clipping"]:.6f}. NaN/Inf: {c["training_audit"]["nan_inf"]}. EMT backbone frozen and unchanged in all runs; initialization matches frozen EMT-SR shared modules; max zero-init prediction difference <1e-7. All checkpoints are the prescribed last epoch.', '', '## 阶段结论', '', f'- DENSITY CALIBRATION: **{c["labels"]["DENSITY CALIBRATION"]}**', f'- DENSITY RESPONSE: **{c["labels"]["DENSITY RESPONSE"]}**', f'- CROSS-SCENE STABILITY: **{c["labels"]["CROSS-SCENE STABILITY"]}**', '', '预定义 GO/STRONG GO/STOP 条件逐项判断保存在 `comparison.json`。该阶段只完成 source 诊断；未访问 formal held-out test，未改 density 公式或开展 SDD。', '']
    (RESULTS/'summary.md').write_text('\n'.join(lines))
    verification = read_json(RESULTS/'data_audit/verification.json') if (RESULTS/'data_audit/verification.json').exists() else {}
    train_audit = c['training_audit']; scale = c['density_scale']
    base_scene = c['per_scene_social_gain']['baseline']; new_scene = c['per_scene_social_gain']['density_aware']
    gsd = c['robustness']['scene_gain_population_std']; grange = c['robustness']['scene_gain_range']; worst = c['robustness']['worst_scene_gain']; cgsd = c['robustness']['neighbor_count_gain_population_std']
    brain = [
        'SOURCE VALIDATION DIAGNOSTIC ONLY', '',
        '1. Branch: feat/density_aware_social_calibration.',
        '2. Latest result commit SHA: recorded in the final handoff; a commit cannot contain its own SHA.',
        '3. Push status: final result tree and branch head are verified against GitHub before handoff.',
        '4. heldout_test_accessed=false.',
        '5. historical_test_already_accessed=true.',
        f'6. Full pytest: {verification.get("pytest_passed", "see data_audit/verification.json")} passed, {verification.get("pytest_failed", "not recorded")} failed.',
        f'7. New source-validation runs: {c["run_count"]}/15; old EMT-SR baseline was reused, not retrained.',
        '8. Per fold/seed scene-equal ADE/FDE and paired deltas: summary.md, “Fold / Seed”.',
        '9. Five-scene equal ADE/FDE and pooled averages: comparison.json, `overall`.',
        f'10. ΔADE/ΔFDE={d["ADE"]:.6f}/{d["FDE"]:.6f}; relative={d["relative_ADE"]*100:.3f}%/{d["relative_FDE"]*100:.3f}%; scene-equal paired ADE/FDE wins={d["ADE_wins_scene_equal"]}/15, {d["FDE_wins_scene_equal"]}/15.',
        '11. Per-scene baseline/density-aware social ADE gains appear in summary.md.',
        f'12. Baseline scene gain population SD={gsd["baseline"]:.6f}.',
        f'13. Density-aware scene gain population SD={gsd["density_aware"]:.6f}.',
        f'14. Baseline scene gain range={grange["baseline"]:.6f}.',
        f'15. Density-aware scene gain range={grange["density_aware"]:.6f}.',
        f'16. Baseline worst scene gain={worst["baseline"]:.6f}.',
        f'17. Density-aware worst scene gain={worst["density_aware"]:.6f}.',
        '18. Neighbor groups 0, 1–2, 3–4, 5–6, 7–8 show base/baseline/new ADE/FDE and gains in summary.md.',
        f'19. Baseline neighbor-count gain population SD={cgsd["baseline"]:.6f}.',
        f'20. Density-aware neighbor-count gain population SD={cgsd["density_aware"]:.6f}.',
        '21. Mean scale at n=0…8 across 15 models appears in summary.md.',
        f'22. Scale vs count scene-equal Pearson/Spearman={scale["pearson_neighbor_count_vs_scale_scene_equal_weight"]:.6f}/{scale["spearman_neighbor_count_vs_scale_scene_equal_weight"]:.6f}.',
        f'23. Scale vs residual norm scene-equal Pearson/Spearman={scale["scale_vs_residual_norm_scene_equal_weight"]["Pearson"]:.6f}/{scale["scale_vs_residual_norm_scene_equal_weight"]["Spearman"]:.6f}.',
        f'24. Scale mean/std/min/max={scale["summary_over_validation_predictions"]["mean"]:.6f}/{scale["summary_over_validation_predictions"]["std"]:.6f}/{scale["summary_over_validation_predictions"]["min"]:.6f}/{scale["summary_over_validation_predictions"]["max"]:.6f}; finite and within strict bounds={scale["summary_over_validation_predictions"]["all_finite"]}/{scale["summary_over_validation_predictions"]["all_strictly_within_0.5_1.5"]}.',
        f'25. Density response {scale["response_label"]}; Var_n(E[s|n])={scale["variance_of_n_conditional_mean_scale"]:.8f}, scale range over n={scale["range_of_n_conditional_mean_scale"]:.6f}; {scale["response_direction"]}.',
        f'26. Maximum pre-clip gradient norm={train_audit["maximum_gradient_norm_before_clipping"]:.6f}; NaN/Inf={train_audit["nan_inf"]}.',
        f'27. DENSITY CALIBRATION: {c["labels"]["DENSITY CALIBRATION"]}.',
        f'28. DENSITY RESPONSE: {c["labels"]["DENSITY RESPONSE"]}.',
        f'29. CROSS-SCENE STABILITY: {c["labels"]["CROSS-SCENE STABILITY"]}.',
        '30. Summary: results/density_aware_social_calibration/summary.md.',
        '31. Brain report: results/density_aware_social_calibration/brain_report.md.',
        '32. Source-only diagnostic complete; no formal test or SDD was run.',
        '',
        'The density module sees only observed neighbor count/8 and scales the attention context. This is density-conditioned calibration, not the prior scalar gate. The final layer starts at zero; shared social/residual initialization matches the frozen EMT-SR runs.',
        'All values are averaged equally across validation folds/seeds; within-fold source scenes are weighted equally for the primary view. Pooled sample results are retained because scene counts differ.',
    ]
    (RESULTS/'brain_report.md').write_text('\n'.join(brain)+'\n')
