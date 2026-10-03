"""Stored-prediction integrity and formal K=1 tables; no model inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statistics
import numpy as np
from scripts.final_protocol import ROOT, RESULTS, SCENES, MODELS, SEEDS, dump, read, sha_file, verify_training_complete, verify_history, utc_now
from scripts.final_prediction_export import displacement_metrics

NAMES = {'ett': 'ETT', 'emt': 'EMT', 'ett_sr': 'ETT-SR', 'emt_sr': 'EMT-SR'}


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |', '|'+'|'.join('---' for _ in headers)+'|'] + ['| '+' | '.join(str(v) for v in row)+' |' for row in rows] + ['']


def comparison(averages, pairs, treated, control):
    delta = {m: averages[treated][m]-averages[control][m] for m in ('ADE', 'FDE')}
    differences = [{'fold': p['fold'], 'seed': p['seed'], **{f'delta_{m}': p['metrics'][treated][m]-p['metrics'][control][m] for m in ('ADE', 'FDE')}} for p in pairs]
    return {'treated': treated, 'control': control, 'absolute_difference': delta, 'relative_change_percent': {m: 100*(averages[treated][m]/averages[control][m]-1) for m in ('ADE', 'FDE')}, 'paired_wins': {m: sum(q[f'delta_{m}'] < 0 for q in differences) for m in ('ADE', 'FDE')}, 'paired_wins_both': sum(q['delta_ADE'] < 0 and q['delta_FDE'] < 0 for q in differences), 'paired_differences': differences}


def main():
    protocol, psha, complete = verify_training_complete()
    historical = verify_history()
    log = read(RESULTS/'data_audit/test_access_log.json')
    expected = {(f, m, s) for f in SCENES for m in MODELS for s in SEEDS}
    observed = {(q['fold'], q['model'], q['seed']) for q in log['evaluations']}
    assert observed == expected and len(log['evaluations']) == 60
    assert all(q['status'] == 'complete' and q['timestamp'] > complete['completed_at_utc'] for q in log['evaluations'])
    runs, training, folds, pairs, artifact_checks, candidates = [], [], {}, [], {}, {}
    for fold in SCENES:
        folds[fold] = {}
        for name in MODELS:
            values = []
            for seed in SEEDS:
                metrics_path = RESULTS/'test'/f'heldout_{fold}'/name/f'seed_{seed}'/'metrics_test.json'
                q = read(metrics_path)
                tr = read(RESULTS/'training'/f'heldout_{fold}'/name/f'seed_{seed}'/'training_report.json')
                assert q['protocol_sha256'] == psha and q['checkpoint_sha256'] == tr['checkpoint_sha256']
                assert q['heldout_test_evaluated'] and not q['nan_inf']
                artifact = ROOT/q['prediction_artifact_path']
                assert sha_file(artifact) == q['prediction_artifact_sha256']
                entry = next(v for v in log['evaluations'] if (v['fold'], v['model'], v['seed']) == (fold, name, seed))
                assert entry['metrics_sha256'] == sha_file(metrics_path) and entry['checkpoint_sha256'] == q['checkpoint_sha256']
                with np.load(artifact, allow_pickle=False) as z:
                    assert len(z['scene_id']) == q['test_sample_count'] and np.all(z['scene_id'] == fold)
                    assert np.all(np.diff(z['frame_ids'], axis=1) == 10)
                    np.testing.assert_array_equal(z['prediction_abs'], z['prediction_rel']+z['last_obs_pos'][:, None])
                    ade, fde = displacement_metrics(z['prediction_abs'], z['future_gt'])
                    np.testing.assert_array_equal(ade, z['sample_ADE'])
                    np.testing.assert_array_equal(fde, z['sample_FDE'])
                    assert abs(ade.mean()-q['test_ADE']) < 1e-12 and abs(fde.mean()-q['test_FDE']) < 1e-12
                    if name.endswith('_sr'):
                        np.testing.assert_allclose(z['prediction_rel'], z['base_prediction']+z['social_residual'], rtol=0, atol=1e-7)
                        mask = z['neighbor_mask']
                        attention = z['attention_weights']
                        assert np.all(attention[~mask] == 0) and np.all(np.isfinite(attention))
                        np.testing.assert_allclose(attention.sum(1), mask.any(1).astype(float), rtol=0, atol=1e-6)
                    if name == 'emt_sr':
                        with np.load(RESULTS/'predictions'/f'heldout_{fold}'/'emt'/f'seed_{seed}.npz', allow_pickle=False) as base:
                            for key in ('scene_id', 'ped_id', 'frame_ids', 'future_gt'):
                                np.testing.assert_array_equal(z[key], base[key])
                            improvement = base['sample_ADE']-z['sample_ADE']
                            rankings = {'top_improvement': np.argsort(-improvement, kind='stable')[:10], 'median_improvement': np.argsort(np.abs(improvement-np.median(improvement)), kind='stable')[:10], 'near_zero_improvement': np.argsort(np.abs(improvement), kind='stable')[:10], 'failure_cases': np.flatnonzero(improvement < 0)[np.argsort(improvement[improvement < 0], kind='stable')[:10]]}
                            candidates[f'{fold}/seed_{seed}'] = {category: [{'sample_index': int(i), 'scene_id': str(z['scene_id'][i]), 'ped_id': str(z['ped_id'][i]), 'frame_ids': z['frame_ids'][i].tolist(), 'EMT_ADE': float(base['sample_ADE'][i]), 'EMT_SR_ADE': float(z['sample_ADE'][i]), 'improvement_ADE': float(improvement[i]), 'prediction_artifact': q['prediction_artifact_path']} for i in indices] for category, indices in rankings.items()}
                values.append(q)
                runs.append(q)
                training.append(tr)
                artifact_checks[f'{fold}/{name}/{seed}'] = {'prediction_sha256': q['prediction_artifact_sha256'], 'sample_count': q['test_sample_count'], 'metrics_recomputed_from_saved_predictions': True}
            folds[fold][name] = {metric: {'mean': statistics.mean(v[f'test_{metric}'] for v in values), 'sample_sd': statistics.stdev(v[f'test_{metric}'] for v in values)} for metric in ('ADE', 'FDE')}
        for seed in SEEDS:
            pairs.append({'fold': fold, 'seed': seed, 'metrics': {name: {metric: next(q[f'test_{metric}'] for q in runs if (q['fold'], q['model'], q['seed']) == (fold, name, seed)) for metric in ('ADE', 'FDE')} for name in MODELS}})
    averages = {name: {m: statistics.mean(folds[f][name][m]['mean'] for f in SCENES) for m in ('ADE', 'FDE')} for name in MODELS}
    comparisons = {'A_EMT_vs_ETT': comparison(averages, pairs, 'emt', 'ett'), 'B_EMT_SR_vs_EMT': comparison(averages, pairs, 'emt_sr', 'emt'), 'C_EMT_SR_vs_ETT_SR': comparison(averages, pairs, 'emt_sr', 'ett_sr')}
    runtime = {name: {'parameter_count': next(q['parameter_count'] for q in runs if q['model'] == name), 'mean_batch_1_latency_ms': statistics.mean(q['runtime']['batch_1_latency_ms'] for q in runs if q['model'] == name), 'mean_batch_128_latency_ms': statistics.mean(q['runtime']['batch_128_latency_ms'] for q in runs if q['model'] == name), 'mean_training_peak_memory_bytes': statistics.mean(q['gpu_peak_memory_bytes'] for q in training if q['model'] == name), 'maximum_training_peak_memory_bytes': max(q['gpu_peak_memory_bytes'] for q in training if q['model'] == name), 'mean_inference_peak_memory_bytes_batch_128': statistics.mean(q['runtime']['batch_128_peak_memory_bytes'] for q in runs if q['model'] == name)} for name in MODELS}
    result = {'scope': 'formal ETH/UCY leave-one-scene-out held-out benchmark, K=1, world meters', 'protocol_sha256': psha, 'test_metrics_accessed_before_freeze': False, 'completed_training_runs': 60, 'completed_test_runs': 60, 'folds': folds, 'equal_weight_five_fold_average': averages, 'paired_runs': pairs, 'comparisons': comparisons, 'runtime': runtime, 'maximum_gradient_norm_before_clipping': max(q['maximum_gradient_norm_before_clipping'] for q in training), 'nan_inf': False, 'historical_frozen_files_verified': historical, 'runs': runs}
    dump(RESULTS/'comparison.json', result)
    dump(RESULTS/'visualization_candidates.json', {'scope': 'saved-data candidates only; no figures or model changes', 'selection': 'up to 10 per category/fold/seed; failure means EMT-SR ADE exceeds EMT ADE', 'candidates': candidates})
    dump(RESULTS/'data_audit/result_integrity.json', {'completed_training_runs': 60, 'completed_test_runs': 60, 'one_access_per_fold_model_seed': True, 'test_started_after_all_training_complete': True, 'protocol_sha256': psha, 'protocol_and_checkpoint_hashes_verified': True, 'historical_files_verified': historical, 'prediction_metrics_recomputed': True, 'nan_inf': False, 'test_metrics_accessed_before_freeze': False, 'verified_at_utc': utc_now(), 'artifacts': artifact_checks})
    main_rows = []
    for f in SCENES:
        main_rows.append([f.upper(), *[f"{folds[f][name]['ADE']['mean']:.6f} ± {folds[f][name]['ADE']['sample_sd']:.6f} / {folds[f][name]['FDE']['mean']:.6f} ± {folds[f][name]['FDE']['sample_sd']:.6f}" for name in MODELS]])
    main_rows.append(['Average', *[f"{averages[name]['ADE']:.6f} / {averages[name]['FDE']:.6f}" for name in MODELS]])
    main_table = table(['Held-out', *[NAMES[n]+' ADE / FDE (m)' for n in MODELS]], main_rows)
    comparison_table = table(['Comparison (treated − control)', 'ΔADE (m)', 'ΔFDE (m)', 'ADE %', 'FDE %', 'ADE wins /15', 'FDE wins /15'], [[NAMES[q['treated']]+' vs '+NAMES[q['control']], f"{q['absolute_difference']['ADE']:+.6f}", f"{q['absolute_difference']['FDE']:+.6f}", f"{q['relative_change_percent']['ADE']:+.4f}", f"{q['relative_change_percent']['FDE']:+.4f}", q['paired_wins']['ADE'], q['paired_wins']['FDE']] for q in comparisons.values()])
    epoch_table = table(['Fold', *[NAMES[n] for n in MODELS]], [[f.upper(), *[protocol['final_epochs'][f][n] for n in MODELS]] for f in SCENES])
    runtime_table = table(['Model', 'Total params', 'Trainable', 'B=1 ms', 'B=128 ms', 'Mean train peak MiB', 'Max train peak MiB', 'Mean B=128 inference peak MiB'], [[NAMES[n], runtime[n]['parameter_count']['total'], runtime[n]['parameter_count']['trainable'], f"{runtime[n]['mean_batch_1_latency_ms']:.6f}", f"{runtime[n]['mean_batch_128_latency_ms']:.6f}", f"{runtime[n]['mean_training_peak_memory_bytes']/2**20:.3f}", f"{runtime[n]['maximum_training_peak_memory_bytes']/2**20:.3f}", f"{runtime[n]['mean_inference_peak_memory_bytes_batch_128']/2**20:.3f}"] for n in MODELS])
    summary = ['# Final ETH/UCY Benchmark', '', 'Formal held-out test, K=1, 8 observed / 12 predicted steps at 2.5 Hz, world meters. All four source scenes used in full. No internal validation remains in final training.', '', f'Protocol SHA256: `{psha}`. All 60 models finished before test access. Exactly 60 logged one-pass evaluations; no test-based model/checkpoint/epoch selection.', '', '## Main table', '', *main_table, 'Per fold: three-seed mean ± sample SD. Average: mean over seeds within fold, then equal weight over five folds. Average has no pooled-window weighting and no fabricated aggregate SD.', '', '## Comparisons', '', *comparison_table, 'Negative differences favor the treated model. Paired wins compare the same fold and seed. No new GO/STOP or significance tests are applied.', '', '## Frozen epochs', '', *epoch_table, 'Epochs are the median of the three prior validation best_epoch values for each fold/model. Fixed AdamW lr=1e-3, weight decay=1e-4, batch 128, SmoothL1 beta=1, clip=5. No scheduler: the old validation-driven plateau monitor does not exist in full-source training. The last frozen epoch is saved, without early stopping.', '', '## Runtime', '', *runtime_table, 'Runtime uses exclusive GPU synchronized wall time, 20 warmups +100 full uncached forwards, batch 1/128, N=8 for SR. Diagnostic attention export is excluded. Peak memory is per-process PyTorch allocated memory; social training includes frozen context caches. Training times include four-process GPU contention. Runtime does not by itself support a general speed claim.', '', '## Audit and artifacts', '', f'Historical files unchanged: {historical}. NaN/Inf=false. Maximum pre-clip gradient norm: {result["maximum_gradient_norm_before_clipping"]:.6f}. All social backbones frozen/eval and initial predictions equal their corresponding final base.', '', 'All saved predictions recompute ADE/FDE exactly. Prediction NPZs include full IDs/frames/ground truth and social neighbors/attention/residual/base outputs. Visualization candidates are saved without drawing figures.', '', 'Validation EMT-ZR remains a separate, explicitly labeled capacity/social-information ablation. It is not mixed into this formal test table. Formal EMT-SR vs EMT combines capacity and social effects.', '', 'Full 30-item Chinese handoff: brain_report.md. Test log: data_audit/test_access_log.json. No tuning, additional models, statistical tests or plots follow this benchmark.']
    (RESULTS/'summary.md').write_text('\n'.join(summary).rstrip()+'\n')
    lines = ['# Final ETH/UCY Benchmark：大脑 AI 交付报告', '', '60/60 正式训练与 60/60 正式 held-out test 均完成。协议在 test 打开前冻结；所有结果原样报告，不根据 test 调整模型。', '']
    def section(i, title, text):
        lines.extend([f'## {i}. {title}', '', text, ''])
    section(1, 'Branch', '`feat/final_eth_ucy_benchmark`，基于 `feat/clean_social_validation` / `458dc02950d3949a653be729f35bfc3cc7d090c6`。')
    section(2, 'Latest commit', '最新为包含本报告的 `exp: evaluate final Mamba social residual benchmark` 提交；最终 SHA 见交付消息和 GitHub 分支提交记录。协议记录的代码 commit：`'+protocol['code_commit']+'`。')
    section(3, 'Push status', '交付前核对本地与 GitHub 分支 HEAD、完整文件树一致，最终推送状态见交付消息。Checkpoint 和数据保留在本地；逐样本预测以 NPZ 保存在本阶段 results/predictions/。')
    section(4, 'Protocol frozen SHA', '`'+psha+'`。协议文件：`results/final_eth_ucy_benchmark/protocol_frozen.json`；配置/代码/数据 SHA：`data_audit/protocol_hashes.json`。')
    section(5, 'Final epoch determination', 'ETT/EMT 从第一阶段、ETT-SR/EMT-SR 从 Clean Social Validation 读取各 fold/model 三 seed 的 best_epoch，中位数四舍五入取整数。来源及哈希在 `data_audit/epoch_determination.json`，数字不人工选择；test 打开后保持不变。Final train 无 validation，因此冻结为无 scheduler、固定原始 lr=1e-3；其余 optimizer/loss/batch/clip/结构不变。')
    section(6, '每 fold/model final epoch', '`configs/final_training_epochs.json`：')
    lines.extend(epoch_table)
    section(7, '60/60 training', '全部完成，均使用四个 source scene 的所有 eligible target windows，target-only 从随机初始化重训。SR 加载同 fold/seed 的 final full-source backbone，并冻结全部 temporal encoder/base decoder。最后一个冻结 epoch 的 checkpoint 为最终模型。`FINAL_TRAINING_COMPLETE.json` 记录 60 组 checkpoint/log 哈希。')
    section(8, 'Test 打开顺序', '全部 60 个模型完成、checkpoint/log/协议/epoch/finite 核验通过后，才生成 completion 文件并提交 pre-test training 证据；随后显式 `--enable-heldout-evaluation`。全部 access timestamp 晚于 completion timestamp，60 个 fold/model/seed 各访问一次。`test_metrics_accessed_before_freeze=false`。Smoke 只导出 source 样本，未预测 ETH 正式 test。')
    section(9, 'Train/test sample counts', '五个正式 fold 的全量窗口数：')
    train_counts = read(RESULTS/'data_audit/train_scene_counts.json')
    test_counts = read(RESULTS/'data_audit/test_scene_counts.json')
    lines.extend(table(['Held-out', 'Train scenes', 'Train samples', 'Test samples'], [[f.upper(), ', '.join(s.upper() for s in train_counts[f]['scene_list']), train_counts[f]['total'], test_counts[f]['total']] for f in SCENES]))
    section(10, 'Social neighbor 统计', 'Train/test 均为同 recording、全部 8 个 observation frame 可见、排除自己、最后观测帧距离最近 8 个，不使用 future eligibility、不使用 radius、不再保留 train/val 过滤。以下按 scene 记录；train fold 为相应四个 scene 的合并。')
    neighbor = read(RESULTS/'data_audit/neighbor_statistics.json')['per_scene']
    lines.extend(table(['Scene', 'Samples', 'Mean', 'Median', 'p90', 'Zero %', '0 %', '1–2 %', '3–4 %', '5+ %'], [[f.upper(), neighbor[f]['sample_count'], f"{neighbor[f]['mean_neighbor_count']:.6f}", neighbor[f]['median_neighbor_count'], neighbor[f]['p90_neighbor_count'], f"{100*neighbor[f]['zero_neighbor_ratio']:.4f}", *[f"{100*neighbor[f]['group_proportions'][g]:.4f}" for g in ('0', '1-2', '3-4', '5+')]] for f in SCENES]))
    neighbor_folds = read(RESULTS/'data_audit/neighbor_statistics.json')['folds']
    lines.extend(table(['Fold', 'Split', 'Samples', 'Mean', 'Median', 'p90', 'Zero %'], [[f.upper(), split, neighbor_folds[f][split]['sample_count'], f"{neighbor_folds[f][split]['mean_neighbor_count']:.6f}", neighbor_folds[f][split]['median_neighbor_count'], neighbor_folds[f][split]['p90_neighbor_count'], f"{100*neighbor_folds[f][split]['zero_neighbor_ratio']:.4f}"] for f in SCENES for split in ('train', 'test')]))
    for i, name in enumerate(MODELS, 11):
        section(i, NAMES[name]+'：15 组正式 test ADE/FDE', '单位 meter；确定性 K=1，非 minADE20/minFDE20。')
        lines.extend(table(['Held-out', 'Seed', 'ADE', 'FDE'], [[p['fold'].upper(), p['seed'], f"{p['metrics'][name]['ADE']:.9f}", f"{p['metrics'][name]['FDE']:.9f}"] for p in pairs]))
    section(15, '每 fold 3-seed mean ± sample SD', 'SD 使用 n−1 分母；主表：')
    lines.extend(main_table)
    section(16, '五 fold 等权平均', '先 fold 内平均三 seed，然后五 fold 等权。以下没有 pooled-window weighting：')
    lines.extend(table(['Model', 'ADE', 'FDE'], [[NAMES[n], f"{averages[n]['ADE']:.9f}", f"{averages[n]['FDE']:.9f}"] for n in MODELS]))
    for i, (key, q) in enumerate(comparisons.items(), 17):
        section(i, NAMES[q['treated']]+' vs '+NAMES[q['control']], '差值=treated−control，relative 分母为 control；负值表示 treated 更好。仅描述，不新增 GO/STOP。')
        lines.extend(table(['Metric', 'Δ (m)', 'Relative %', 'Paired wins /15'], [[m, f"{q['absolute_difference'][m]:+.9f}", f"{q['relative_change_percent'][m]:+.6f}", q['paired_wins'][m]] for m in ('ADE', 'FDE')]))
        lines.extend([f"两指标同时改善：{q['paired_wins_both']}/15。", ''])
    section(20, 'Parameters', 'SR trainable 仅 relation embedding、cross-attention、residual decoder；backbone frozen。')
    lines.extend(table(['Model', 'Total', 'Trainable', 'Frozen'], [[NAMES[n], *[runtime[n]['parameter_count'][k] for k in ('total', 'trainable', 'frozen')]] for n in MODELS]))
    section(21, 'Latency', '独占 GPU，完整 uncached 模型，eval/no_grad，20 warmups +100 synchronized CUDA forwards；B=1/128、SR N=8。不包含 attention 导出/数据传输。不据此作未经支持的 Mamba 更快声明。')
    lines.extend(runtime_table)
    section(22, 'GPU memory', '表中 train 峰值为单 process PyTorch allocated，SR 包括 frozen context cache；四 worker 共用 GPU，训练 wall time 含竞争，不能视为独立速度。各 run 与 inference 峰值完整保存。')
    section(23, 'NaN/Inf', '**false**。训练 loss、gradient、checkpoint tensor、预测、attention 与 artifact metrics 全部 finite。')
    section(24, 'Max gradient norm', f"裁剪前全局最大：**{result['maximum_gradient_norm_before_clipping']:.9f}**；clip 阈值保持 5.0。所有 SR zero-init max_abs_diff <1e-7，backbone state before/after SHA 相同。历史冻结文件 {historical} 个保持不变。")
    verification = read(RESULTS/'data_audit/verification.json')
    section(25, 'Pytest', f"**{verification['pytest_passed']} passed**，完整输出见 `data_audit/pytest.txt`。覆盖 source/test 隔离、full-source 数量、observation-only neighbor、N=8、frozen backbone、zero-init、protocol 修改拒绝、60-run guard、attention 导出预测不变、artifact 重算指标。")
    section(26, 'Test access log', '`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/data_audit/test_access_log.json`，记录每次首次 evaluation 的 UTC timestamp、branch、commit、protocol SHA、checkpoint SHA、fold/model/seed 和完成状态。')
    section(27, 'Prediction artifacts', '`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/predictions/heldout_<fold>/<model>/seed_<seed>.npz`，共 60 个。包括 scene/ped/frame、obs_abs、future_gt、prediction_abs/rel、last_obs_pos、逐样本 ADE/FDE。SR 额外包含 neighbor_abs/mask/relation/IDs/frames、attention_weights、social_residual、base_prediction。Attention 诊断另行计算，不替换原模型预测路径。保存预测可精确重算表中指标。')
    section(28, 'Visualization candidates', '`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/visualization_candidates.json`：每 fold/seed 按 EMT ADE−EMT-SR ADE 保存 top/median/near-zero/failure 每类至多 10 个候选，仅数据索引，未画图。')
    section(29, 'Summary', '`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/summary.md`。完整机器可读比较：`comparison.json`；完整性：`data_audit/result_integrity.json`。')
    section(30, 'Brain report 与停止', '`/home/lrj/Mamba-T/results/final_eth_ucy_benchmark/brain_report.md`。正式 EMT-SR vs EMT 同时包含 capacity 和 social 效果；validation EMT-ZR 的消融单独标注来源，不与正式 held-out 数值混表。没有执行 bootstrap/t-test/Wilcoxon、SDD、K=20、radius tuning、新模块、SOTA 对比或绘图。全部结果原样汇报，等待大脑 AI 审查；本阶段到此停止。')
    (RESULTS/'brain_report.md').write_text('\n'.join(lines).rstrip()+'\n')
    print({'average': averages, 'comparisons': {k: {v: q[v] for v in ('absolute_difference', 'relative_change_percent', 'paired_wins')} for k, q in comparisons.items()}, 'historical_unchanged': historical})


if __name__ == '__main__':
    main()
