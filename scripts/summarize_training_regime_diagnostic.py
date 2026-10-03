"""Analyze only source validation; formal averages are quoted from the request."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.training_regime_protocol import ROOT, RESULTS, MODELS, SCENES, SEEDS, read, dump, sha_file, verify_protocol, verify_historical_results, utc_now
from scripts.training_regime_utils import paths, diagnostic_best
from scripts.eth_ucy_utils import state_hash
from scripts.training_regime_metrics import report_samples

LABELS = {'ett': 'ETT-fixed', 'emt': 'EMT-fixed', 'ett_sr': 'ETT-SR-fixed', 'emt_sr': 'EMT-SR-fixed'}


def model_summary(rows):
    folds = {}
    for fold in SCENES:
        subset = [r for r in rows if r['fold'] == fold]
        folds[fold] = {m: {'mean': float(np.mean([r[f'validation_{m}'] for r in subset])), 'sd': float(np.std([r[f'validation_{m}'] for r in subset], ddof=1))} for m in ('ADE', 'FDE')}
    return {'folds': folds, 'five_fold_equal_mean': {m: float(np.mean([folds[f][m]['mean'] for f in SCENES])) for m in ('ADE', 'FDE')}}


def comparison(candidate, reference):
    assert [(r['fold'], r['seed']) for r in candidate] == [(r['fold'], r['seed']) for r in reference]
    out = {'direction': 'candidate minus reference; negative delta/percent is better', 'paired_runs': []}
    for a, b in zip(candidate, reference):
        row = {'fold': a['fold'], 'seed': a['seed']}
        for m in ('ADE', 'FDE'):
            row[f'delta_{m}'] = a[f'validation_{m}']-b[f'validation_{m}']
        out['paired_runs'].append(row)
    a, b = model_summary(candidate)['five_fold_equal_mean'], model_summary(reference)['five_fold_equal_mean']
    for m in ('ADE', 'FDE'):
        out[m] = {'candidate': a[m], 'reference': b[m], 'delta': a[m]-b[m], 'percent': 100*(a[m]-b[m])/b[m], 'wins': sum(r[f'delta_{m}'] < 0 for r in out['paired_runs']), 'ties': sum(r[f'delta_{m}'] == 0 for r in out['paired_runs']), 'total': len(candidate)}
    return out


def aggregate_strata(rows, key):
    labels = rows[0][key]
    out = {}
    for label in labels:
        fields = [k for k in rows[0][key][label] if k != 'sample_count']
        folds = {}
        for fold in SCENES:
            subset = [r[key][label] for r in rows if r['fold'] == fold]
            total = sum(r['sample_count'] for r in subset)
            q = {'sample_count_across_seed_predictions': total}
            for field in fields:
                q[field] = sum(r['sample_count']*r[field] for r in subset if r[field] is not None)/total if total else None
            folds[fold] = q
        out[label] = {'sample_count_across_seed_predictions': sum(q['sample_count_across_seed_predictions'] for q in folds.values()), 'present_folds': sum(q['sample_count_across_seed_predictions'] > 0 for q in folds.values()), 'folds': folds}
        for field in fields:
            vals = [q[field] for q in folds.values() if q[field] is not None]
            out[label][field] = float(np.mean(vals)) if vals else None
    return out


def norm_summary(rows):
    # Mean of 15 run-level statistics, including quantiles; never a pooled quantile.
    return {name: {stat: float(np.mean([r['correction_statistics'][name][stat] for r in rows])) for stat in ('mean', 'median', 'p90', 'p95', 'max')} for name in ('residual_norm', 'base_trajectory_norm', 'correction_ratio')}


def assert_numeric_equal(a, b, path=''):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            assert_numeric_equal(a[k], b[k], f'{path}/{k}')
    elif a is None:
        assert b is None, path
    else:
        assert np.isclose(a, b, rtol=1e-10, atol=1e-12), (path, a, b)


def audit_runs(protocol, psha):
    rows, inventory = {m: [] for m in MODELS}, []
    for fold in SCENES:
        for seed in SEEDS:
            sample_identity = None
            bases = {}
            for model in MODELS:
                folder, checkpoint = paths(fold, model, seed)
                q = read(folder/'metrics_validation.json')
                history = read(folder/'validation_history.json')
                n = protocol['final_epochs'][fold][model]
                assert q['protocol_sha256'] == psha and q['epochs_run'] == q['final_epoch'] == len(history) == n
                assert [h['epoch'] for h in history] == list(range(1, n+1))
                assert all(h['lr'] == .001 for h in history)
                assert not q['validation_controls_training'] and not q['early_stopping'] and q['scheduler'] is None
                assert not q['heldout_test_accessed'] and q['historical_test_already_accessed'] and not q['nan_inf']
                assert q['checkpoint_rule'] == 'last epoch' and not q['best_used_for_checkpoint_selection']
                assert q['train_validation_pedestrian_overlap'] == q['train_cross_split_neighbor_violations'] == 0
                assert fold not in q['train_scene_list'] and set(q['train_scene_list']) == set(SCENES)-{fold}
                for k, value in diagnostic_best(history).items():
                    assert q[k] == value
                assert history[-1] == q['final_epoch_metrics']
                assert sha_file(checkpoint) == q['checkpoint_sha256']
                ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
                assert ck['epoch'] == n and ck['protocol_sha256'] == psha and ck['final_model_state_sha256'] == q['final_model_state_sha256']
                social = model.endswith('_sr')
                tensors = ck['new_modules' if social else 'model']
                assert all(torch.isfinite(t).all() for t in tensors.values())
                if social:
                    base = bases[model.split('_')[0]]
                    assert q['base_checkpoint_sha256'] == base['checkpoint_sha256']
                    assert q['backbone_state_sha256_before'] == q['backbone_state_sha256_after'] == base['final_model_state_sha256']
                    assert q['initial_output_base_max_abs_diff'] < 1e-7 and q['backbone_unchanged'] and q['backbone_frozen'] and q['backbone_eval']
                    base_ck = torch.load(ROOT/base['checkpoint_path'], map_location='cpu', weights_only=True)
                    tensors = {**{f'backbone.{k}': v for k, v in base_ck['model'].items()}, **tensors}
                else:
                    bases[model] = q
                assert state_hash(tensors) == q['final_model_state_sha256']
                sample_file = folder/'validation_samples.npz'
                assert sha_file(sample_file) == q['validation_samples_sha256']
                with np.load(sample_file, allow_pickle=False) as z:
                    samples = {k: z[k] for k in z.files}
                assert set(samples['scene_id']) == set(SCENES)-{fold}
                identity = tuple(samples[k].tobytes() for k in ('scene_id', 'target_ped_id', 'frame_ids'))
                if sample_identity is None:
                    sample_identity = identity
                assert identity == sample_identity
                for k, value in report_samples(samples, social).items():
                    assert_numeric_equal(value, q[k], k)
                rows[model].append(q)
                inventory.append({'fold': fold, 'seed': seed, 'model': model, 'epoch': n, 'path': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': q['checkpoint_sha256'], 'state_sha256': q['final_model_state_sha256'], 'sample_sha256': q['validation_samples_sha256'], 'finite': True, 'last_epoch_verified': True})
    return rows, inventory


def historical_source_rows():
    rows, files = {m: [] for m in MODELS}, {}
    for fold in SCENES:
        for seed in SEEDS:
            for model in MODELS:
                if model.endswith('_sr'):
                    path = ROOT/'results/clean_social_validation'/f'heldout_{fold}'/model/f'seed_{seed}/metrics_validation.json'
                    q = read(path)
                else:
                    path = ROOT/'results/clean_social_validation/data_audit/base_validation'/f'{fold}_{model}_{seed}.json'
                    q = read(path)['reproduced_validation']
                    q = {f'validation_{m}': q[m] for m in ('ADE', 'FDE')}
                files[str(path.relative_to(ROOT))] = sha_file(path)
                rows[model].append({**q, 'fold': fold, 'seed': seed})
    return rows, files


def diagnose(comparisons, norms, protocol, gaps):
    emt = comparisons['emt_sr_vs_emt']
    ett = comparisons['ett_sr_vs_ett']
    old = comparisons['fixed_emt_sr_vs_old_clean_emt_sr']
    emt_improves = all(emt[m]['delta'] < 0 for m in ('ADE', 'FDE'))
    emt_worse = all(emt[m]['delta'] >= 0 for m in ('ADE', 'FDE'))
    ett_improves = all(ett[m]['delta'] < 0 for m in ('ADE', 'FDE'))
    # Descriptive labels, no statistical significance or causal thresholds implied.
    regime = 'SUPPORTED' if emt_worse and gaps['emt_sr']['final_best_gap_ADE'] > 0 else ('PARTIAL' if any(old[m]['delta'] > 0 for m in ('ADE', 'FDE')) or not emt_improves else 'NOT SUPPORTED')
    larger = all(norms['emt_sr'][k]['mean'] > norms['ett_sr'][k]['mean'] for k in ('residual_norm', 'correction_ratio'))
    compatibility = 'SUPPORTED' if ett_improves and emt_worse and larger else ('NOT SUPPORTED' if emt_improves and ett_improves and not larger else 'UNCLEAR')
    formal = protocol['historical_formal_context']
    formal_worse = all(formal['emt_sr'][m] > formal['emt'][m] for m in ('ADE', 'FDE'))
    cross = 'SUPPORTED' if emt_improves and formal_worse else 'UNCLEAR'
    return {'TRAINING REGIME MISMATCH': regime, 'MAMBA-SOCIAL COMPATIBILITY RISK': compatibility, 'CROSS-SCENE GENERALIZATION RISK': cross}


def fmt(x):
    return 'NA' if x is None else f'{x:.6f}'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |', *['| '+' | '.join(str(x) for x in r)+' |' for r in rows]])+'\n'


def write_reports(out):
    rows, means = out['runs'], out['model_summary']
    comparisons, labels = out['comparisons'], out['diagnostic_labels']
    common = ['SOURCE VALIDATION DIAGNOSTIC ONLY', '', '本阶段 heldout_test_accessed=false；historical_test_already_accessed=true。所有新指标来自原 source validation；正式均值仅引用用户请求，没有加载或重算正式 test。', '', '原 pedestrian manifest、clean neighbor 数据、模型结构和 final_training_epochs.json 保持冻结。先新训练 fixed backbone，再冻结其最后 epoch checkpoint 训练 SR。固定 AdamW lr=0.001，无 scheduler/early stop；validation 隔离 RNG 且验证模型状态不变；只保存最后 epoch。', '', '五 fold 等权，fold 内三个 seed 等权；标准差为三个 seed 的样本标准差。分层先在每 fold 按样本数汇总三个 seed，再对非空 fold 等权。sample_count 是三次 seed 预测的数量，同一 validation 样本被计数三次，不表示独立样本。', '', 'diagnostic best 按 minimum ADE 取 epoch，FDE 是同一 epoch 的 FDE；独立 minimum FDE 另列。best 仅事后分析，未用于 checkpoint 或正式 test。', '', '历史对比同时改变了 backbone 和 SR 的训练链、epoch、LR schedule、停止规则和 checkpoint 选择；不能解释为 scheduler 单因素因果消融。量纲：ADE/FDE/residual norm 为米，correction ratio 无量纲，epsilon=1e-8。', '']
    summary = common + [table(['模型', 'ADE', 'FDE'], [[LABELS[m], *[fmt(means[m]['five_fold_equal_mean'][k]) for k in ('ADE','FDE')]] for m in MODELS])]
    summary += [table(['比较 candidate − reference', 'ΔADE', 'ADE %', 'ADE wins/15', 'ΔFDE', 'FDE %', 'FDE wins/15'], [[key, fmt(c['ADE']['delta']), fmt(c['ADE']['percent']), c['ADE']['wins'], fmt(c['FDE']['delta']), fmt(c['FDE']['percent']), c['FDE']['wins']] for key,c in comparisons.items()])]
    summary += ['模型 final − best ADE gap（15 run 等权）：'+', '.join(f'{LABELS[m]}={fmt(out["gaps"][m]["final_best_gap_ADE"])}' for m in MODELS)+'.', '', '分层、每 run checkpoint/state SHA、曲线和全部数值见 comparison.json / brain_report.md。', '', '诊断标签为描述性证据判断，不代表因果证明或统计显著性。'+out['interpretation'], '', *[f'{k}: {v}' for k,v in labels.items()]]
    (RESULTS/'summary.md').write_text('\n'.join(summary)+'\n')
    report = common + ['以下完整覆盖要求的 28 项；结果提交不能在自身内容中引用自身 SHA，最终 SHA 和远端同步状态在交付回复给出，可用本分支 git rev-parse HEAD 核验。', '']
    def item(n, title, text):
        report.extend([f'{n}. {title}', '', text, ''])
    item(1, 'branch', 'diag/source_validation_training_regime')
    item(2, 'latest commit', f'代码提交：{out["code_commit"]}；结果提交为本分支 HEAD，具体 SHA 在交付回复给出。')
    item(3, 'push status', '代码与结果通过已认证 GitHub API 发布；最终远端 ref 和本地 tree 校验在交付回复给出。')
    item(4, '本阶段 held-out test 再次访问', 'false；source-only 数据入口和分析入口拒绝 heldout_test。旧 test loader 的 pytest 用合成 archive 验证路由。')
    item(5, 'historical_test_already_accessed', 'true；历史正式 test 已经打开，本阶段不重算。')
    item(6, '运行矩阵', f'{out["completed_runs"]}/60；另 ETH seed 42 四模型各 2 epoch smoke 通过。全部保存最后冻结 epoch。')
    for number, model in zip(range(7,11), MODELS):
        item(number, LABELS[model]+' 15 组 source val ADE/FDE', table(['fold','seed','epoch','ADE','FDE'], [[r['fold'],r['seed'],r['final_epoch'],fmt(r['validation_ADE']),fmt(r['validation_FDE'])] for r in rows[model]]))
    item(11, '五 fold 等权平均及 fold 内 seed 标准差', table(['模型','fold','ADE mean ± SD','FDE mean ± SD'], [[LABELS[m],f, fmt(means[m]['folds'][f]['ADE']['mean'])+' ± '+fmt(means[m]['folds'][f]['ADE']['sd']),fmt(means[m]['folds'][f]['FDE']['mean'])+' ± '+fmt(means[m]['folds'][f]['FDE']['sd'])] for m in MODELS for f in SCENES])+ '\n'+table(['模型','五 fold ADE','五 fold FDE'],[[LABELS[m],fmt(means[m]['five_fold_equal_mean']['ADE']),fmt(means[m]['five_fold_equal_mean']['FDE'])] for m in MODELS]))
    for number, key in ((12,'emt_sr_vs_emt'),(13,'ett_sr_vs_ett'),(14,'emt_sr_vs_ett_sr')):
        c = comparisons[key]
        item(number,key,table(['指标','candidate','reference','Δ','%','wins/15'],[[m,fmt(c[m]['candidate']),fmt(c[m]['reference']),fmt(c[m]['delta']),fmt(c[m]['percent']),c[m]['wins']] for m in ('ADE','FDE')]))
    keys = ('fixed_emt_sr_vs_old_clean_emt_sr','fixed_ett_sr_vs_old_clean_ett_sr','fixed_emt_vs_old_clean_base_emt','fixed_ett_vs_old_clean_base_ett')
    item(15,'Fixed vs Old Clean：SR 和对应旧 backbone',table(['比较 candidate − reference','ΔADE','ADE %','ADE wins/15','ΔFDE','FDE %','FDE wins/15'],[[key,fmt(comparisons[key]['ADE']['delta']),fmt(comparisons[key]['ADE']['percent']),comparisons[key]['ADE']['wins'],fmt(comparisons[key]['FDE']['delta']),fmt(comparisons[key]['FDE']['percent']),comparisons[key]['FDE']['wins']] for key in keys])+'\n旧 base 来自 Clean data_audit/base_validation 的 reproduced_validation，沿用当时真实 backbone checkpoint。旧结果文件 SHA 记录在 data_audit/historical_source_references.json。')
    item(16,'每 run final / diagnostic best / gap',table(['模型','fold','seed','final ADE','best epoch','best ADE','ADE gap','final FDE','FDE at best ADE','FDE gap','indep min FDE','min-FDE gap'],[[LABELS[m],r['fold'],r['seed'],fmt(r['validation_ADE']),r['diagnostic_best_validation_epoch'],fmt(r['diagnostic_best_validation_ADE']),fmt(r['final_best_gap_ADE']),fmt(r['validation_FDE']),fmt(r['diagnostic_best_validation_FDE']),fmt(r['final_best_gap_FDE']),fmt(r['diagnostic_minimum_FDE']),fmt(r['final_minimum_FDE_gap'])] for m in MODELS for r in rows[m]])+'\n'+table(['模型','final ADE mean','best ADE mean','ADE gap mean','FDE gap at best ADE mean','independent min-FDE gap mean'],[[LABELS[m],fmt(out['gaps'][m]['validation_ADE']),fmt(out['gaps'][m]['diagnostic_best_validation_ADE']),fmt(out['gaps'][m]['final_best_gap_ADE']),fmt(out['gaps'][m]['final_best_gap_FDE']),fmt(out['gaps'][m]['final_minimum_FDE_gap'])] for m in MODELS]))
    item(17,'每 fold 训练曲线结论',table(['fold','模型','best ADE epochs (42/123/2024)','mean final−best ADE','mean final−best FDE','train loss 下降且 last val ADE > best 的 runs'],[[f,LABELS[m],'/'.join(str(x['best_ADE_epoch']) for x in out['curve_conclusions'][f][m]['runs']),fmt(out['curve_conclusions'][f][m]['ADE_gap_mean']),fmt(out['curve_conclusions'][f][m]['FDE_gap_mean']),out['curve_conclusions'][f][m]['train_loss_down_final_ADE_above_best_runs']] for f in SCENES for m in MODELS])+'\n正 gap 表示固定最后 epoch 错过更优 source validation 状态；训练 loss 下降同时 gap > 0 是过拟合/优化波动的描述性迹象，两者不能仅凭曲线区分。')
    def strata_table(key, social_only=False):
        models = ('ett_sr','emt_sr') if social_only else MODELS
        result = []
        for model in models:
            for label,q in out['strata'][model][key].items():
                result.append([LABELS[model],label,q['sample_count_across_seed_predictions'],q['present_folds'],fmt(q.get('base_ADE')),fmt(q['ADE']),fmt(q.get('gain_ADE')),fmt(q.get('base_FDE')),fmt(q['FDE']),fmt(q.get('gain_FDE')),fmt(q.get('improvement_rate_ADE'))])
        return table(['模型','组','count (3 seeds)','非空 fold','base ADE','model ADE','ADE gain','base FDE','model FDE','FDE gain','ADE improvement rate'],result)
    item(18,'neighbor count 分层',strata_table('neighbor_count')+'\n分组 0/1–2/3–4/5+；fold/seed 原始分组值在每 run metrics_validation.json，fold 聚合值在 comparison.json。')
    item(19,'neighbor distance 分层',strata_table('neighbor_distance')+'\n按最后 observation frame 有效邻居距离均值：no-neighbor、<2m、[2m,4m]、>4m。未调整 radius。')
    for number,model in ((20,'emt_sr'),(21,'ett_sr')):
        item(number,LABELS[model]+' residual norm',table(['fold','seed','mean','median','p90','p95','max'],[[r['fold'],r['seed'],*[fmt(r['correction_statistics']['residual_norm'][k]) for k in ('mean','median','p90','p95','max')]] for r in rows[model]])+'\n15 run-level statistics 等权平均：'+str(out['norm_statistics'][model]['residual_norm']))
    item(22,'correction/base ratio',table(['模型','统计量','mean','median','p90','p95','max'],[[LABELS[m],stat,*[fmt(out['norm_statistics'][m][stat][k]) for k in ('mean','median','p90','p95','max')]] for m in ('ett_sr','emt_sr') for stat in ('base_trajectory_norm','correction_ratio')])+'\n这里是 15 个 run 内统计量的均值，median/p90/p95/max 列不是 pooled quantile；全局最大 ratio 单独记录在 comparison.json。BaseNorm 是相对最后 observation 位置的预测位移范数均值。近静止 base 会产生较大 ratio，应结合 residual 米制范数解释。')
    item(23,'correction magnitude 分层及潜在过度修正',strata_table('correction_magnitude',True)+'\n'+str(out['overcorrection_association'])+'\n分组边界 [0,.05],(.05,.10],(.10,.20],>.20。不同 backbone 的 ratio 分组成员不同，不能当作配对因果比较。大修正组 improvement rate 较低仅为关联。无邻居样本仍可通过 target context 产生 residual，不能将该组收益归因于社会交互。')
    item(24,'max gradient before clipping',fmt(out['maximum_gradient_norm_before_clipping'])+'；clip=5；各 run/epoch 原始梯度范数已保存。')
    item(25,'NaN/Inf', '无；60 run 的 checkpoint tensor、validation samples、history 及指标均完成 finite/hash/epoch 检查。历史冻结文件数：'+str(out['historical_frozen_files'])+'。')
    item(26,'pytest',str(read(RESULTS/'data_audit/verification.json')['pytest_passed'])+' passed；完整输出 data_audit/pytest.txt；被动 validation 的 CPU/CUDA 反事实测试验证 metric/RNG 改变不影响最终权重；编译检查见 data_audit/verification.json。')
    item(27,'summary.md','results/training_regime_diagnostic/summary.md')
    item(28,'brain_report.md','results/training_regime_diagnostic/brain_report.md；comparison.json 保留完整精度及每 run/fold 分层。')
    report += ['用户给出的历史正式均值（仅引用）：'+str(out['historical_formal_context']), '', out['interpretation'], '', '以下仅为诊断标签；不宣布最终论文模型，不启动下一阶段。', '', *[f'{k}: {v}' for k,v in labels.items()]]
    (RESULTS/'brain_report.md').write_text('\n'.join(report)+'\n')


def main(mode='source_validation'):
    if mode != 'source_validation':
        raise PermissionError('Diagnostic analysis rejects heldout_test before any file access')
    protocol, psha = verify_protocol()
    frozen_count = verify_historical_results()
    assert read(RESULTS/'smoke/smoke_gate.json')['passed']
    rows, inventory = audit_runs(protocol, psha)
    old, references = historical_source_rows()
    comparisons = {'emt_sr_vs_emt': comparison(rows['emt_sr'],rows['emt']), 'ett_sr_vs_ett': comparison(rows['ett_sr'],rows['ett']), 'emt_sr_vs_ett_sr': comparison(rows['emt_sr'],rows['ett_sr'])}
    for m in MODELS:
        key = f'fixed_{m}_vs_old_clean_{m}' if m.endswith('_sr') else f'fixed_{m}_vs_old_clean_base_{m}'
        comparisons[key] = comparison(rows[m],old[m])
    norms = {m:norm_summary(rows[m]) for m in ('ett_sr','emt_sr')}
    gaps = {m:{key:float(np.mean([r[key] for r in rows[m]])) for key in ('validation_ADE','validation_FDE','diagnostic_best_validation_ADE','diagnostic_best_validation_FDE','final_best_gap_ADE','final_best_gap_FDE','final_minimum_FDE_gap')} for m in MODELS}
    groups = {m:{key:aggregate_strata(rows[m],key) for key in ('neighbor_count','neighbor_distance',*(['correction_magnitude'] if m.endswith('_sr') else []))} for m in MODELS}
    curves = {}
    for fold in SCENES:
        curves[fold] = {}
        for model in MODELS:
            subset = [r for r in rows[model] if r['fold'] == fold]
            details = []
            for r in subset:
                folder,_ = paths(fold,model,r['seed'])
                history = read(folder/'validation_history.json')
                details.append({'seed':r['seed'],'best_ADE_epoch':r['diagnostic_best_validation_epoch'],'first_train_loss':history[0]['train_loss'],'final_train_loss':history[-1]['train_loss'],'final_ADE_above_best':r['final_best_gap_ADE'] > 0,'train_loss_down':history[-1]['train_loss'] < history[0]['train_loss']})
            curves[fold][model] = {'runs':details,'ADE_gap_mean':float(np.mean([r['final_best_gap_ADE'] for r in subset])),'FDE_gap_mean':float(np.mean([r['final_best_gap_FDE'] for r in subset])),'train_loss_down_final_ADE_above_best_runs':sum(r['train_loss_down'] and r['final_ADE_above_best'] for r in details)}
    association = {}
    for m in ('ett_sr','emt_sr'):
        low,high = groups[m]['correction_magnitude']['0-5%'],groups[m]['correction_magnitude']['>20%']
        association[m] = {'improvement_rate_0_5':low['improvement_rate_ADE'],'improvement_rate_over_20':high['improvement_rate_ADE'],'large_group_lower_improvement_rate':high['improvement_rate_ADE'] < low['improvement_rate_ADE'] if high['improvement_rate_ADE'] is not None and low['improvement_rate_ADE'] is not None else None,'causal_claim':False}
    labels = diagnose(comparisons,norms,protocol,gaps)
    interpretation = ('Source validation 中 fixed EMT-SR 对自身 base 的变化和 final−best gap 为训练制度证据；旧 Clean 对比是整个训练链改变。若 source 社交收益保留而用户提供的正式 test 社交收益为负，训练制度单独不能解释正式失败，支持跨场景泛化风险。Mamba 专属兼容风险需要 ETT 收益保留、EMT 收益丢失以及更大 EMT 修正共同支持。标签不代表统计显著性，15 个运行共享 fold 数据；不将残差大小与错误率关联解释为因果。')
    out = {'scope':'SOURCE VALIDATION DIAGNOSTIC ONLY','heldout_test_accessed':False,'historical_test_already_accessed':True,'protocol_sha256':psha,'code_commit':protocol['code_commit'],'completed_runs':len(inventory),'runs':rows,'model_summary':{m:model_summary(rows[m]) for m in MODELS},'historical_model_summary':{m:model_summary(old[m]) for m in MODELS},'comparisons':comparisons,'gaps':gaps,'curve_conclusions':curves,'strata':groups,'strata_aggregation':'seed predictions sample-weighted within fold; equal mean over nonempty folds; counts count repeated seed predictions','norm_statistics':norms,'norm_aggregation':'mean of 15 run-level mean/median/p90/p95/max; not pooled quantiles','global_maximum_correction_ratio':{m:max(r['correction_statistics']['correction_ratio']['max'] for r in rows[m]) for m in ('ett_sr','emt_sr')},'overcorrection_association':association,'diagnostic_labels':labels,'interpretation':interpretation,'maximum_gradient_norm_before_clipping':max(r['maximum_gradient_norm_before_clipping'] for runs in rows.values() for r in runs),'nan_inf':False,'historical_frozen_files':frozen_count,'historical_formal_context':protocol['historical_formal_context']}
    assert out['completed_runs'] == 60
    dump(RESULTS/'data_audit/checkpoint_inventory.json',inventory)
    dump(RESULTS/'data_audit/historical_source_references.json',{'file_sha256':references,'source_validation_only':True})
    dump(RESULTS/'data_audit/result_integrity.json',{'completed_runs':60,'verified_at_utc':utc_now(),'checkpoint_sha_state_hash_epoch_finite_passed':True,'sample_sha_metrics_reproduction_and_order_passed':True,'historical_frozen_files':frozen_count,'heldout_test_accessed':False,'historical_test_already_accessed':True,'protocol_sha256':psha})
    dump(RESULTS/'comparison.json',out)
    write_reports(out)
    print('SOURCE VALIDATION DIAGNOSTIC ONLY: 60/60 complete',flush=True)
    for k,v in labels.items():
        print(f'{k}: {v}',flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='source_validation')
    main(parser.parse_args().mode)
