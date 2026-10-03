"""Finalize descriptive interpretation from saved statistics; no data/model forward."""
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path('/home/lrj/Mamba-T/results/social_distribution_shift')


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, data):
    (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


c = read('comparison.json')
obs = {
    'neighbor_count', 'mean_neighbor_distance', 'nearest_neighbor_distance',
    'mean_relative_speed', 'mean_closing_speed', 'max_closing_speed',
}
rankings = {}
for model, m in c['models'].items():
    rankings[model] = {}
    for weight, audits in m['correlations'].items():
        rankings[model][weight] = {}
        for target in ('gain', 'residual'):
            rows = [{'feature': k, **v} for k, v in audits[target].items()]
            rankings[model][weight][target] = {}
            for scope in ('observation_only', 'all_reported_features'):
                eligible = [r for r in rows if scope != 'observation_only' or r['feature'] in obs]
                rankings[model][weight][target][scope] = {
                    measure: sorted(
                        [r for r in eligible if r[measure] is not None and math.isfinite(r[measure])],
                        key=lambda r: (-abs(r[measure]), r['feature']),
                    )
                    for measure in ('Pearson', 'Spearman')
                }
c['association_rankings'] = rankings

notes = []
notes.append(
    '计数的 scene dependency 最大，但不代表它最能解释 gain。最近距离的 mean |SMD|=0.398501；'
    'mean/max closing 的值分别为 0.154869/0.115847。RELATIVE-MOTION SHIFT 的 SUPPORTED '
    '由相对速度而非径向 closing 驱动；SOCIAL DISTANCE SHIFT 由距离集合变量驱动，不能推广为每个距离变量都强。'
)
for model, name in (('emt_sr', 'EMT'), ('ett_sr', 'ETT')):
    m = c['models'][model]
    gain = rankings[model]['scene_equal']['gain']['observation_only']
    p, s = gain['Pearson'][0], gain['Spearman'][0]
    r = m['correlations']['scene_equal']['residual']['mean_relative_speed']
    notes.append(
        f'{name} scene 等权：观察变量中 ADE gain 的最大 |Pearson| 来自 {p["feature"]} '
        f'({p["Pearson"]:.6f})，最大 |Spearman| 来自 {s["feature"]} ({s["Spearman"]:.6f})；'
        f'相对速度与 residual norm 的 Pearson/Spearman 为 {r["Pearson"]:.6f}/{r["Spearman"]:.6f}。'
        'gain 的单调关联总体较弱，残差大小与相对速度的关联较强。两种相关性排序和符号均保存在 association_rankings，'
        '不把残差与 gain 的共享预测量当成观察因素的因果作用。'
    )
emt = c['models']['emt_sr']['correlations']['scene_equal']
ett = c['models']['ett_sr']['correlations']['scene_equal']
e_all = emt['attention']['neighbor_count vs attention_entropy']
e_two = emt['attention_n_ge_2']['neighbor_count vs attention_entropy']
e_max = emt['attention_n_ge_2']['neighbor_count vs attention_max']
e_dist = emt['within_neighbor_count']['8']['mean_neighbor_distance vs attention_max']
e_close = emt['within_neighbor_count']['8']['max_closing_speed vs attention_max']
t_entropy = ett['attention_n_ge_2']['neighbor_count vs attention_entropy']
t_dist = ett['within_neighbor_count']['8']['mean_neighbor_distance vs attention_max']
t_close = ett['within_neighbor_count']['8']['max_closing_speed vs attention_max']
notes.append(
    f'EMT 的 count–entropy Pearson 从全体 {e_all["Pearson"]:.6f} 降为 n>=2 的 {e_two["Pearson"]:.6f} '
    f'(Spearman={e_two["Spearman"]:.6f})，全体相关性受到 n<=1 entropy=0 的定义影响。'
    f'n>=2 的 count–attention_max 仍有关联 ({e_max["Pearson"]:.6f}/{e_max["Spearman"]:.6f})；'
    f'固定 N=8 后，mean distance–attention_max 为 {e_dist["Pearson"]:.6f}/{e_dist["Spearman"]:.6f}，'
    f'max closing–attention_max 为 {e_close["Pearson"]:.6f}/{e_close["Spearman"]:.6f}。'
    '因此 attention concentration 与密度相关，但本审计不支持“仅由数量驱动、与交互状态无关”。'
)
notes.append(
    f'ETT 的 n>=2 count–entropy 为 {t_entropy["Pearson"]:.6f}/{t_entropy["Spearman"]:.6f}；'
    f'固定 N=8 的 mean distance–attention_max 为 {t_dist["Pearson"]:.6f}/{t_dist["Spearman"]:.6f}，'
    f'max closing–attention_max 为 {t_close["Pearson"]:.6f}/{t_close["Spearman"]:.6f}。'
    '这显示 EMT/ETT 响应不同；backbone 和各自训练的 SR 模块共同参与响应，不能归因于单一 Mamba 机制。'
)
eth_e = c['models']['emt_sr']['saturation']['eth']
eth_t = c['models']['ett_sr']['saturation']['eth']
notes.append(
    f'ETH source-validation 的 N=8 只有 {eth_e["N=8"]["sample_count_unique"]} 个唯一窗口，'
    f'EMT ADE gain={eth_e["N=8"]["ADE_gain"]:.6f}，N<8 为 {eth_e["N<8"]["ADE_gain"]:.6f}；'
    f'ETT 对应 {eth_t["N=8"]["ADE_gain"]:.6f}/{eth_t["N<8"]["ADE_gain"]:.6f}。'
    '这不是 ETH formal test。其他 scene 的 N=8 ADE gain 为正，UNIV 没有 N<8 样本可作内部比较；'
    '不能断言 N=8 普遍有害，不能据此修改 N 或 radius。'
)
pair = c['paired_response']['per_scene']
assert all(v['residual_norm']['mean'] > 0 and v['attention_entropy']['mean'] > 0 for v in pair.values())
notes.append(
    '同一窗口配对后，EMT 的 mean residual norm 和 entropy 的 scene 均值在五个 scene 都高于 ETT。'
    'EMT−ETT 的 ADE gain 均值只有 UNIV 为正，其他四个 scene 为负；更大的 correction 不意味着更大的 gain。'
    '本阶段只报告响应现象，不给出机制因果解释或模型优劣排名。'
)
cm = c['classifier']['metrics']
notes.append(
    f'按 pedestrian 分组的 scene classifier，logistic scene 等权 Accuracy={cm["logistic"]["scene_equal"]["Accuracy"]:.6f}，'
    f'Macro F1={cm["logistic"]["scene_equal"]["Macro_F1"]:.6f}，高于 majority 的 '
    f'{cm["majority"]["scene_equal"]["Accuracy"]:.6f}/{cm["majority"]["scene_equal"]["Macro_F1"]:.6f}；'
    f'但 pooled Accuracy={cm["logistic"]["pooled"]["Accuracy"]:.6f} 低于 majority '
    f'{cm["majority"]["pooled"]["Accuracy"]:.6f}。五个观察统计含有一定 scene 识别信息，不能称为非常容易识别 scene，'
    '也不宣称显著性或证明 domain shift。ETH 只有 9 名目标 pedestrian，窗口相关，CV 第 2 fold 的 validation 仅含 4 类；'
    '分组避免目标 pedestrian 跨 fold，但不是对新 recording/domain 的验证。'
)
e = c['models']['emt_sr']['overall']['scene_equal']
t = c['models']['ett_sr']['overall']['scene_equal']
notes.append(
    f'本阶段 scene 等权 ADE/FDE gain：EMT={e["ADE_gain"]["mean"]:.6f}/{e["FDE_gain"]["mean"]:.6f}，'
    f'ETT={t["ADE_gain"]["mean"]:.6f}/{t["FDE_gain"]["mean"]:.6f}。'
    '先对同一窗口的四 source folds×三 seeds 平均，再对五 scene 等权；这与上一阶段按 fold 等权、'
    'fold 内按样本数汇总的统计对象不同。UNIV 占唯一窗口的 70.3373%，主结论使用 scene 等权；'
    '新的等权均值不覆盖或重算旧阶段结论。'
)
c['interpretation'] = c['interpretation'][:4] + notes
write('comparison.json', c)

block = '\n\n'.join(notes)
for name in ('summary.md', 'brain_report.md'):
    p = ROOT / name
    text = p.read_text()
    # Literal pipes in table headers need escaping in GitHub-flavored Markdown.
    text = text.replace('mean |SMD| | max |SMD|', 'mean abs(SMD) | max abs(SMD)')
    text = text.replace('| |SMD| |', '| abs(SMD) |')
    marker = '\nSOCIAL DENSITY SHIFT: SUPPORTED\n' if name == 'summary.md' else '\n已停止；候选方向由大脑 AI 后续选择。\n'
    assert marker in text
    start = '\n\n补充解释（所有数字来自上述 source-validation 统计）：\n\n'
    if start in text:
        text = text.split(start, 1)[0] + marker + text.split(marker, 1)[1]
    text = text.replace(marker, '\n\n补充解释（所有数字来自上述 source-validation 统计）：\n\n' + block + '\n' + marker, 1)
    p.write_text(text)

old = read('data_audit/numerical_audit_fix.json')
old['status'] = 'superseded exploratory attempt; not used in final analysis'
old['superseded_by'] = 'data_audit/attention_kernel_verification.json'
old['final_method'] = 'original SDPA identity-value probe; reconstructed context difference exactly zero in all 30 final runs'
write('data_audit/numerical_audit_fix.json', old)

integrity = read('data_audit/result_integrity.json')
audits = integrity['forward_audits']
keys = audits[0]['source_validation_reproduction_max_abs_diff']
maxdiff = {k: max(a['source_validation_reproduction_max_abs_diff'][k] for a in audits) for k in keys}
policy = {
    'prediction_reproduction_definition': 'numeric forward comparison within rtol=1e-4, atol=1e-5, not a bitwise prediction claim',
    'saved_ADE_FDE_residual_norm_correction_ratio_reused_unchanged': True,
    'missing_statistics_only': ['attention_entropy', 'attention_max', 'attention_top2_sum', 'residual_final_norm', 'observation social features'],
    'reproduction_max_abs_diff_all_runs': maxdiff,
    'attention_context_max_abs_diff_all_runs': max(a['attention_context_max_abs_diff'] for a in audits),
    'protocol_sha256': (ROOT/'protocol_frozen.sha256').read_text().strip(),
    'earlier_attempts': 'data_audit/attempt_1, attempt_2, attempt_3 are superseded forensic records; none used as final analysis samples',
    'historical_frozen_files': integrity['historical_frozen_files'],
    'heldout_test_accessed': False,
    'historical_test_already_accessed': True,
    'trajectory_training': False,
}
write('data_audit/final_reproduction_audit.json', policy)
integrity['prediction_reproduction_definition'] = policy['prediction_reproduction_definition']
integrity['saved_response_metrics_reused_unchanged'] = True
integrity['reproduction_max_abs_diff_all_runs'] = maxdiff
# Keep this deterministic derivation script alongside the derived results.
(ROOT/'data_audit/finalize_reports.py').write_bytes(Path(__file__).read_bytes())
files = [p for p in ROOT.rglob('*') if p.is_file() and p != ROOT/'data_audit/result_integrity.json']
integrity['output_sha256'] = {str(p.relative_to(ROOT.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
write('data_audit/result_integrity.json', integrity)
assert [int(n) for n in re.findall(r'^(\d+)\. ', (ROOT/'brain_report.md').read_text(), re.M)] == list(range(1, 36))
print(json.dumps({'derived_output_files': len(files), 'brain_report_items': 35, 'max_saved_response_diff': maxdiff}, indent=2))
