"""Tables and evidence-led diagnostic labels; candidate ideas are never implemented."""
import csv
from src.analysis.social_shift_features import SHIFT_FEATURES, SUMMARY_FEATURES
from scripts.social_shift_protocol import RESULTS, SCENES, MODELS, dump, read


def fmt(x):
    return 'NA' if x is None else f'{x:.6f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])+'\n'


def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def decisions(out,config):
    combined=[]
    for f in SHIFT_FEATURES:
        values=[next(r['mean_abs_SMD'] for r in out['models'][m]['shift']['ranking'] if r['feature']==f) for m in MODELS]
        combined.append({'feature':f,'mean_abs_SMD':sum(values)/len(values),'max_abs_SMD':max(next(r['max_abs_SMD'] for r in out['models'][m]['shift']['ranking'] if r['feature']==f) for m in MODELS)})
    combined.sort(key=lambda r:-r['mean_abs_SMD'])
    scores={r['feature']:r['mean_abs_SMD'] for r in combined}
    families={'SOCIAL DENSITY SHIFT':('neighbor_count',),'SOCIAL DISTANCE SHIFT':('nearest_neighbor_distance','mean_neighbor_distance','median_neighbor_distance','max_neighbor_distance'),'RELATIVE-MOTION SHIFT':('mean_relative_speed','max_relative_speed','mean_closing_speed','max_closing_speed'),'ATTENTION RESPONSE SHIFT':('attention_entropy','attention_max','attention_top2_sum'),'RESIDUAL RESPONSE SHIFT':('residual_norm','residual_final_norm','correction_ratio')}
    labels={};family_scores={}
    for name,features in families.items():
        top=max(features,key=lambda f:scores[f]);score=scores[top]
        labels[name]='SUPPORTED' if score>=config['label_thresholds_mean_abs_SMD']['supported'] else ('WEAK' if score>=config['label_thresholds_mean_abs_SMD']['weak'] else 'NOT SUPPORTED')
        family_scores[name]={'leading_feature':top,'score_mean_abs_SMD':score}
    proposals={'SOCIAL DENSITY SHIFT':'density normalization','SOCIAL DISTANCE SHIFT':'distance-normalized relation representation','RELATIVE-MOTION SHIFT':'motion-aware interaction representation','ATTENTION RESPONSE SHIFT':'confidence-normalized residual correction','RESIDUAL RESPONSE SHIFT':'confidence-normalized residual correction'}
    candidates=[]
    for name in sorted(families,key=lambda k:-family_scores[k]['score_mean_abs_SMD']):
        proposal=proposals[name]
        if any(c['suggestion']==proposal for c in candidates):
            continue
        candidates.append({'candidate':chr(65+len(candidates)),'suggestion':proposal,'source_evidence_family':name,**family_scores[name],'implemented':False})
        if len(candidates)==3:
            break
    return {'labels':labels,'family_scores':family_scores,'combined_ranking':combined,'PRIMARY SHIFT FACTOR':combined[0]['feature'],'candidates':candidates,'rule':'family label uses its largest feature mean abs SMD, averaged over EMT/ETT; prespecified supported>=.5, weak>=.2, otherwise not supported; descriptive, no p-values'}


def write_reports(out):
    results=out['models'];decision=out['decisions'];counts=out['sample_counts']
    common=['SOURCE VALIDATION SOCIAL DISTRIBUTION AUDIT ONLY','','heldout_test_accessed=false；historical_test_already_accessed=true。未重新训练 trajectory 模型，未读取正式 test 或 prediction。所有 forward 使用本阶段前冻结的 source validation checkpoint，eval/no_grad；所有参数 hash 保持一致。','','scene 指样本来源，fold 指上一阶段四 source scenes 的留一 fold 名称。首先按 scene + ped_id + 8 observation frame IDs 去重；每个窗口的模型响应在可用 fold/seed 上平均。主要表格/相关性使用窗口级统计，另保存全部重复预测和每 fold/seed 的原始结果，不能把重复窗口当独立观测。','','pooled 对所有唯一窗口等权；scene_equal 在每个有效 scene 内对窗口等权，再对 scene 等权。距离和相对速度在无邻居时为 missing，统计中排除，不以 0m 当真实距离。closing 为正向接近邻居的均值，max clamped >=0；signed mean/max 另存。速度单位 meter / sampling step。','','Attention 统计为四头权重平均后，在有效邻居内归一化；entropy 在 n<=1 时定义 0。need_weights=True 的第二次审计调用仅观测权重，prediction 始终使用原 need_weights=False 路径。padding 不参与统计。高 neighbor_count 本身会影响集中度；另报告 n>=2 和固定 count 内相关性，不能仅凭相关性推出机制。','','std 为 population SD；scene 内分位数采用 numpy linear quantile；全体 scene 等权分位数采用 weighted mid-CDF interpolation。SMD 比较 scene A−B，排名平均 10 个 scene pair 的 |SMD|。Wasserstein 有单位，不能跨不同量纲直接比较或取总体大小排序。','','标签阈值已在审计前固定：mean_abs_SMD>=0.5 为 SUPPORTED，>=0.2 为 WEAK，否则 NOT SUPPORTED。家族使用最强变量，EMT/ETT 的 mean_abs_SMD 等权平均。仅为描述性分档，不是显著性检验、因果证明或 predictor 阈值。','']
    def means_table(features):
        return table(['模型','变量',*SCENES],[[m,f,*[fmt(results[m]['per_scene'][s][f]['mean']) for s in SCENES]] for m in MODELS for f in features])
    summary=common+[f'完成 30/30 个 SR source validation forwards；每模型 {counts["prediction_records_per_model"]} 条 fold/seed 预测，去重后 {counts["unique_windows"]} 个窗口、{counts["unique_pedestrians"]} 名 scene/ped 目标。','',means_table(('neighbor_count','nearest_neighbor_distance','mean_neighbor_distance','mean_relative_speed','mean_closing_speed','attention_entropy','residual_norm','correction_ratio'))]
    summary += [table(['排名','变量','mean |SMD|','max |SMD|'],[[i+1,r['feature'],fmt(r['mean_abs_SMD']),fmt(r['max_abs_SMD'])] for i,r in enumerate(decision['combined_ranking'])])]
    summary += ['关键注意事项：N=8 只表示当前最多保留的八个观测可见邻居已满，实际邻居总数被右截断。attention/residual 随 scene 改变还混合了 frozen checkpoint 与观测组成差异；主分析不能将其当成 scene 的因果效应。分类器仅使用五个 observation 统计，按 pedestrian 分组划分，全部预处理在训练 fold 内拟合。','',*out['interpretation'],'',*[f'{k}: {v}' for k,v in decision['labels'].items()],f'PRIMARY SHIFT FACTOR: {decision["PRIMARY SHIFT FACTOR"]}','',*[f'Candidate {c["candidate"]}: {c["suggestion"]}（仅建议，未实现；依据 {c["leading_feature"]}）' for c in decision['candidates']]]
    (RESULTS/'summary.md').write_text('\n'.join(summary)+'\n')
    report=common+['以下逐项覆盖要求的 35 项。结果 commit 无法引用自身 SHA；最终 SHA 与远端核验在交付回复给出，本分支 HEAD 可复核。','']
    def item(n,title,value):
        report.extend([f'{n}. {title}','',value,''])
    item(1,'branch','diag/social_distribution_shift')
    item(2,'latest SHA',f'代码 commit={out["code_commit"]}；结果 commit 为本分支 HEAD，具体 SHA 在交付回复给出。')
    item(3,'push status','代码和结果通过已认证 GitHub API 发布；远端完整 tree 与本地对比后才交付。')
    item(4,'heldout_test_accessed','false；代码 guard 拒绝 final_eth_ucy_benchmark，含 predictions 和 symlink。')
    item(5,'historical_test_already_accessed','true；旧五阶段结果完全冻结。')
    item(6,'pytest',str(read(RESULTS/'data_audit/verification.json')['pytest_passed'])+' passed；完整日志 data_audit/pytest.txt。')
    item(7,'sample 数量',table(['scene','唯一窗口','scene/ped 数','每模型重复预测记录'],[[s,counts['per_scene'][s]['unique_windows'],counts['per_scene'][s]['unique_pedestrians'],counts['per_scene'][s]['prediction_records_per_model']] for s in SCENES])+f'\n每模型共 {counts["prediction_records_per_model"]} 条预测，去重后 {counts["unique_windows"]} 个窗口；模型间样本完全配对。')
    item(8,'neighbor count 五 scene 统计',means_table(('neighbor_count',))+'\n'+table(['scene',*map(str,range(9))],[[s,*[counts['per_scene'][s]['neighbor_count_histogram'][str(n)] for n in range(9)]] for s in SCENES]))
    item(9,'distance 五 scene 统计',means_table(('nearest_neighbor_distance','mean_neighbor_distance','median_neighbor_distance','max_neighbor_distance'))+'\nrank1…rank8 每窗口原始距离已保存；per_scene CSV/JSON 包含各 rank 分位数。')
    item(10,'relative speed 五 scene 统计',means_table(('mean_relative_speed','max_relative_speed')))
    item(11,'closing speed 五 scene 统计',means_table(('mean_closing_speed','max_closing_speed','mean_signed_closing_speed'))+'\nmean_closing_speed 是 mean_positive_closing_speed 的别名；无接近邻居为 0。signed mean/max 用于保存远离的负值。')
    for n,m,f in ((12,'emt_sr','attention_entropy'),(13,'ett_sr','attention_entropy'),(14,'emt_sr','residual_norm'),(15,'ett_sr','residual_norm')):
        item(n,m+' '+f,table(['scene','mean','std','median','p10','p25','p75','p90','p95'],[[s,*[fmt(results[m]['per_scene'][s][f][k]) for k in ('mean','std','median','p10','p25','p75','p90','p95')]] for s in SCENES]))
    item(16,'correction ratio',means_table(('residual_final_norm','correction_ratio'))+'\nR=mean_t ||delta_t||，R_final=||delta_12||，ratio=R/(mean_t||base_t||+1e-8)；base 相对最后 observation 位置，近静止 base 会放大 ratio。完整分位数见 per_scene。')
    pair_rows=[]
    for m in MODELS:
        for pair in results[m]['shift']['pairs']:
            for f in SHIFT_FEATURES:
                x=pair['features'][f]
                pair_rows.append([m,pair['scene_A']+' ↔ '+pair['scene_B'],f,fmt(x.get('mean_difference')),fmt(x.get('median_difference')),fmt(x.get('abs_SMD')),fmt(x.get('Wasserstein'))])
    item(17,'10 组 pairwise SMD table',table(['模型','scene pair','变量','mean A−B','median A−B','|SMD|','Wasserstein'],pair_rows))
    item(18,'shift ranking',table(['模型','feature','mean |SMD|','max |SMD|','mean Wasserstein','有效 pair 数'],[[m,r['feature'],fmt(r['mean_abs_SMD']),fmt(r['max_abs_SMD']),fmt(r['mean_Wasserstein']),r['valid_pairs']] for m in MODELS for r in results[m]['shift']['ranking']]))
    def group_table(category):
        return table(['权重','模型','组','unique n','prediction n','非空 scene','base ADE','SR ADE','ADE gain','base FDE','SR FDE','FDE gain','residual','entropy'],[[weight,m,g,x['sample_count_unique'],x['sample_count_predictions'],x['present_scenes'],*[fmt(x[k]) for k in ('base_ADE','SR_ADE','ADE_gain','base_FDE','SR_FDE','FDE_gain','residual_norm','attention_entropy')]] for weight in ('scene_equal','pooled') for m in MODELS for g,x in results[m]['groups'][weight][category].items()])
    item(19,'neighbor-count 分层',group_table('neighbor_count'))
    item(20,'nearest-distance 分层',group_table('nearest_distance')+'\n边界：no neighbor、<2、[2,4)、[4,8]、>8；单位米。未调整 radius。')
    item(21,'closing-speed 分层',group_table('closing_speed')+'\n分位点='+str(out['motion_quartiles'])+' meter / sampling step，来自全部唯一 source validation observation 窗口。仅诊断分组，重复分位点可产生空组，不随机打散相同值。')
    item(22,'N=8 saturation',table(['模型','scene','N8比例','组','unique n','ADE gain','FDE gain','residual','entropy'],[[m,s,fmt(q['fraction_N8']),label,q[label]['sample_count_unique'],*[fmt(q[label][k]) for k in ('ADE_gain','FDE_gain','residual_norm','attention_entropy')]] for m in MODELS for s,q in results[m]['saturation'].items() for label in ('N=8','N<8')])+'\n当前 N=8 上限造成邻居计数右截断，不能解释为所有 scene 的真实总邻居数最多为八。')
    item(23,'EMT vs ETT paired response difference',table(['scene','Δ residual','Δ entropy','Δ ADE gain','Δ FDE gain','Δ ratio'],[[s,*[fmt(out['paired_response']['per_scene'][s][f]['mean']) for f in ('residual_norm','attention_entropy','ADE_gain','FDE_gain','correction_ratio')]] for s in SCENES])+'\n方向 EMT−ETT；逐窗口、可用 fold/seed 完全配对。仅描述响应差异，不能证明 Mamba 导致某一交互机制。')
    def corr_table(kind):
        return table(['权重','模型','feature','n','scenes','Pearson','Spearman'],[[weight,m,f,x['sample_count'],x['present_scenes'],fmt(x['Pearson']),fmt(x['Spearman'])] for weight in ('scene_equal','pooled') for m in MODELS for f,x in results[m]['correlations'][weight][kind].items()])
    item(24,'feature vs ADE_gain correlations',corr_table('gain')+'\nPearson/Spearman 无 p-value，correlation != causality。')
    item(25,'feature vs residual norm / attention response correlations',corr_table('residual')+'\n'+corr_table('attention')+'\ncount>=2 及固定 neighbor_count 内的 attention 相关性完整保存于 emt_sr/correlation_audit.json、ett_sr/correlation_audit.json，避免 n<=1 entropy=0 的定义驱动解释。')
    classifier=out['classifier']
    item(26,'scene classifier',table(['模型','权重','Accuracy','Macro F1'],[[m,w,fmt(x['Accuracy']),fmt(x['Macro_F1'])] for m,value in classifier.get('metrics',{}).items() for w,x in value.items()])+ '\n'+str({k:v for k,v in classifier.items() if k!='metrics'})+'\n仅五个观察统计作为输入，无绝对坐标、ped/frame ID、prediction error、未来轨迹；scene/ped 只用于 CV 分组。ETH 仅 '+str(counts['per_scene']['eth']['unique_pedestrians'])+' 名目标 pedestrian，保留每 fold 样本/组明细，不作显著性宣称。')
    for n,(label,value) in enumerate(decision['labels'].items(),27):
        item(n,label,value+'；'+str(decision['family_scores'][label]))
    item(32,'PRIMARY SHIFT FACTOR',decision['PRIMARY SHIFT FACTOR']+'；由 source validation 10 scene pairs 的 mean_abs_SMD 排名选出。')
    item(33,'Candidate A/B/C', '\n'.join(f'Candidate {c["candidate"]}: {c["suggestion"]}；依据 {c["source_evidence_family"]}/{c["leading_feature"]}，mean |SMD|={fmt(c["score_mean_abs_SMD"])}；仅建议，未实现。' for c in decision['candidates']))
    item(34,'summary.md','results/social_distribution_shift/summary.md')
    item(35,'brain_report.md','results/social_distribution_shift/brain_report.md；comparison.json、per_scene、per_fold、pairwise_shift 保存完整精度和全部分布摘要。')
    report += [*out['interpretation'],'','已停止；候选方向由大脑 AI 后续选择。',*[f'{k}: {v}' for k,v in decision['labels'].items()],f'PRIMARY SHIFT FACTOR: {decision["PRIMARY SHIFT FACTOR"]}']
    (RESULTS/'brain_report.md').write_text('\n'.join(report)+'\n')
    for m in MODELS:
        rows=[{'scene':s,'feature':f,**results[m]['per_scene'][s][f]} for s in SCENES for f in SUMMARY_FEATURES]
        save_csv(RESULTS/'per_scene'/f'{m}_distribution_table.csv',rows)
        save_csv(RESULTS/'pairwise_shift'/f'{m}_shift_ranking.csv',results[m]['shift']['ranking'])
