"""Thirty source validation forwards and a descriptive scene-shift audit."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.analysis.social_shift_features import SUMMARY_FEATURES, SHIFT_FEATURES, OBS_FEATURES, ATTENTION_FEATURES, RESPONSE_FEATURES
from scripts.social_shift_protocol import ROOT,RESULTS,SCENES,MODELS,SEEDS,read,dump,sha_file,freeze_protocol,verify_inputs,verify_history,require_validation
from scripts.social_shift_inference import extract_run,load_npz
from scripts.social_shift_statistics import collapse_windows,subset,summarize_scenes,overall_summary,correlation_audits,pairwise_shift,grouped_performance,saturation_by_scene,paired_response,scene_classifier
from scripts.social_shift_report import decisions,write_reports


def analyze(raw,config):
    unique={m:collapse_windows(raw[m]) for m in MODELS}
    for key in ('sample_uid','scene','ped_id','observation_frame_ids'):
        np.testing.assert_array_equal(unique[MODELS[0]][key],unique[MODELS[1]][key])
    for key in ('sample_uid','scene','fold','seed'):
        np.testing.assert_array_equal(raw[MODELS[0]][key],raw[MODELS[1]][key])
    windows=unique['emt_sr']
    quartiles=np.quantile(windows['max_closing_speed'],[.25,.5,.75]).tolist()
    count={s:{'unique_windows':int((windows['scene']==s).sum()),'unique_pedestrians':len(np.unique(windows['ped_id'][windows['scene']==s])),'prediction_records_per_model':int((raw['emt_sr']['scene']==s).sum()),'neighbor_count_histogram':{str(n):int(((windows['scene']==s)&(windows['neighbor_count']==n)).sum()) for n in range(9)}} for s in SCENES}
    counts={'prediction_records_per_model':len(raw['emt_sr']['scene']),'prediction_records_both_models':sum(len(v['scene']) for v in raw.values()),'unique_windows':len(windows['scene']),'unique_pedestrians':sum(v['unique_pedestrians'] for v in count.values()),'scenes':5,'source_folds':5,'seeds':3,'per_scene':count}
    out={'scope':config['scope'],'code_commit':config['code_commit'],'heldout_test_accessed':False,'historical_test_already_accessed':True,'trajectory_training':False,'completed_frozen_forwards':30,'sample_counts':counts,'motion_quartiles':quartiles,'models':{}}
    for model,samples in unique.items():
        np.savez_compressed(RESULTS/model/'unique_windows.npz',**samples)
        value={'per_scene':summarize_scenes(samples),'overall':{'pooled':overall_summary(samples),'scene_equal':overall_summary(samples,True)},'correlations':correlation_audits(samples),'shift':pairwise_shift(samples),'groups':grouped_performance(samples,quartiles),'saturation':saturation_by_scene(samples)}
        value['conditional_entropy_n_ge_2']={s:overall_summary(subset(samples,(samples['scene']==s)&(samples['neighbor_count']>=2)))['attention_entropy'] for s in SCENES}
        out['models'][model]=value
        dump(RESULTS/model/'correlation_audit.json',value['correlations'])
        dump(RESULTS/model/'performance_sensitivity.json',value['groups'])
        dump(RESULTS/model/'saturation_audit.json',value['saturation'])
        dump(RESULTS/'pairwise_shift'/f'{model}.json',value['shift'])
        for scene in SCENES:
            source=subset(raw[model],raw[model]['scene']==scene)
            dump(RESULTS/'per_scene'/scene/f'{model}.json',{'primary_unique_window_mean_response':value['per_scene'][scene],'raw_prediction_distribution':overall_summary(source),'conditional_entropy_n_ge_2':value['conditional_entropy_n_ge_2'][scene],'heldout_test_accessed':False,'historical_test_already_accessed':True})
    classifier,predictions=scene_classifier(windows,config['classifier'])
    out['classifier']=classifier
    dump(RESULTS/'data_audit/scene_classifier.json',classifier)
    if predictions is not None:
        np.savez_compressed(RESULTS/'data_audit/source_scene_classification_oof.npz',**predictions)
    out['paired_response']=paired_response(unique['emt_sr'],unique['ett_sr'])
    out['decisions']=decisions(out,config)
    rank=out['decisions']['combined_ranking']
    saturation=out['models']['emt_sr']['saturation']
    emt_corr=out['models']['emt_sr']['correlations']['scene_equal']['gain']
    strongest=sorted(({'feature':f,**v} for f,v in emt_corr.items() if v['Spearman'] is not None),key=lambda r:-abs(r['Spearman']))
    out['interpretation']=[f'主要 source-validation shift factor 为 {rank[0]["feature"]}，EMT/ETT 等权 mean |SMD|={rank[0]["mean_abs_SMD"]:.6f}。它是分布依赖性最大变量，不是因果解释或模型优劣排名。',f'N=8 饱和比例：'+', '.join(f'{s}={saturation[s]["fraction_N8"]:.6f}' for s in SCENES)+'. 计数右截断使该审计无法观察超过八名邻居的密度差异。','EMT scene-equal ADE gain 相关性（按 |Spearman| 排序）：'+', '.join(f'{v["feature"]}: Pearson={v["Pearson"]:.6f}, Spearman={v["Spearman"]:.6f}' for v in strongest[:5])+'. 残差大小、比值和 gain 共享预测变量，相关性不能解释为观测因素的因果作用。','Attention 响应需结合 n>=2 和固定邻居 count 内相关性；n<=1 entropy=0、top2 在 n<=2 时饱和是定义引入的效应。不同 scene 的 prediction response 还混合 checkpoint、pedestrian 和观测轨迹组成，source 审计不能证明正式 held-out 失败的机制。']
    dump(RESULTS/'comparison.json',out)
    dump(RESULTS/'data_audit/sample_counts.json',counts)
    write_reports(out)
    return out


def main(mode='source_validation',one_run=False):
    require_validation(mode)
    torch.set_num_threads(1)
    config=freeze_protocol()
    definitions={'observation_features':list(OBS_FEATURES),'attention_features':list(ATTENTION_FEATURES),'prediction_response_features':list(RESPONSE_FEATURES),'outcomes_only_not_classifier_features':['base_ADE','SR_ADE','ADE_gain','base_FDE','SR_FDE','FDE_gain'],'observation_time':'only 8 observation positions; relative motion uses last 2 observation frames','relative_speed_unit':'meter / sampling step','distance_unit':'meter','closing':config['closing_definition'],'attention':config['attention_definition'],'missing':config['zero_neighbor_definition'],'aggregation':config['primary_aggregation'],'epsilon':1e-8,'distance_bins':config['distance_bins'],'motion_quartiles':config['motion_quartiles'],'heldout_test_accessed':False,'historical_test_already_accessed':True}
    dump(RESULTS/'data_audit/feature_definition.json',definitions)
    records={m:[] for m in MODELS};audits=[]
    for fold in SCENES:
        for seed in SEEDS:
            for model in MODELS:
                folder=RESULTS/model/f'heldout_{fold}'/f'seed_{seed}'
                if (folder/'forward_audit.json').exists():
                    audit=read(folder/'forward_audit.json');assert audit['sample_sha256']==sha_file(folder/'samples.npz') and audit['weights_unchanged']
                    samples=load_npz(folder/'samples.npz')
                else:
                    samples,audit=extract_run(fold,model,seed)
                records[model].append(samples);audits.append(audit)
                # Preserve every run/scene distribution, before deduplication or weighting.
                scene_summaries=summarize_scenes(samples)
                per_scene={s:{f:scene_summaries[s][f] for f in SUMMARY_FEATURES} for s in SCENES if s!=fold}
                dump(RESULTS/'per_fold'/f'heldout_{fold}_{model}_seed_{seed}.json',{'scene_distributions':per_scene,'source_sample_count':len(samples['scene']),'heldout_test_accessed':False,'historical_test_already_accessed':True})
                if one_run:
                    verify_inputs()
                    return
    assert len(audits)==30
    raw={m:{k:np.concatenate([r[k] for r in records[m]]) for k in records[m][0]} for m in MODELS}
    out=analyze(raw,config)
    verify_inputs();history=verify_history()
    outputs={str(p.relative_to(ROOT)):sha_file(p) for p in RESULTS.rglob('*') if p.is_file() and p.name!='result_integrity.json'}
    dump(RESULTS/'data_audit/result_integrity.json',{'source_forwards':30,'parameter_hash_unchanged_all_runs':True,'original_source_prediction_reproduced':True,'forward_audits':audits,'output_sha256':outputs,'historical_frozen_files':len(history),'sample_counts':out['sample_counts'],'heldout_test_accessed':False,'historical_test_already_accessed':True,'trajectory_training':False})
    print('SOCIAL DISTRIBUTION AUDIT COMPLETE: 30 frozen source forwards; no trajectory training',flush=True)
    for k,v in out['decisions']['labels'].items():
        print(f'{k}: {v}',flush=True)
    print('PRIMARY SHIFT FACTOR:',out['decisions']['PRIMARY SHIFT FACTOR'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',default='source_validation');parser.add_argument('--one-run',action='store_true')
    args=parser.parse_args();main(args.mode,args.one_run)
