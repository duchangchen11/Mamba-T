from social_residual_utils import *
import statistics

NAMES={'ett':'ETT','emt':'EMT','emt_zr':'EMT-ZR','emt_sr':'EMT-SR','emt_gsr':'EMT-GSR','ett_sr':'ETT-SR'}
ALL_MODELS=tuple(NAMES)

def paired_comparison(folds,pairs,treated,control,ade_threshold=None,fde_threshold=None,min_wins=None):
    equal={n:{m:statistics.mean(folds[f][n][m]['mean'] for f in SCENES) for m in ('ADE','FDE')} for n in (treated,control)}
    changes={m:100*(equal[treated][m]/equal[control][m]-1) for m in ('ADE','FDE')}
    deltas=[{'fold':p['fold'],'seed':p['seed'],**{f'delta_{m}':p['metrics'][treated][m]-p['metrics'][control][m] for m in ('ADE','FDE')}} for p in pairs]
    wins={m:sum(q[f'delta_{m}']<0 for q in deltas) for m in ('ADE','FDE')}
    result={'treated':treated,'control':control,'five_fold_equal_weight_average':equal,'relative_change_percent':changes,'paired_wins':wins,'paired_wins_both':sum(q['delta_ADE']<0 and q['delta_FDE']<0 for q in deltas),'paired_differences':deltas}
    if min_wins is not None:
        qualifying=[m for m,threshold in (('ADE',ade_threshold),('FDE',fde_threshold)) if changes[m]<=-threshold and wins[m]>=min_wins]
        result.update({'qualifying_metrics':qualifying,'decision':'GO' if qualifying else 'STOP','rule':f'>={ade_threshold}% ADE or >={fde_threshold}% FDE improvement; >= {min_wins}/15 wins must hold for the same qualifying metric'})
    return result

def main():
    folds={};pairs=[];runs=[];gate_runs=[];group_runs=[]
    for fold in SCENES:
        samples={n:[] for n in ALL_MODELS}
        for seed in (42,123,2024):
            metrics={}
            for n in ALL_MODELS:
                if n in ('ett','emt'):
                    q=json.loads((BASE_RESULTS/f'heldout_{fold}'/n/f'seed_{seed}/metrics_validation.json').read_text())
                    group=json.loads((RESULTS/'data_audit/base_validation'/f'{fold}_{n}_{seed}.json').read_text())['reproduced_validation']['neighbor_groups']
                else:
                    q=json.loads((RESULTS/f'heldout_{fold}'/n/f'seed_{seed}/metrics_validation.json').read_text())
                    assert not q['smoke'] and q['protocol']['max_epochs']==50 and q['backbone_unchanged'] and not q['nan_inf']
                    assert q['initial_output_base_max_abs_diff']<1e-7
                    runs.append(q);group=q['neighbor_groups']
                    if 'gate_statistics' in q:gate_runs.append({'fold':fold,'seed':seed,**q['gate_statistics']})
                assert not q['heldout_test_accessed']
                metrics[n]={m:q[f'validation_{m}'] for m in ('ADE','FDE')};samples[n].append(metrics[n])
                group_runs.append({'fold':fold,'seed':seed,'model':n,'groups':group})
            pairs.append({'fold':fold,'seed':seed,'metrics':metrics})
        folds[fold]={n:{m:{'mean':statistics.mean(q[m] for q in samples[n]),'sample_sd':statistics.stdev(q[m] for q in samples[n])} for m in ('ADE','FDE')} for n in ALL_MODELS}
    assert len(runs)==60 and len(pairs)==15 and len(gate_runs)==15
    equal={n:{m:statistics.mean(folds[f][n][m]['mean'] for f in SCENES) for m in ('ADE','FDE')} for n in ALL_MODELS}
    social=paired_comparison(folds,pairs,'emt_sr','emt_zr',1.5,2.5,10)
    gate=paired_comparison(folds,pairs,'emt_gsr','emt_sr',.75,1.25,9)
    backbone=paired_comparison(folds,pairs,'emt_sr','ett_sr')
    capacity=paired_comparison(folds,pairs,'emt_zr','emt')
    audit=json.loads((RESULTS/'data_audit/neighbor_statistics.json').read_text())
    group_aggregates={}
    for n in ALL_MODELS:
        group_aggregates[n]={}
        for group in BUCKETS:
            foldmeans={}
            for f in SCENES:
                rows=[q['groups'][group] for q in group_runs if q['model']==n and q['fold']==f]
                assert len(rows)==3 and len(set(q['sample_count'] for q in rows))==1
                foldmeans[f]={'sample_count_per_seed':rows[0]['sample_count'],**{m:statistics.mean(q[m] for q in rows) if rows[0]['sample_count'] else None for m in ('ADE','FDE')}}
            valid=[q for q in foldmeans.values() if q['sample_count_per_seed']]
            group_aggregates[n][group]={'folds':foldmeans,'five_fold_equal_weight_mean':{m:statistics.mean(q[m] for q in valid) if valid else None for m in ('ADE','FDE')},'participating_fold_count':len(valid)}
    frozen=json.loads((RESULTS/'data_audit/stage1_freeze_inventory.json').read_text())
    assert all(sha_file(ROOT/path)==checksum for path,checksum in frozen.items())
    comparison={'scope':'source-scene validation only; no heldout evaluation','baseline_source':'frozen results/eth_ucy_mamba_baseline','folds':folds,'paired_runs':pairs,'equal_weight_five_fold_average':equal,'social_comparison':social,'gate_comparison':gate,'social_backbone_comparison':backbone,'capacity_comparison':capacity,'gate_statistics':gate_runs,'neighbor_group_metrics':group_runs,'neighbor_group_aggregates':group_aggregates,'neighbor_statistics':audit,'maximum_gradient_norm_before_clipping':max(q['maximum_gradient_norm_before_clipping'] for q in runs),'nan_inf':False,'stage1_frozen_inventory_verified':True,'heldout_test_accessed':False,'runs':runs}
    dump(RESULTS/'comparison.json',comparison)
    lines=['# ETH/UCY Frozen Social Residual Validation','',f'**SOCIAL: {social["decision"]}. GATE: {gate["decision"]}.** Source-scene validation only; heldout_test_accessed=false.','', 'Baselines reuse stage-one results and checkpoints. All temporal encoders and decoders remain frozen and in eval mode. No backbone fine-tuning, radius, neighbor-count search or held-out evaluation.','', '| Model | Equal-weight ADE (m) | Equal-weight FDE (m) |','|---|---|---|']
    for n in ALL_MODELS:lines.append(f'| {NAMES[n]} | {equal[n]["ADE"]:.6f} | {equal[n]["FDE"]:.6f} |')
    lines+=['','## Core comparisons','', '| Comparison | ΔADE % | ΔFDE % | ADE wins /15 | FDE wins /15 | Decision |','|---|---|---|---|---|---|']
    for label,q in (('EMT-SR vs EMT-ZR',social),('EMT-GSR vs EMT-SR',gate),('EMT-SR vs ETT-SR',backbone),('EMT-ZR vs EMT',capacity)):
        lines.append(f'| {label} | {q["relative_change_percent"]["ADE"]:+.4f} | {q["relative_change_percent"]["FDE"]:+.4f} | {q["paired_wins"]["ADE"]} | {q["paired_wins"]["FDE"]} | {q.get("decision","descriptive")} |')
    lines+=['','Change = (treated / control − 1) × 100; negative favors treated. The threshold and paired-win count must hold for the same metric.','',social['rule']+'. '+gate['rule']+'.','', '## Fifteen paired validation runs','', '| Fold | Seed | ETT ADE/FDE | EMT ADE/FDE | EMT-ZR ADE/FDE | EMT-SR ADE/FDE | EMT-GSR ADE/FDE | ETT-SR ADE/FDE |','|---|---|---|---|---|---|---|---|']
    for p in pairs:lines.append('| '+' | '.join([p['fold'].upper(),str(p['seed'])]+[f'{p["metrics"][n]["ADE"]:.6f} / {p["metrics"][n]["FDE"]:.6f}' for n in ALL_MODELS])+' |')
    lines+=['','## Fold means and sample SD','', '| Fold | Model | ADE mean ± sample SD | FDE mean ± sample SD |','|---|---|---|---|']
    for f in SCENES:
        for n in ALL_MODELS:lines.append(f'| {f.upper()} | {NAMES[n]} | {folds[f][n]["ADE"]["mean"]:.6f} ± {folds[f][n]["ADE"]["sample_sd"]:.6f} | {folds[f][n]["FDE"]["mean"]:.6f} ± {folds[f][n]["FDE"]["sample_sd"]:.6f} |')
    lines+=['','## Neighbor audit','',f'Total social windows: {audit["total_social_samples"]}; N=8, no radius. Neighbors must occur at all eight observed frames in the same recording. Original target windows, labels, order and split manifests are unchanged.','',audit['neighbor_split_policy'],'', '| Fold | Split | Samples | Pedestrians | Mean neighbors | Median | p90 | Zero-neighbor ratio |','|---|---|---|---|---|---|---|---|']
    for f in SCENES:
        for split in ('train','val'):
            q=audit['folds'][f][split];lines.append(f'| {f.upper()} | {split} | {q["sample_count"]} | {q["pedestrian_count"]} | {q["mean_neighbor_count"]:.6f} | {q["median_neighbor_count"]:.1f} | {q["p90_neighbor_count"]:.1f} | {q["zero_neighbor_ratio"]:.6f} |')
    lines+=['','## Gate statistics at best validation checkpoint','', '| Fold | Seed | Mean | Population SD | p10 | p50 | p90 | Zero-neighbor mean | Has-neighbor mean |','|---|---|---|---|---|---|---|---|---|']
    for q in gate_runs:lines.append('| '+' | '.join([q['fold'].upper(),str(q['seed'])]+['n/a' if q[k] is None else f'{q[k]:.6f}' for k in ('gate_mean','gate_std','gate_p10','gate_p50','gate_p90','zero_neighbor_gate_mean','has_neighbor_gate_mean')])+' |')
    lines+=['','Scalar gate values describe the learned model; they do not independently establish interaction necessity or causal interpretation. Zero-neighbor validation groups have only 2–10 samples per fold.','', '## Neighbor-count group ADE/FDE','', 'Each fold is averaged over seeds, then folds are equally weighted. Per-run group sample counts and metrics are in comparison.json; groups with no samples use null, not fabricated zero errors.','', '| Model | Neighbor count | ADE (m) | FDE (m) |','|---|---|---|---|']
    for n in ALL_MODELS:
        for group in BUCKETS:
            q=group_aggregates[n][group]['five_fold_equal_weight_mean'];lines.append(f'| {NAMES[n]} | {group} | {q["ADE"]:.6f} | {q["FDE"]:.6f} |')
    lines+=['','## Runtime and trainable capacity','',f'Maximum pre-clip gradient norm: {comparison["maximum_gradient_norm_before_clipping"]:.6f}. NaN/Inf: false. First-stage result/checkpoint/manifest SHA256 inventory: unchanged.','', '| Model | Total params | Trainable params | Mean peak training allocated MiB | Full latency B=1,N=8 ms | Full latency B=128,N=8 ms |','|---|---|---|---|---|---|']
    for n in VARIANTS:
        rows=[q for q in runs if q['model']==n];q=rows[0]
        lines.append(f'| {NAMES[n]} | {q["parameter_count"]["total"]} | {q["parameter_count"]["trainable"]} | {statistics.mean(q["gpu_peak_memory_bytes"]/2**20 for q in rows):.3f} | {statistics.mean(q["latency"]["batch_1_neighbors_8_latency_ms"] for q in rows):.6f} | {statistics.mean(q["latency"]["batch_128_neighbors_8_latency_ms"] for q in rows):.6f} |')
    lines+=['','Training uses cached eval contexts to avoid repeatedly encoding frozen target/neighbor histories. Training memory includes resident source caches; it is not model-only memory. Precomputation timing is stored per fold/backbone/seed. Training wall times include parallel GPU contention. Latency measures the full uncached model, with 20 warmups and 100 synchronized eval/no_grad forwards, including the single shared temporal encoder for target and valid neighbors. Zero-neighbor and eight-neighbor cases, batch 1 and 128, are saved for every run.','', '## Reproduction and verification','', 'Use existing ped_intent environment. Run build_eth_ucy_social_sequences.py, pytest, then run_social_residual.py (four-model ETH/seed42 smoke gate before 60 runs). Raw data and checkpoints remain ignored. All output is in results/social_residual. Original stage-one files remain unchanged.','', 'Both social and gate decisions are internal validation findings. No held-out scene metrics are computed.']
    (RESULTS/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'average':equal,'social':{k:social[k] for k in ('relative_change_percent','paired_wins','decision')},'gate':{k:gate[k] for k in ('relative_change_percent','paired_wins','decision')}},indent=2))
if __name__=='__main__':main()
