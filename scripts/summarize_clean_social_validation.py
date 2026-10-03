from clean_social_utils import *
import statistics
from scripts.social_residual_utils import BUCKETS
MODELS=('emt_zr','old_emt_sr','emt_sr','ett_sr')
NAMES={'emt_zr':'Frozen EMT-ZR','old_emt_sr':'Old EMT-SR','emt_sr':'Clean EMT-SR','ett_sr':'Clean ETT-SR'}


def compare(folds,pairs,treated,control):
    means={n:{m:statistics.mean(folds[f][n][m]['mean'] for f in SCENES) for m in ('ADE','FDE')} for n in (treated,control)}
    diffs=[{'fold':p['fold'],'seed':p['seed'],**{f'delta_{m}':p['metrics'][treated][m]-p['metrics'][control][m] for m in ('ADE','FDE')}} for p in pairs]
    return {'treated':treated,'control':control,'equal_weight_average':means,'absolute_difference':{m:means[treated][m]-means[control][m] for m in ('ADE','FDE')},'relative_change_percent':{m:100*(means[treated][m]/means[control][m]-1) for m in ('ADE','FDE')},'paired_wins':{m:sum(q[f'delta_{m}']<0 for q in diffs) for m in ('ADE','FDE')},'paired_wins_both':sum(q['delta_ADE']<0 and q['delta_FDE']<0 for q in diffs),'paired_differences':diffs}


def main():
    pairs=[];folds={};runs=[];references=[];groups=[]
    frozen=json.loads((RESULTS/'data_audit/frozen_inventory.json').read_text())
    for fold in SCENES:
        values={n:[] for n in MODELS}
        for seed in (42,123,2024):
            metrics={}
            for n in MODELS:
                directory=OLD_RESULTS if n in ('emt_zr','old_emt_sr') else RESULTS
                folder=directory/f'heldout_{fold}'/('emt_sr' if n=='old_emt_sr' else n)/f'seed_{seed}'
                p=folder/'metrics_validation.json';q=json.loads(p.read_text());metrics[n]={m:q[f'validation_{m}'] for m in ('ADE','FDE')}
                assert not q['heldout_test_accessed'];values[n].append(metrics[n])
                groups.append({'fold':fold,'seed':seed,'model':n,'groups':q['neighbor_groups']})
                if n=='emt_zr':
                    checkpoint=ROOT/'checkpoints/social_residual'/f'heldout_{fold}'/n/f'seed_{seed}.pt'
                    msha=sha_file(p);csha=sha_file(checkpoint)
                    assert frozen[str(p.relative_to(ROOT))]==msha and frozen[str(checkpoint.relative_to(ROOT))]==csha
                    references.append({'fold':fold,'seed':seed,'metrics_path':str(p.relative_to(ROOT)),'metrics_sha256':msha,'checkpoint_path':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':csha,'retrained':False,'validation':metrics[n]})
                elif n in VARIANTS:
                    assert q['backbone_unchanged'] and not q['nan_inf'] and not q['smoke'] and q['protocol']==CONFIG
                    assert q['train_cross_split_neighbor_violations']==0 and q['initial_output_base_max_abs_diff']<1e-7
                    assert q['latency'] is not None;runs.append(q)
            pairs.append({'fold':fold,'seed':seed,'metrics':metrics})
        folds[fold]={n:{m:{'mean':statistics.mean(v[m] for v in values[n]),'sample_sd':statistics.stdev(v[m] for v in values[n])} for m in ('ADE','FDE')} for n in MODELS}
    assert len(runs)==30 and len(pairs)==15 and len(references)==15
    primary=compare(folds,pairs,'emt_sr','emt_zr');backbone=compare(folds,pairs,'emt_sr','ett_sr');old=compare(folds,pairs,'emt_sr','old_emt_sr')
    qualifying=[m for m,threshold in (('ADE',1.5),('FDE',2.5)) if primary['relative_change_percent'][m]<=-threshold and primary['paired_wins'][m]>=10]
    strong=primary['relative_change_percent']['ADE']<=-2 and primary['relative_change_percent']['FDE']<=-3 and primary['paired_wins_both']>=12
    decision='STRONG GO' if strong else 'GO' if qualifying else 'STOP'
    primary.update({'decision':decision,'qualifying_metrics':qualifying,'go_rule':'>=1.5% ADE or >=2.5% FDE improvement, with >=10/15 wins in the same metric','strong_go_rule':'>=2% ADE AND >=3% FDE improvement, with >=12/15 paired runs improving both metrics'})
    means={n:{m:statistics.mean(folds[f][n][m]['mean'] for f in SCENES) for m in ('ADE','FDE')} for n in MODELS}
    gain={}
    for m in ('ADE','FDE'):
        original=100*(1-means['old_emt_sr'][m]/means['emt_zr'][m]);clean=100*(1-means['emt_sr'][m]/means['emt_zr'][m])
        gain[m]={'old_improvement_percent':original,'clean_improvement_percent':clean,'improvement_reduction_percentage_points':original-clean,'retained_improvement_percent':100*clean/original if original else None}
    aggregate={}
    for n in MODELS:
        aggregate[n]={}
        for group in BUCKETS:
            byfold={}
            for fold in SCENES:
                rows=[q['groups'][group] for q in groups if q['fold']==fold and q['model']==n]
                assert len(rows)==3 and len(set(q['sample_count'] for q in rows))==1
                count=rows[0]['sample_count'];byfold[fold]={'sample_count_per_seed':count,**{m:statistics.mean(q[m] for q in rows) if count else None for m in ('ADE','FDE')}}
            present=[q for q in byfold.values() if q['sample_count_per_seed']]
            aggregate[n][group]={'folds':byfold,'total_validation_samples_per_seed':sum(q['sample_count_per_seed'] for q in byfold.values()),'participating_fold_count':len(present),'equal_weight_mean':{m:statistics.mean(q[m] for q in present) if present else None for m in ('ADE','FDE')}}
    split_audit=json.loads((RESULTS/'data_audit/neighbor_split_audit.json').read_text())
    neighbors=json.loads((RESULTS/'data_audit/neighbor_statistics.json').read_text());distances=json.loads((RESULTS/'data_audit/distance_statistics.json').read_text())
    assert split_audit['train_cross_split_neighbor_violations']==0
    count=verify_frozen()
    output={'scope':'source-scene validation only; strict train neighbor membership; validation inputs identical to old','folds':folds,'paired_runs':pairs,'equal_weight_five_fold_average':means,'primary_comparison':primary,'clean_backbone_comparison':backbone,'clean_vs_old_comparison':old,'social_gain_retention':gain,'emt_zr_references':references,'neighbor_group_metrics':groups,'neighbor_group_aggregates':aggregate,'neighbor_split_audit':split_audit,'neighbor_statistics':neighbors,'distance_statistics':distances,'maximum_gradient_norm_before_clipping':max(q['maximum_gradient_norm_before_clipping'] for q in runs),'nan_inf':False,'frozen_files_verified':count,'model_architecture_changed':False,'training_hyperparameters_changed':False,'heldout_test_accessed':False,'decision':decision,'runs':runs}
    dump(RESULTS/'comparison.json',output);dump(RESULTS/'data_audit/emt_zr_references.json',references)
    lines=['# Clean Social Validation on ETH/UCY','',f'**CLEAN SOCIAL: {decision}**. Source-scene validation only; heldout_test_accessed=false.','', 'Train neighbors must belong to the same scene’s frozen train target-pedestrian set, the same recording, and be visible in all eight observed frames. Selection excludes the target and ranks by last-observation distance. N=8, no radius. Validation uses all observation-visible neighbors and is identical to the previous stage. Audit-only split labels do not enter the model.','', '| Model | Five-fold equal-weight ADE (m) | FDE (m) |','|---|---|---|']
    for n in MODELS:lines.append(f'| {NAMES[n]} | {means[n]["ADE"]:.6f} | {means[n]["FDE"]:.6f} |')
    lines+=['','## Core comparisons','', '| Comparison (treated minus control) | ΔADE (m) | ΔFDE (m) | ADE change % | FDE change % | ADE wins /15 | FDE wins /15 | Both wins /15 |','|---|---|---|---|---|---|---|---|']
    for label,q in (('Clean EMT-SR vs Frozen EMT-ZR',primary),('Clean EMT-SR vs Clean ETT-SR',backbone),('Clean EMT-SR vs Old EMT-SR',old)):
        lines.append(f'| {label} | {q["absolute_difference"]["ADE"]:+.6f} | {q["absolute_difference"]["FDE"]:+.6f} | {q["relative_change_percent"]["ADE"]:+.4f} | {q["relative_change_percent"]["FDE"]:+.4f} | {q["paired_wins"]["ADE"]} | {q["paired_wins"]["FDE"]} | {q["paired_wins_both"]} |')
    lines+=['',primary['go_rule']+'. '+primary['strong_go_rule']+'. Negative changes favor Clean EMT-SR.','', '## Old social gain vs clean gain','', '| Metric | Old improvement vs ZR % | Clean improvement vs ZR % | Gain reduction (percentage points) |','|---|---|---|---|']
    for m,q in gain.items():lines.append(f'| {m} | {q["old_improvement_percent"]:.6f} | {q["clean_improvement_percent"]:.6f} | {q["improvement_reduction_percentage_points"]:+.6f} |')
    lines+=['','## Fifteen paired validation runs','', '| Fold | Seed | Frozen EMT-ZR ADE/FDE | Old EMT-SR ADE/FDE | Clean EMT-SR ADE/FDE | Clean ETT-SR ADE/FDE |','|---|---|---|---|---|---|']
    for p in pairs:lines.append('| '+' | '.join([p['fold'].upper(),str(p['seed'])]+[f'{p["metrics"][n]["ADE"]:.6f} / {p["metrics"][n]["FDE"]:.6f}' for n in MODELS])+' |')
    lines+=['','## Fold mean ± sample SD','', '| Fold | Model | ADE (m) | FDE (m) |','|---|---|---|---|']
    for f in SCENES:
        for n in MODELS:lines.append(f'| {f.upper()} | {NAMES[n]} | {folds[f][n]["ADE"]["mean"]:.6f} ± {folds[f][n]["ADE"]["sample_sd"]:.6f} | {folds[f][n]["FDE"]["mean"]:.6f} ± {folds[f][n]["FDE"]["sample_sd"]:.6f} |')
    lines+=['','## Neighbor counts and split audit','', 'Training cross-split neighbor violations: **0**. Every effective training neighbor belongs to the frozen train target set. Validation neighbor train/val/other proportions are audit-only.','', '| Fold | Split | Samples | Mean | Median | p90 | Zero ratio | Old mean | 0 ratio | 1–2 ratio | 3–4 ratio | 5+ ratio |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for f in SCENES:
        for split in ('train','val'):
            q=neighbors['folds'][f][split];parts=[f.upper(),split,str(q['sample_count'])]+[f'{q[k]:.6f}' for k in ('mean_neighbor_count','median_neighbor_count','p90_neighbor_count','zero_neighbor_ratio','old_mean_neighbor_count')]+[f'{q["group_proportions"][g]:.6f}' for g in BUCKETS]
            lines.append('| '+' | '.join(parts)+' |')
    lines+=['','| Validation fold | Train-side neighbor ratio | Val-side ratio | Other/non-target-eligible ratio |','|---|---|---|---|']
    for f in SCENES:
        q=split_audit['folds'][f]['val']['membership_ratios'];lines.append('| '+' | '.join([f.upper()]+[f'{q[k]:.6f}' for k in ('train','val','other')])+' |')
    lines+=['','## Neighbor distance audit (meters)','', '| Fold | Split | Effective neighbors | Mean | Median | p25 | p75 | p90 | Max |','|---|---|---|---|---|---|---|---|---|']
    for f in SCENES:
        for split in ('train','val'):
            q=distances['folds'][f][split];lines.append('| '+' | '.join([f.upper(),split,str(q['valid_neighbor_count'])]+[f'{q[k]:.6f}' if q[k] is not None else 'n/a' for k in ('mean','median','p25','p75','p90','max')])+' |')
    lines+=['','| Fold | Split | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Rank 6 | Rank 7 | Rank 8 |','|---|---|---|---|---|---|---|---|---|---|']
    for f in SCENES:
        for split in ('train','val'):
            q=distances['folds'][f][split]['rank_mean_distance'];lines.append('| '+' | '.join([f.upper(),split]+['n/a' if q[f'rank{i}'] is None else f'{q[f"rank{i}"]:.6f}' for i in range(1,9)])+' |')
    lines+=['','Distances are observations at the last observed frame, ranked per sample. No distance threshold is applied or tuned.','', '## Validation neighbor-count stratification','', 'Validation groups are identical for all controls. Average each fold over seeds, then equally weight folds with observations. Empty groups use null. Per-fold/per-seed sample counts are in comparison.json.','', '| Model | Neighbor group | Samples per seed across five validation folds | ADE (m) | FDE (m) |','|---|---|---|---|---|']
    for n in MODELS:
        for group,q in aggregate[n].items():
            avg=q['equal_weight_mean'];lines.append(f'| {NAMES[n]} | {group} | {q["total_validation_samples_per_seed"]} | {avg["ADE"]:.6f} | {avg["FDE"]:.6f} |')
    lines+=['','## Frozen models, protocol and runtime','', f'All {count} frozen files unchanged, including both old results directories, checkpoints, old social data, manifests and model source. No gate experiments. EMT-ZR is reused with metrics and checkpoint SHA256 in data_audit/emt_zr_references.json. All new initial outputs equal their frozen base (max_abs_diff <1e-7); all backbone state SHA256 values match after training.','',f'Maximum pre-clip gradient norm: {output["maximum_gradient_norm_before_clipping"]:.6f}. NaN/Inf: false.','', 'The model and hyperparameters match stage two: 50 epochs, batch 128, AdamW 1e-3/1e-4, SmoothL1 beta=1, clip 5, ReduceLROnPlateau factor .5/patience 4, early stopping 8, checkpoint minimum source validation ADE. Only train-neighbor eligibility changes.','', '| Model | Total params | Trainable | Mean training peak allocated MiB | Mean uncached latency B=1,N=8 ms | B=128,N=8 ms |','|---|---|---|---|---|---|']
    for n in VARIANTS:
        rows=[q for q in runs if q['model']==n];q=rows[0]
        lines.append(f'| {NAMES[n]} | {q["parameter_count"]["total"]} | {q["parameter_count"]["trainable"]} | {statistics.mean(q["gpu_peak_memory_bytes"]/2**20 for q in rows):.3f} | {statistics.mean(q["latency"]["batch_1_neighbors_8_latency_ms"] for q in rows):.6f} | {statistics.mean(q["latency"]["batch_128_neighbors_8_latency_ms"] for q in rows):.6f} |')
    lines+=['','Training uses frozen eval context caches; memory includes resident caches, and wall times include parallel GPU contention. Latency is measured after training, on the complete uncached model, including shared temporal encoding; 20 warmups +100 synchronized eval/no_grad forwards; batch 1/128, zero/eight neighbors. No data transfer.','', 'No held-out metrics, backbone unfreezing, radius/N/layer/lr/loss changes, gate, or other new models are initiated.']
    verification=RESULTS/'data_audit/verification.json'
    if verification.exists():
        q=json.loads(verification.read_text());lines+=['',f'Verification: {q["pytest_passed"]} pytest tests passed; py_compile passed. Full 20-item Chinese handoff: brain_report.md.']
    (RESULTS/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'average':means,'primary':{k:primary[k] for k in ('relative_change_percent','paired_wins','paired_wins_both','decision')},'backbone':{k:backbone[k] for k in ('relative_change_percent','paired_wins')},'clean_vs_old':{k:old[k] for k in ('absolute_difference','relative_change_percent')},'gain_retention':gain},indent=2))
if __name__=='__main__':main()
