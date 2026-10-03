from eth_ucy_utils import *
import statistics, subprocess

def main():
    pairs=[];folds={};runs=[]
    for fold in SCENES:
        fold_runs={n:[] for n in ('ett','emt')}
        for seed in (42,123,2024):
            r={n:json.loads((RESULTS/f'heldout_{fold}'/n/f'seed_{seed}/metrics_validation.json').read_text()) for n in ('ett','emt')}
            assert r['ett']['decoder_initialization_sha256']==r['emt']['decoder_initialization_sha256']
            for n in r:
                assert not r[n]['heldout_test_accessed'] and not r[n]['nan_inf']
                fold_runs[n].append(r[n]);runs.append(r[n])
            pairs.append({'fold':fold,'seed':seed,'ETT_ADE':r['ett']['validation_ADE'],'EMT_ADE':r['emt']['validation_ADE'],'delta_ADE':r['emt']['validation_ADE']-r['ett']['validation_ADE'],'ETT_FDE':r['ett']['validation_FDE'],'EMT_FDE':r['emt']['validation_FDE'],'delta_FDE':r['emt']['validation_FDE']-r['ett']['validation_FDE']})
        folds[fold]={}
        for metric in ('ADE','FDE'):
            for n in fold_runs:
                vals=[r[f'validation_{metric}'] for r in fold_runs[n]]
                folds[fold][f'{n}_{metric}']={'mean':statistics.mean(vals),'sample_sd':statistics.stdev(vals)}
            a=folds[fold][f'ett_{metric}']['mean'];b=folds[fold][f'emt_{metric}']['mean']
            folds[fold][f'delta_{metric}']=b-a
            folds[fold][f'relative_change_{metric}_percent']=100*(b/a-1)
    avg={f'{n}_{m}':statistics.mean(folds[f][f'{n}_{m}']['mean'] for f in SCENES) for n in ('ett','emt') for m in ('ADE','FDE')}
    relative={m:100*(avg[f'emt_{m}']/avg[f'ett_{m}']-1) for m in ('ADE','FDE')}
    wins={m:sum(p[f'delta_{m}']<0 for p in pairs) for m in ('ADE','FDE')}
    # Tie criterion B to the SAME qualifying metric as A; also report joint wins.
    qualifying=[m for m,threshold in (('ADE',2),('FDE',3)) if relative[m]<=-threshold and wins[m]>=10]
    c=all(folds[f][f'relative_change_{m}_percent']<=10 for f in SCENES for m in ('ADE','FDE'))
    decision='GO' if qualifying and c else 'STOP'
    comparison={'scope':'internal source-validation only; no held-out performance claim','folds':folds,'paired_comparisons':pairs,'equal_weight_five_fold_average':avg,'relative_change_percent':relative,'emt_wins':wins,'emt_wins_both':sum(p['delta_ADE']<0 and p['delta_FDE']<0 for p in pairs),'go_rule':'A and B must hold for the same metric (ADE or FDE); C for both metrics in every fold','qualifying_metrics':qualifying,'condition_C':c,'decision':decision,'heldout_test_accessed':False,'maximum_preclip_gradient':max(r['maximum_gradient_norm_before_clipping'] for r in runs),'nan_inf':False,'runs':runs}
    dump(RESULTS/'comparison.json',comparison)
    lines=['# ETH/UCY matched baseline validation','',f'Internal decision: **{decision}**. Source-scene validation only; held-out test accessed: **false**.','', 'Five-fold equal-weight means:','']
    lines += [f'- {n.upper()}: ADE {avg[f"{n}_ADE"]:.6f} m; FDE {avg[f"{n}_FDE"]:.6f} m.' for n in ('ett','emt')]
    lines += [f'- EMT relative changes: ADE {relative["ADE"]:+.3f}%; FDE {relative["FDE"]:+.3f}%.',f'- EMT paired wins: ADE {wins["ADE"]}/15; FDE {wins["FDE"]}/15; both {comparison["emt_wins_both"]}/15.','', '| Fold | ETT ADE | EMT ADE | ΔADE | ETT FDE | EMT FDE | ΔFDE |','|---|---|---|---|---|---|---|']
    for f in SCENES:
        q=folds[f];fmt=lambda n,m:f'{q[f"{n}_{m}"]["mean"]:.5f} ± {q[f"{n}_{m}"]["sample_sd"]:.5f}'
        lines.append(f'| {f.upper()} | {fmt("ett","ADE")} | {fmt("emt","ADE")} | {q["delta_ADE"]:+.5f} | {fmt("ett","FDE")} | {fmt("emt","FDE")} | {q["delta_FDE"]:+.5f} |')
    lines += ['','Mean ± sample SD across three seeds; Δ = EMT − ETT. Negative favors EMT.','', '| Fold | Seed | ETT ADE | EMT ADE | ΔADE | ETT FDE | EMT FDE | ΔFDE |','|---|---|---|---|---|---|---|---|']
    for p in pairs:lines.append('| '+ ' | '.join([p['fold'],str(p['seed'])]+[f'{p[k]:.6f}' for k in ('ETT_ADE','EMT_ADE','delta_ADE','ETT_FDE','EMT_FDE','delta_FDE')])+' |')
    counts=json.loads((RESULTS/'data_audit/counts.json').read_text())
    lines+=['','## Data audit','', '| Scene | Raw pedestrians | Eligible pedestrians | Windows |','|---|---|---|---|']
    for f,q in counts['scenes'].items():lines.append(f'| {f} | {q["raw_pedestrians"]} | {q["eligible_pedestrians"]} | {q["windows"]} |')
    lines+=['','| Fold | Train pedestrians | Validation pedestrians | Train windows | Validation windows |','|---|---|---|---|---|']
    for f,q in counts['folds'].items():lines.append('| '+' | '.join([f]+[str(q[k]) for k in ('train_pedestrians','validation_pedestrians','train_windows','validation_windows')])+' |')
    lines+=['','## Runtime','',f'Maximum pre-clip gradient norm: {comparison["maximum_preclip_gradient"]:.6f}. NaN/Inf: false.','', '| Fold | Model | Seed | Parameters | Best epoch | Seconds | Peak allocated MiB | Single latency ms | Batch 128 latency ms |','|---|---|---|---|---|---|---|---|---|']
    for r in runs:lines.append(f'| {r["heldout_scene"]} | {r["model"]} | {r["seed"]} | {r["parameter_count"]} | {r["best_epoch"]} | {r["training_seconds"]:.2f} | {r["gpu_peak_memory_bytes"]/2**20:.2f} | {r["latency"]["batch_1_latency_ms"]:.3f} | {r["latency"]["batch_128_latency_ms"]:.3f} |')
    lines+=['','Latency uses 20 warmup forwards and 100 synchronized eval forwards; GPU memory is peak allocated memory during training/validation.','', '## Reproduction','', 'Environment: Python 3.10.21; Torch 2.5.1+cu124; CUDA runtime 12.4; NVIDIA GeForce RTX 3080; mamba-ssm 2.2.6.post3.','', 'Data source: [Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN/tree/333d3a57b4d2705e129b21aefefa09c79b2b9ae1/datasets/raw/all_data). SHA256 and exact URLs: `data_audit/data_provenance.json`. Raw provenance: `/home/lrj/Mamba-T/data/raw/eth_ucy/DATA_PROVENANCE.md`.','', 'UNIV uses students001 and students003 only. Pedestrian IDs include the recording name. All eligible contiguous track windows use stride 1. No interpolation or future-derived input. Hash threshold gives approximately 85/15 tracks, not an exact count quota.','', 'GO requires >=2% ADE improvement and >=10 ADE paired wins, or >=3% FDE improvement and >=10 FDE paired wins; neither metric may worsen >10% on any fold. Validation selection/decision is exploratory and does not establish held-out benchmark superiority.','', 'No hyperparameter search or stage 2 is initiated.']
    (RESULTS/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:comparison[k] for k in ('equal_weight_five_fold_average','relative_change_percent','emt_wins','decision')},indent=2))
if __name__=='__main__':main()
