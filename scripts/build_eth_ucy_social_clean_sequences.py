import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib
import numpy as np
from src.data.eth_ucy_social_clean_dataset import *
from src.data.eth_ucy_social_dataset import ETHUCYSocialDataset,neighbor_statistics
AUDIT=ROOT/'results/clean_social_validation/data_audit'


def distance_statistics(relation,mask):
    distance=relation[:,:,2];values=distance[mask]
    stats={k:None for k in ('mean','median','p25','p75','p90','max')}
    if len(values):stats.update({'mean':float(values.mean()),'median':float(np.median(values)),'p25':float(np.quantile(values,.25)),'p75':float(np.quantile(values,.75)),'p90':float(np.quantile(values,.9)),'max':float(values.max())})
    stats['units']='meter';stats['valid_neighbor_count']=len(values)
    stats['rank_mean_distance']={f'rank{i+1}':float(distance[:,i][mask[:,i]].mean()) if mask[:,i].any() else None for i in range(8)}
    stats['rank_sample_count']={f'rank{i+1}':int(mask[:,i].sum()) for i in range(8)}
    return stats


def main():
    AUDIT.mkdir(parents=True,exist_ok=True);audits={};counts={};distances={}
    # Each fold accesses only its four source scenes.
    for fold in SCENES:
        manifest=load_manifest(fold);audits[fold]={};counts[fold]={};distances[fold]={}
        for scene in SCENES:
            if scene==fold:continue
            lookups={Path(f).stem:recording_lookup(ROOT/'data/raw/eth_ucy',f) for f in FILES[scene]}
            for split in ('train','val'):
                z=build_clean_scene(scene,split,manifest,lookups);p=CLEAN_PROCESSED/fold/scene;p.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(p/f'{split}.npz',**z)
        for split in ('train','val'):
            d=ETHUCYSocialCleanDataset(fold,split);old=ETHUCYSocialDataset(fold,split)
            assert d.track_ids==old.track_ids
            for k in ('target_history','future_target','future_abs','last_obs_pos'):np.testing.assert_array_equal(d.arrays[k].numpy(),old.arrays[k].numpy())
            if split=='val':
                for k in FIELDS:np.testing.assert_array_equal(d.arrays[k].numpy(),old.arrays[k].numpy())
            mask=d.arrays['neighbor_mask'].numpy();ids=d.audit['neighbor_ids'];labels=d.audit['neighbor_split'];violation_count=0
            for i,scene in enumerate(d.audit['scene_id']):
                valid=ids[i][mask[i]]
                if split=='train':
                    tr=set(manifest['train'][str(scene)]);va=set(manifest['val'][str(scene)])
                    violation_count+=sum(p not in tr or p in va for p in valid)
            valid_count=int(mask.sum());membership={s:int((labels[mask]==s).sum()) for s in ('train','val','other')}
            if split=='train':assert violation_count==0 and membership['train']==valid_count
            audits[fold][split]={'target_sample_count':len(d),'valid_neighbor_count':valid_count,'membership_counts':membership,'membership_ratios':{s:n/valid_count if valid_count else None for s,n in membership.items()},'cross_split_neighbor_violations':violation_count if split=='train' else None,'all_targets_match_frozen_split':True,'validation_features_identical_to_old':split=='val'}
            stats=neighbor_statistics(mask);stats['pedestrian_count']=len(d.tracks);n=mask.sum(1)
            stats['group_proportions']={'0':float((n==0).mean()),'1-2':float(((n>=1)&(n<=2)).mean()),'3-4':float(((n>=3)&(n<=4)).mean()),'5+':float((n>=5).mean())}
            old_mean=float(old.arrays['neighbor_mask'].sum(1).float().mean());stats.update({'old_mean_neighbor_count':old_mean,'mean_neighbor_count_change':stats['mean_neighbor_count']-old_mean})
            counts[fold][split]=stats;distances[fold][split]=distance_statistics(d.arrays['neighbor_relation'].numpy(),mask)
        assert not ETHUCYSocialCleanDataset(fold,'train').tracks&ETHUCYSocialCleanDataset(fold,'val').tracks
    (AUDIT/'neighbor_split_audit.json').write_text(json.dumps({'folds':audits,'train_cross_split_neighbor_violations':0,'train_all_valid_neighbors_belong_to_train':True,'validation_policy':'all currently observation-visible pedestrians, no future filtering','heldout_test_accessed':False},indent=2))
    (AUDIT/'neighbor_statistics.json').write_text(json.dumps({'folds':counts,'N':8,'radius':None,'target_split_changed':False,'validation_neighbors_changed':False,'heldout_test_accessed':False},indent=2))
    (AUDIT/'distance_statistics.json').write_text(json.dumps({'folds':distances,'policy':'audit only; no radius or neighbor changes based on distance','units':'meter'},indent=2))
    provenance=json.loads((ROOT/'results/social_residual/data_audit/data_provenance.json').read_text());provenance['train_neighbor_policy']='same recording, eight observed frames, not target, member of frozen scene train target set';provenance['validation_neighbor_policy']='all observation-visible neighbors, identical to old validation inputs'
    (AUDIT/'data_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({'cross_split_train_neighbor_violations':0,'folds':counts},indent=2))
if __name__=='__main__':main()
