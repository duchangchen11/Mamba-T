"""Fold-specific train-only neighbors; unchanged observation-visible validation."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.eth_ucy_social_dataset import ROOT,SCENES,FILES,BASE_RESULTS,recording_lookup,select_neighbors
CLEAN_PROCESSED=ROOT/'data/processed/eth_ucy_social_clean'
FIELDS=('target_history','neighbor_history','neighbor_mask','neighbor_relation','future_target','last_obs_pos','future_abs')
AUDIT_FIELDS=('target_ped_id','neighbor_ids','target_split','neighbor_split','frame_ids','neighbor_frame_ids','scene_id')


def load_manifest(fold):
    if fold not in SCENES:raise ValueError('unknown fold')
    manifest=json.loads((BASE_RESULTS/'data_audit'/f'manifest_{fold}.json').read_text())
    if manifest['heldout_scene']!=fold:raise ValueError('heldout mismatch')
    return manifest


def select_clean_neighbors(lookup,obs_frames,target_id,last_target,split,train_ids,val_ids):
    if split not in ('train','val'):raise ValueError('Only train/val')
    # This is the only lookup access: eight current observation frames.
    recording=target_id.split(':')[0]
    observed={int(f):{p:xy for p,xy in lookup[int(f)].items() if p.split(':')[0]==recording and (split=='val' or p in train_ids)} for f in obs_frames}
    output=select_neighbors(observed,obs_frames,target_id,last_target)
    labels=np.full(8,'padding',dtype='<U8')
    for i,p in enumerate(output['neighbor_ids']):
        if not output['neighbor_mask'][i]:continue
        labels[i]='train' if p in train_ids else 'val' if p in val_ids else 'other'
    output['neighbor_split']=labels
    if split=='train' and not np.all(labels[output['neighbor_mask']]=='train'):raise ValueError('train neighbor contamination')
    return output


def build_clean_scene(scene,split,manifest,lookups=None):
    if scene==manifest['heldout_scene']:raise ValueError('Heldout scene cannot be built as a source')
    train=set(manifest['train'][scene]);val=set(manifest['val'][scene]);allowed=train if split=='train' else val
    if lookups is None:lookups={Path(f).stem:recording_lookup(ROOT/'data/raw/eth_ucy',f) for f in FILES[scene]}
    with np.load(ROOT/'data/processed/eth_ucy'/scene/'sequences.npz',allow_pickle=False) as z:
        ids=z['ped_id'];idx=np.flatnonzero(np.isin(ids,list(allowed)))
        a={k:z[k][idx] for k in z.files}
    neighbors=[]
    for i,p in enumerate(a['ped_id']):
        neighbors.append(select_clean_neighbors(lookups[str(p).split(':')[0]],a['frame_ids'][i,:8],str(p),a['last_obs_pos'][i],split,train,val))
    if not neighbors:raise ValueError(f'empty {scene}/{split}')
    output={k:np.stack([n[k] for n in neighbors]) for k in neighbors[0]}
    output.update({'target_history':a['obs_input'],'future_target':a['future_target'],'future_abs':a['future_abs'],'last_obs_pos':a['last_obs_pos'],'target_ped_id':a['ped_id'],'target_split':np.full(len(idx),split),'scene_id':a['scene_id'],'frame_ids':a['frame_ids']})
    return output


class ETHUCYSocialCleanDataset(Dataset):
    def __init__(self,heldout,split,processed=None):
        if heldout not in SCENES or split not in ('train','val'):raise ValueError('Only source train/val')
        manifest=load_manifest(heldout);parts=[];meta=[];self.tracks=set();self.scenes=set();self.track_ids=[]
        for scene in SCENES:
            if scene==heldout:continue
            path=Path(processed or CLEAN_PROCESSED)/heldout/scene/f'{split}.npz'
            with np.load(path,allow_pickle=False) as z:
                allowed=set(manifest[split][scene]);peds=z['target_ped_id'];mask=z['neighbor_mask'];ids=z['neighbor_ids']
                if not set(peds.tolist())<=allowed or not np.all(z['target_split']==split):raise ValueError('target split mismatch')
                if not np.all(z['scene_id']==scene):raise ValueError('scene mismatch')
                if split=='train':
                    train=set(manifest['train'][scene]);val=set(manifest['val'][scene]);neighbors=set(ids[mask].tolist())
                    if not neighbors<=train or neighbors&val:raise ValueError('cross-split training neighbor')
                    if not np.all(z['neighbor_split'][mask]=='train'):raise ValueError('invalid neighbor audit label')
                parts.append({k:torch.from_numpy(z[k].copy()) for k in FIELDS})
                meta.append({k:z[k] for k in AUDIT_FIELDS})
                self.track_ids.extend((scene,str(p)) for p in peds)
                self.tracks.update((scene,str(p)) for p in peds);self.scenes.add(scene)
        self.arrays={k:torch.cat([p[k] for p in parts]) for k in FIELDS}
        self.audit={k:np.concatenate([p[k] for p in meta]) for k in AUDIT_FIELDS}
        if not self.track_ids:raise ValueError('empty dataset')
    def __len__(self):return len(self.track_ids)
    def __getitem__(self,i):return {k:v[i] for k,v in self.arrays.items()}
