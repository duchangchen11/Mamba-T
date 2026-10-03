"""Observation-only nearest neighbors; frozen stage-one target-track splits."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.eth_ucy_dataset import ROOT, SCENES, FILES
SOCIAL_PROCESSED=ROOT/'data/processed/eth_ucy_social'
BASE_RESULTS=ROOT/'results/eth_ucy_mamba_baseline'


def observed_motion(position):
    displacement=np.zeros_like(position)
    displacement[1:]=np.diff(position,axis=0)
    return np.concatenate((position-position[-1],displacement),axis=1)


def select_neighbors(frame_lookup, observation_frames, target_id, last_target):
    """Only these eight keys are accessed. No trajectory length/future filtering."""
    frames=[frame_lookup[int(f)] for f in observation_frames]
    candidates=set.intersection(*(set(f) for f in frames))-{target_id}
    ordered=sorted(candidates,key=lambda p:(float(np.linalg.norm(frames[-1][p]-last_target)),p))[:8]
    history=np.zeros((8,8,4),dtype=np.float32);relation=np.zeros((8,3),dtype=np.float32)
    mask=np.zeros(8,dtype=bool);ids=np.full(8,'',dtype='<U64');aligned=np.full((8,8),-1,dtype=np.int64)
    for i,ped in enumerate(ordered):
        position=np.stack([f[ped] for f in frames]).astype(np.float32)
        history[i]=observed_motion(position)
        xy=position[-1]-last_target;relation[i]=[xy[0],xy[1],np.linalg.norm(xy)]
        mask[i]=True;ids[i]=ped;aligned[i]=observation_frames
    return {'neighbor_history':history,'neighbor_mask':mask,'neighbor_relation':relation,'neighbor_ids':ids,'neighbor_frame_ids':aligned}


def recording_lookup(raw,filename):
    a=np.loadtxt(raw/filename);lookup={}
    for frame,ped,x,y in a:
        lookup.setdefault(int(frame),{})[f'{Path(filename).stem}:{int(ped)}']=np.array([x,y],dtype=np.float32)
    return lookup


def build_social_scene(scene):
    lookups={Path(f).stem:recording_lookup(ROOT/'data/raw/eth_ucy',f) for f in FILES[scene]}
    with np.load(ROOT/'data/processed/eth_ucy'/scene/'sequences.npz',allow_pickle=False) as z:
        target={k:z[k] for k in z.files}
    neighbors=[]
    for i,ped in enumerate(target['ped_id']):
        frames=target['frame_ids'][i,:8];lookup=lookups[str(ped).split(':')[0]]
        neighbors.append(select_neighbors(lookup,frames,str(ped),target['last_obs_pos'][i]))
    out={k:np.stack([n[k] for n in neighbors]) for k in neighbors[0]}
    out.update({'target_history':target['obs_input'],'future_target':target['future_target'],'last_obs_pos':target['last_obs_pos'],'future_abs':target['future_abs'],'frame_ids':target['frame_ids'],'ped_id':target['ped_id'],'scene_id':target['scene_id']})
    return out


class ETHUCYSocialDataset(Dataset):
    def __init__(self,heldout,split,processed=None):
        if heldout not in SCENES or split not in ('train','val'):raise ValueError('Only source train/val are allowed')
        manifest=json.loads((BASE_RESULTS/'data_audit'/f'manifest_{heldout}.json').read_text())
        if manifest['heldout_scene']!=heldout:raise ValueError('manifest mismatch')
        self.tracks=set();self.scenes=set();parts=[];self.track_ids=[]
        for scene in SCENES:
            if scene==heldout:continue
            with np.load(Path(processed or SOCIAL_PROCESSED)/scene/'sequences.npz',allow_pickle=False) as z:
                peds=z['ped_id'];allowed=set(manifest[split][scene]);indices=np.flatnonzero(np.isin(peds,list(allowed)))
                arrays={k:torch.from_numpy(z[k][indices].copy()) for k in ('target_history','neighbor_history','neighbor_mask','neighbor_relation','future_target','last_obs_pos','future_abs')}
                parts.append(arrays)
                self.track_ids.extend((scene,str(peds[i])) for i in indices)
                self.tracks.update((scene,str(peds[i])) for i in indices);self.scenes.add(scene)
        self.arrays={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
        if not self.track_ids:raise ValueError('empty split')
    def __len__(self):return len(self.track_ids)
    def __getitem__(self,i):return {k:a[i] for k,a in self.arrays.items()}


def neighbor_statistics(mask):
    a=np.asarray(mask.sum(-1));return {'sample_count':len(a),'mean_neighbor_count':float(a.mean()),'median_neighbor_count':float(np.median(a)),'p90_neighbor_count':float(np.quantile(a,.9)),'zero_neighbor_ratio':float((a==0).mean()),'count_histogram':{str(i):int((a==i).sum()) for i in range(9)}}
