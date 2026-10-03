"""Target-only, fixed track split, contiguous 2.5 Hz ETH/UCY windows."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
SCENES = ('eth','hotel','univ','zara1','zara2')
FILES = {'eth':['biwi_eth.txt'], 'hotel':['biwi_hotel.txt'], 'univ':['students001.txt','students003.txt'], 'zara1':['crowds_zara01.txt'], 'zara2':['crowds_zara02.txt']}
ROOT = Path(__file__).resolve().parents[2]

def track_split(scene, ped_id):
    return 'val' if int(hashlib.sha1(f"{scene}/{ped_id}".encode()).hexdigest(),16) / 2**160 < .15 else 'train'

def features(obs, future):
    last = obs[-1].copy()
    displacement = np.zeros_like(obs)
    displacement[1:] = np.diff(obs,axis=0)
    return np.concatenate((obs-last,displacement),axis=1), future-last, last

def build_scene(raw, scene):
    samples=[]; canonical=[]; stats=[]
    for filename in FILES[scene]:
        a=np.loadtxt(raw/filename)
        if a.ndim!=2 or a.shape[1]!=4 or not np.isfinite(a).all(): raise ValueError(filename)
        if not np.equal(a[:,:2],np.floor(a[:,:2])).all(): raise ValueError('noninteger IDs')
        if len(np.unique(a[:,:2],axis=0))!=len(a): raise ValueError('duplicate frame/pedestrian')
        frames=np.unique(a[:,0]).astype(np.int64)
        diffs=np.diff(frames); step=int(np.gcd.reduce(diffs))
        if step<=0: raise ValueError('invalid frame stride')
        stats.append({'file':filename,'frame_stride':step,'rows':len(a),'pedestrians':len(np.unique(a[:,1]))})
        for ped in np.unique(a[:,1]):
            t=a[a[:,1]==ped];t=t[np.argsort(t[:,0])]
            # Recording-qualified IDs also qualify canonical frames to avoid video collisions.
            pid=f"{Path(filename).stem}:{int(ped)}"
            canonical.extend((scene,pid,int(r[0]),float(r[2]),float(r[3])) for r in t)
            for start in range(len(t)-19):
                w=t[start:start+20]
                if not np.all(np.diff(w[:,0])==step): continue
                pos=w[:,2:].astype(np.float32);obs=pos[:8];future=pos[8:]
                x,y,last=features(obs,future)
                samples.append((obs,future,x,y,last,pid,w[:,0].astype(np.int64)))
    if not samples: raise ValueError(f'no windows: {scene}')
    names=['obs_abs','future_abs','obs_input','future_target','last_obs_pos','ped_id','frame_ids']
    out={k:np.stack([s[i] for s in samples]) for i,k in enumerate(names)}
    out['scene_id']=np.full(len(samples),scene)
    return out,canonical,stats

class ETHUCYDataset(Dataset):
    def __init__(self, heldout, split, processed=None):
        if heldout not in SCENES or split not in ('train','val'): raise ValueError('Training only accepts source train/val')
        processed=Path(processed or ROOT/'data/processed/eth_ucy')
        manifest=__import__('json').loads((processed/'manifests'/f'{heldout}.json').read_text())
        self.items=[];self.tracks=set();self.scenes=set()
        for scene in SCENES:
            if scene==heldout: continue  # Never open held-out NPZ.
            with np.load(processed/scene/'sequences.npz',allow_pickle=False) as z:
                arrays={k:z[k] for k in ('obs_input','future_target','last_obs_pos','future_abs','ped_id')}
                allowed=set(manifest[split][scene])
                for i,ped in enumerate(arrays['ped_id']):
                    if str(ped) not in allowed: continue
                    self.items.append({k:torch.from_numpy(arrays[k][i].copy()) for k in ('obs_input','future_target','last_obs_pos','future_abs')})
                    self.tracks.add((scene,str(ped))); self.scenes.add(scene)
        if not self.items: raise ValueError('empty split')
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]
