import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv, json, hashlib, shutil
import numpy as np
from src.data.eth_ucy_dataset import *

def main():
    raw=ROOT/'data/raw/eth_ucy';processed=ROOT/'data/processed/eth_ucy'
    audit=ROOT/'results/eth_ucy_mamba_baseline/data_audit';audit.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((raw/'dataset_manifest.json').read_text())
    provenance.update({'coordinates':'world x/y, meters, as distributed; no transform','sampling_hz':2.5,'sample_interval_seconds':.4,'preprocessing':'local contiguous pedestrian windows, stride 1, 8+12; no interpolation; recording-qualified pedestrian IDs','acquired_local_date':'2026-10-03'})
    text='# ETH/UCY data provenance\n\n'+json.dumps(provenance,indent=2)+'\n'
    (raw/'DATA_PROVENANCE.md').write_text(text);(audit/'DATA_PROVENANCE.md').write_text(text)
    (audit/'data_provenance.json').write_text(json.dumps(provenance,indent=2))
    scenes={};tracks={}
    for scene in SCENES:
        z,canonical,stats=build_scene(raw,scene)
        p=processed/scene;p.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p/'sequences.npz',**z)
        with (p/'canonical.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['scene','ped_id','frame_id','x','y']);writer.writerows(sorted(canonical,key=lambda r:(r[0],r[1],r[2])))
        tracks[scene]=sorted(set(z['ped_id'].tolist()))
        scenes[scene]={'raw_pedestrians':sum(s['pedestrians'] for s in stats),'eligible_pedestrians':len(tracks[scene]),'windows':len(z['ped_id']),'files':stats}
    m=processed/'manifests';m.mkdir(exist_ok=True)
    folds={}
    for heldout in SCENES:
        manifest={'heldout_scene':heldout,'heldout_test_accessed':False,'split_rule':'sha1(scene/ped_id) / 2**160 < 0.15 => validation','train':{},'val':{},'heldout_tracks':tracks[heldout]}
        for scene in SCENES:
            if scene==heldout: continue
            for split in ('train','val'):manifest[split][scene]=[p for p in tracks[scene] if track_split(scene,p)==split]
        payload=json.dumps(manifest,indent=2);(m/f'{heldout}.json').write_text(payload);(audit/f'manifest_{heldout}.json').write_text(payload)
        tr=ETHUCYDataset(heldout,'train');va=ETHUCYDataset(heldout,'val')
        assert not tr.tracks & va.tracks and heldout not in tr.scenes|va.scenes
        folds[heldout]={'train_pedestrians':len(tr.tracks),'validation_pedestrians':len(va.tracks),'train_windows':len(tr),'validation_windows':len(va)}
    (audit/'counts.json').write_text(json.dumps({'scenes':scenes,'folds':folds},indent=2))
    print(json.dumps({'scenes':scenes,'folds':folds},indent=2))
if __name__=='__main__':main()
