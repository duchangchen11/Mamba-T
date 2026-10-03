import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json
import numpy as np
from src.data.eth_ucy_social_dataset import *


def main():
    audit=ROOT/'results/social_residual/data_audit';audit.mkdir(parents=True,exist_ok=True)
    counts={};folds={}
    for scene in SCENES:
        z=build_social_scene(scene);p=SOCIAL_PROCESSED/scene;p.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p/'sequences.npz',**z)
        counts[scene]=neighbor_statistics(z['neighbor_mask'])
    for fold in SCENES:
        folds[fold]={}
        for split in ('train','val'):
            d=ETHUCYSocialDataset(fold,split)
            folds[fold][split]=neighbor_statistics(d.arrays['neighbor_mask'].numpy())
            folds[fold][split]['pedestrian_count']=len(d.tracks)
        train=ETHUCYSocialDataset(fold,'train');val=ETHUCYSocialDataset(fold,'val')
        assert not train.tracks&val.tracks and fold not in train.scenes|val.scenes
        manifest=BASE_RESULTS/'data_audit'/f'manifest_{fold}.json'
        folds[fold]['stage1_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    report={'scenes':counts,'folds':folds,'total_social_samples':sum(q['sample_count'] for q in counts.values()),'N':8,'radius':None,'selection':'nearest at last observation; present at all eight observed frames; same recording; no future access','split':'unchanged stage-one manifest; target pedestrian split','neighbor_split_policy':'Candidates may be any observed pedestrian in the same source recording, including tracks assigned to the other target split; only their eight current observation coordinates are used, never their future labels.','heldout_test_accessed':False}
    (audit/'neighbor_statistics.json').write_text(json.dumps(report,indent=2))
    provenance=json.loads((BASE_RESULTS/'data_audit/data_provenance.json').read_text())
    provenance['social_protocol']=report['selection'];provenance['N']=8
    (audit/'data_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
