import pytest,numpy as np,torch
from pathlib import Path
from src.data.eth_ucy_social_clean_dataset import *
from src.data.eth_ucy_social_dataset import ETHUCYSocialDataset

@pytest.mark.parametrize('fold',SCENES)
def test_train_neighbors_are_train_targets_and_split_unchanged(fold):
    d=ETHUCYSocialCleanDataset(fold,'train');old=ETHUCYSocialDataset(fold,'train');manifest=load_manifest(fold)
    assert d.track_ids==old.track_ids and d.tracks==old.tracks
    for key in ('target_history','future_target','future_abs','last_obs_pos'):assert torch.equal(d.arrays[key],old.arrays[key])
    mask=d.arrays['neighbor_mask'].numpy();relation=d.arrays['neighbor_relation'].numpy()
    for scene in SCENES:
        if scene==fold:continue
        rows=d.audit['scene_id']==scene;neighbors=set(d.audit['neighbor_ids'][rows][mask[rows]].tolist())
        assert neighbors<=set(manifest['train'][scene])
        assert not neighbors&set(manifest['val'][scene])
    assert np.all(d.audit['neighbor_split'][mask]=='train')
    assert np.all(d.audit['neighbor_split'][~mask]=='padding')
    assert np.all(d.arrays['neighbor_history'].numpy()[~mask]==0)
    assert np.all(relation[~mask]==0)
    for i in range(len(d)):assert np.all(np.diff(relation[i,mask[i],2])>=0)
    expected=np.broadcast_to(d.audit['frame_ids'][:,None,:8],(len(d),8,8))
    np.testing.assert_array_equal(d.audit['neighbor_frame_ids'][mask],expected[mask])
    for key in FIELDS:assert torch.isfinite(d.arrays[key]).all()

@pytest.mark.parametrize('fold',SCENES)
def test_validation_inputs_identical_and_heldout_not_opened(fold,monkeypatch):
    original=np.load
    def guarded(path,*args,**kwargs):
        assert Path(path).parent.name!=fold;return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    d=ETHUCYSocialCleanDataset(fold,'val');old=ETHUCYSocialDataset(fold,'val')
    assert d.track_ids==old.track_ids
    for key in FIELDS:assert torch.equal(d.arrays[key],old.arrays[key])
    tr=ETHUCYSocialCleanDataset(fold,'train');assert not tr.tracks&d.tracks
    assert fold not in tr.scenes|d.scenes


def test_train_filters_before_ranking_and_val_observation_only():
    frames={f:{'video:target':np.array([0,0],dtype=np.float32),'video:val':np.array([1,0],dtype=np.float32),'video:train':np.array([4,0],dtype=np.float32),'video:other':np.array([2,0],dtype=np.float32),'other_recording:train':np.array([.1,0],dtype=np.float32)} for f in range(8)}
    train={'video:target','video:train','other_recording:train'};val={'video:val'}
    class ObservationOnly(dict):
        def __getitem__(self,key):
            assert 0<=key<8;return super().__getitem__(key)
    a=select_clean_neighbors(ObservationOnly(frames),np.arange(8),'video:target',np.zeros(2,dtype=np.float32),'train',train,val)
    assert a['neighbor_mask'].sum()==1 and a['neighbor_ids'][0]=='video:train'
    b=select_clean_neighbors(ObservationOnly(frames),np.arange(8),'video:target',np.zeros(2,dtype=np.float32),'val',train,val)
    assert b['neighbor_ids'][:3].tolist()==['video:val','video:other','video:train']
    assert b['neighbor_split'][:3].tolist()==['val','other','train']
    frames[8]={'video:future_only':np.zeros(2)};frames[7]['video:future_only']=np.zeros(2)
    c=select_clean_neighbors(ObservationOnly(frames),np.arange(8),'video:target',np.zeros(2,dtype=np.float32),'val',train,val)
    for key in b:np.testing.assert_array_equal(b[key],c[key])


def test_reject_contaminated_saved_data(tmp_path):
    fold='eth';scene='hotel';manifest=load_manifest(fold)
    source=CLEAN_PROCESSED/fold/scene/'train.npz'
    with np.load(source) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['neighbor_ids'][0,0]=manifest['val'][scene][0];arrays['neighbor_mask'][0,0]=True;arrays['neighbor_split'][0,0]='train'
    p=tmp_path/fold/scene;p.mkdir(parents=True);np.savez_compressed(p/'train.npz',**arrays)
    # eth loader starts with hotel: membership is checked before opening subsequent scenes.
    with pytest.raises(ValueError,match='cross-split'):ETHUCYSocialCleanDataset(fold,'train',tmp_path)

@pytest.mark.parametrize('kind',['emt','ett'])
def test_clean_frozen_checkpoint_zero_init_base_regression(kind):
    from scripts.clean_social_utils import prepare_clean_pair
    from scripts.social_residual_utils import forward_cached
    from src.models.social_residual import SocialResidualPredictor
    # Actual first-stage checkpoint; use a small clean source subset rather than full cache.
    from scripts.social_residual_utils import load_backbone,cache_contexts
    from src.data.eth_ucy_dataset import ROOT
    b,meta=load_backbone(kind,'eth',42,ROOT/'results/clean_social_validation/data_audit/stage1_freeze_inventory.json')
    d=ETHUCYSocialCleanDataset('eth','val')
    class Small:
        arrays={k:v[:3] for k,v in d.arrays.items()}
        def __len__(self):return 3
    cached=cache_contexts(b,Small());m=SocialResidualPredictor(b,kind+'_sr',42).cuda().train()
    assert not b.training and all(not p.requires_grad for p in b.parameters())
    out=forward_cached(m,cached,slice(0,3));assert torch.equal(out['future_pred'],cached['base_prediction'])
    with torch.no_grad():raw=b(Small.arrays['target_history'].cuda())['future_pred']
    assert torch.equal(out['future_pred'],raw)
    out['future_pred'].square().mean().backward();assert all(p.grad is None for p in b.parameters())
