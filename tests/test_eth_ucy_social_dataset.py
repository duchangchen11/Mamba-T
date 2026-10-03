import numpy as np
import pytest
from src.data.eth_ucy_social_dataset import *

@pytest.mark.parametrize('scene',SCENES)
def test_social_shapes_alignment_padding(scene):
    with np.load(SOCIAL_PROCESSED/scene/'sequences.npz') as z:
        n=len(z['target_history']);mask=z['neighbor_mask'];history=z['neighbor_history'];relation=z['neighbor_relation']
        assert z['target_history'].shape==(n,8,4)
        assert history.shape==(n,8,8,4) and mask.shape==(n,8)
        assert relation.shape==(n,8,3) and z['future_target'].shape==(n,12,2)
        for a in (z['target_history'],history,relation,z['future_target']):assert np.isfinite(a).all()
        assert np.all(history[~mask]==0) and np.all(relation[~mask]==0)
        assert np.all(z['neighbor_ids'][~mask]=='')
        frames=np.broadcast_to(z['frame_ids'][:,None,:8],(n,8,8))
        np.testing.assert_array_equal(z['neighbor_frame_ids'][mask],frames[mask])
        assert np.all(np.diff(mask.astype(int),axis=1)<=0)
        np.testing.assert_allclose(history[mask][:,-1,:2],0)
        for i in range(n):assert np.all(np.diff(relation[i,mask[i],2])>=0)
        # Resolve every sampled neighbor from source observations, including own motion coordinates.
        lookups={Path(f).stem:recording_lookup(ROOT/'data/raw/eth_ucy',f) for f in FILES[scene]}
        for i in np.linspace(0,n-1,min(n,40),dtype=int):
            pid=str(z['ped_id'][i]);lookup=lookups[pid.split(':')[0]]
            expected=select_neighbors(lookup,z['frame_ids'][i,:8],pid,z['last_obs_pos'][i])
            for key in expected:np.testing.assert_array_equal(z[key][i],expected[key])


def test_selection_observation_only_and_missing_frames():
    frames={i:{'target':np.array([0,0],dtype=np.float32),'near':np.array([1,0],dtype=np.float32),'far':np.array([3,0],dtype=np.float32)} for i in range(8)}
    frames[0].pop('near')
    class ObservationsOnly(dict):
        def __getitem__(self,key):
            assert 0<=key<8;return super().__getitem__(key)
    a=select_neighbors(ObservationsOnly(frames),np.arange(8),'target',np.zeros(2,dtype=np.float32))
    assert a['neighbor_ids'][0]=='far' and a['neighbor_mask'].sum()==1
    frames[8]={'future_only':np.zeros(2),'near':np.zeros(2)}
    b=select_neighbors(ObservationsOnly(frames),np.arange(8),'target',np.zeros(2,dtype=np.float32))
    for key in a:np.testing.assert_array_equal(a[key],b[key])
