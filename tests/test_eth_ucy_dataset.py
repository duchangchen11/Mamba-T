import numpy as np
import pytest
from src.data.eth_ucy_dataset import *

@pytest.mark.parametrize('scene',SCENES)
def test_windows(scene):
    with np.load(ROOT/'data/processed/eth_ucy'/scene/'sequences.npz') as z:
        assert z['obs_input'].shape[1:]==(8,4)
        assert z['future_target'].shape[1:]==(12,2)
        for k in ('obs_input','future_target','obs_abs','future_abs','last_obs_pos'): assert np.isfinite(z[k]).all()
        np.testing.assert_allclose(z['obs_input'][:,-1,:2],0,atol=1e-7)
        np.testing.assert_allclose(z['future_target']+z['last_obs_pos'][:,None],z['future_abs'],atol=2e-6)
        # Recorded source observations alone must reproduce every input feature.
        obs=z['obs_abs'];delta=np.zeros_like(obs);delta[:,1:]=np.diff(obs,axis=1)
        np.testing.assert_allclose(z['obs_input'],np.concatenate((obs-obs[:,-1,None],delta),axis=2))
        assert np.all(np.diff(z['frame_ids'],axis=1)==np.diff(z['frame_ids'],axis=1)[:,:1])

def test_no_future_features():
    obs=np.arange(16,dtype=np.float32).reshape(8,2)
    x1,_,_=features(obs,np.zeros((12,2),dtype=np.float32))
    x2,_,_=features(obs,np.full((12,2),1e6,dtype=np.float32))
    np.testing.assert_array_equal(x1,x2)
