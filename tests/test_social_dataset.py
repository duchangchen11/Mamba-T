import pytest
import numpy as np
from src.data.eth_ucy_social_dataset import *
from src.data.eth_ucy_dataset import ETHUCYDataset

@pytest.mark.parametrize('fold',SCENES)
def test_frozen_splits_and_heldout_not_opened(fold,monkeypatch):
    original=np.load
    def guarded(path,*args,**kwargs):
        assert Path(path).parent.name!=fold;return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    tr=ETHUCYSocialDataset(fold,'train');va=ETHUCYSocialDataset(fold,'val')
    assert not tr.tracks&va.tracks and fold not in tr.scenes|va.scenes
    for split,social in (('train',tr),('val',va)):
        old=ETHUCYDataset(fold,split)
        assert social.tracks==old.tracks and len(social)==len(old)
        for i in (0,len(social)//2,len(social)-1):
            np.testing.assert_array_equal(social[i]['target_history'].numpy(),old[i]['obs_input'].numpy())
            np.testing.assert_array_equal(social[i]['future_target'].numpy(),old[i]['future_target'].numpy())


def test_no_heldout_evaluation_split():
    with pytest.raises(ValueError):ETHUCYSocialDataset('eth','test')
