import pytest
import numpy as np
from src.data.eth_ucy_dataset import *

@pytest.mark.parametrize('heldout',SCENES)
def test_split(heldout,monkeypatch):
    original=np.load;opened=[]
    def guarded(path,*args,**kwargs):
        assert Path(path).parent.name!=heldout
        opened.append(str(path));return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    tr=ETHUCYDataset(heldout,'train');va=ETHUCYDataset(heldout,'val')
    assert not tr.tracks & va.tracks
    assert heldout not in tr.scenes|va.scenes
    assert len(opened)==8
    for scene,ped in tr.tracks:assert track_split(scene,ped)=='train'
    for scene,ped in va.tracks:assert track_split(scene,ped)=='val'
    assert tr.tracks==ETHUCYDataset(heldout,'train').tracks

def test_reject_heldout_loader():
    with pytest.raises(ValueError):ETHUCYDataset('eth','test')
