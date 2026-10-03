from pathlib import Path
import numpy as np
import pytest
from src.data.eth_ucy_dataset import ROOT, SCENES
from src.data.eth_ucy_final_dataset import ETHUCYFinalDataset


@pytest.mark.parametrize('fold', SCENES)
def test_final_train_full_sources_and_never_opens_heldout(fold, monkeypatch):
    original = np.load
    expected = 0
    for scene in SCENES:
        if scene != fold:
            with original(ROOT/'data/processed/eth_ucy'/scene/'sequences.npz') as z:
                expected += len(z['ped_id'])
    def guarded(path, *args, **kwargs):
        assert Path(path).parent.name != fold
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    dataset = ETHUCYFinalDataset(fold, 'final_train')
    assert set(dataset.scenes) == set(SCENES)-{fold}
    assert len(dataset) == expected
    assert fold not in set(dataset.audit['scene_id'])
    assert dataset.arrays['obs_input'].shape == (expected, 8, 4)


@pytest.mark.parametrize('fold', SCENES)
def test_test_loader_only_opens_heldout(fold, monkeypatch):
    # Check legacy routing against synthetic data; never reopen a real held-out file.
    from contextlib import contextmanager
    n = 2
    synthetic = {'scene_id': np.full(n, fold), 'ped_id': np.array(['synthetic:1', 'synthetic:2']), 'frame_ids': np.tile(np.arange(20)*10, (n, 1)), 'obs_abs': np.zeros((n, 8, 2), dtype=np.float32), 'obs_input': np.zeros((n, 8, 4), dtype=np.float32), 'future_target': np.zeros((n, 12, 2), dtype=np.float32), 'last_obs_pos': np.zeros((n, 2), dtype=np.float32), 'future_abs': np.zeros((n, 12, 2), dtype=np.float32)}
    class Archive(dict):
        @property
        def files(self):
            return list(self)
    @contextmanager
    def guarded(path, *args, **kwargs):
        assert Path(path).parent.name == fold
        yield Archive(synthetic)
    monkeypatch.setattr(np, 'load', guarded)
    dataset = ETHUCYFinalDataset(fold, 'heldout_test')
    assert set(dataset.audit['scene_id']) == {fold}


def test_rejects_ambiguous_modes():
    with pytest.raises(ValueError):
        ETHUCYFinalDataset('eth', 'train')
