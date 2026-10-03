from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.eth_ucy_dataset import ROOT, SCENES
from src.data.eth_ucy_final_dataset import ETHUCYFinalDataset
from src.data.eth_ucy_social_final_dataset import ETHUCYSocialFinalDataset, select_final_neighbors


@pytest.mark.parametrize('fold', SCENES)
def test_final_social_all_source_targets_without_heldout(fold, monkeypatch):
    original = np.load
    def guarded(path, *args, **kwargs):
        assert Path(path).parent.name != fold
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    social = ETHUCYSocialFinalDataset(fold, 'final_train')
    target = ETHUCYFinalDataset(fold, 'final_train')
    assert len(social) == len(target)
    np.testing.assert_array_equal(social.audit['ped_id'], target.audit['ped_id'])
    assert torch.equal(social.arrays['target_history'], target.arrays['obs_input'])
    assert social.arrays['neighbor_history'].shape == (len(social), 8, 8, 4)
    assert social.arrays['neighbor_relation'].shape == (len(social), 8, 3)
    mask = social.arrays['neighbor_mask'].numpy()
    ids = social.audit['neighbor_ids']
    expected_frames = np.broadcast_to(social.audit['frame_ids'][:, None, :8], (len(social), 8, 8))
    np.testing.assert_array_equal(social.audit['neighbor_frame_ids'][mask], expected_frames[mask])
    assert all(str(p) != str(n) and str(p).split(':')[0] == str(n).split(':')[0] for p, neighbors, valid in zip(social.audit['ped_id'], ids, mask) for n in neighbors[valid])


def test_neighbors_read_only_observed_keys_and_accept_noneligible_tracks():
    class ObservationOnly(dict):
        def __getitem__(self, key):
            assert key in range(8)
            return super().__getitem__(key)
    lookup = ObservationOnly({f: {'v:t': np.zeros(2, dtype=np.float32), **{f'v:n{i}': np.array([i+1, 0], dtype=np.float32) for i in range(10)}, 'other:n': np.array([.01, 0], dtype=np.float32)} for f in range(8)})
    lookup[8] = {'v:future_only': np.zeros(2)}
    out = select_final_neighbors(lookup, np.arange(8), 'v:t', np.zeros(2))
    assert out['neighbor_ids'].tolist() == [f'v:n{i}' for i in range(8)]
    assert out['neighbor_mask'].all()
    # None of these neighbors has a future entry or target-window eligibility.
    np.testing.assert_array_equal(out['neighbor_relation'][:, 2], np.arange(1, 9))


def test_final_neighbor_padding():
    lookup = {f: {'v:t': np.zeros(2)} for f in range(8)}
    out = select_final_neighbors(lookup, np.arange(8), 'v:t', np.zeros(2))
    assert not out['neighbor_mask'].any()
    assert not out['neighbor_history'].any() and not out['neighbor_relation'].any()
