"""Full-source training and single-scene testing, without validation manifests."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.eth_ucy_dataset import ROOT, SCENES


def final_scenes(heldout, mode):
    if heldout not in SCENES or mode not in ('final_train', 'heldout_test'):
        raise ValueError('Expected a known fold and explicit final_train/heldout_test mode')
    return tuple(s for s in SCENES if s != heldout) if mode == 'final_train' else (heldout,)


class ETHUCYFinalDataset(Dataset):
    def __init__(self, heldout, mode, processed=None):
        self.heldout, self.mode = heldout, mode
        self.scenes = final_scenes(heldout, mode)
        parts = []
        for scene in self.scenes:
            with np.load(Path(processed or ROOT/'data/processed/eth_ucy')/scene/'sequences.npz', allow_pickle=False) as z:
                part = {k: z[k] for k in z.files}
            if not np.all(part['scene_id'] == scene):
                raise ValueError('Scene label mismatch')
            parts.append(part)
        self.audit = {k: np.concatenate([p[k] for p in parts]) for k in ('scene_id', 'ped_id', 'frame_ids', 'obs_abs')}
        self.arrays = {k: torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in ('obs_input', 'future_target', 'last_obs_pos', 'future_abs')}
        if not len(self) or not all(torch.isfinite(v).all() for v in self.arrays.values()):
            raise ValueError('Empty/nonfinite final dataset')

    def __len__(self):
        return len(self.audit['ped_id'])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.arrays.items()}
