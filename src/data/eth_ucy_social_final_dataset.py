"""Final social data: every eligible source target, observation-visible neighbors."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.eth_ucy_dataset import ROOT
from src.data.eth_ucy_final_dataset import final_scenes
from src.data.eth_ucy_social_dataset import select_neighbors

FINAL_PROCESSED = ROOT/'data/processed/eth_ucy_social_final'
FIELDS = ('target_history', 'neighbor_history', 'neighbor_mask', 'neighbor_relation', 'future_target', 'last_obs_pos', 'future_abs')


def select_final_neighbors(lookup, observation_frames, target_id, last_target):
    recording = target_id.split(':')[0]
    # Access only the eight observed keys; do not inspect trajectory eligibility.
    observed = {int(f): {p: xy for p, xy in lookup[int(f)].items() if p.split(':')[0] == recording} for f in observation_frames}
    return select_neighbors(observed, observation_frames, target_id, last_target)


class ETHUCYSocialFinalDataset(Dataset):
    def __init__(self, heldout, mode, processed=None):
        self.heldout, self.mode = heldout, mode
        self.scenes = final_scenes(heldout, mode)
        parts = []
        for scene in self.scenes:
            with np.load(Path(processed or FINAL_PROCESSED)/scene/'sequences.npz', allow_pickle=False) as z:
                part = {k: z[k] for k in z.files}
            if not np.all(part['scene_id'] == scene):
                raise ValueError('Scene mismatch')
            if part['neighbor_mask'].shape[1:] != (8,):
                raise ValueError('N must equal 8')
            mask = part['neighbor_mask']
            for key in ('neighbor_history', 'neighbor_relation'):
                if np.any(part[key][~mask] != 0):
                    raise ValueError('Nonzero padding')
            parts.append(part)
        self.arrays = {k: torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in FIELDS}
        self.audit = {k: np.concatenate([p[k] for p in parts]) for k in ('scene_id', 'ped_id', 'frame_ids', 'obs_abs', 'neighbor_ids', 'neighbor_frame_ids')}
        if not len(self) or not all(torch.isfinite(v).all() for v in self.arrays.values()):
            raise ValueError('Empty/nonfinite social final dataset')

    def __len__(self):
        return len(self.audit['ped_id'])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.arrays.items()}
