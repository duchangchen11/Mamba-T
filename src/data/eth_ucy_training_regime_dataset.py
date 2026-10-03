"""Source-only diagnostic wrapper; held-out and full-source modes are forbidden."""
from torch.utils.data import Dataset
from src.data.eth_ucy_social_clean_dataset import ETHUCYSocialCleanDataset


def require_source_mode(mode):
    if mode not in ('train', 'val'):
        raise PermissionError('SOURCE VALIDATION DIAGNOSTIC ONLY: heldout_test/final modes prohibited')


class TrainingRegimeSourceDataset(Dataset):
    def __init__(self, heldout, mode, social=False, processed=None):
        require_source_mode(mode)
        source = ETHUCYSocialCleanDataset(heldout, mode, processed)
        self.heldout, self.mode, self.social = heldout, mode, social
        self.arrays, self.audit = source.arrays, source.audit
        self.tracks, self.track_ids, self.scenes = source.tracks, source.track_ids, source.scenes
        if heldout in self.scenes:
            raise PermissionError('Held-out scene appeared in source diagnostic')

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, index):
        if self.social:
            return {k: v[index] for k, v in self.arrays.items()}
        return {'obs_input': self.arrays['target_history'][index], 'future_target': self.arrays['future_target'][index]}
