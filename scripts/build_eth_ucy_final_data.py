"""Build independent final social files; data quality checks never predict test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data.eth_ucy_dataset import ROOT, SCENES, FILES
from src.data.eth_ucy_social_dataset import recording_lookup, neighbor_statistics
from src.data.eth_ucy_social_final_dataset import FINAL_PROCESSED, select_final_neighbors
from scripts.eth_ucy_utils import dump


def main():
    per_scene = {}
    masks = {}
    for scene in SCENES:
        with np.load(ROOT/'data/processed/eth_ucy'/scene/'sequences.npz', allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        assert np.all(np.diff(a['frame_ids'], axis=1) == 10)
        lookups = {Path(f).stem: recording_lookup(ROOT/'data/raw/eth_ucy', f) for f in FILES[scene]}
        neighbors = [select_final_neighbors(lookups[str(p).split(':')[0]], a['frame_ids'][i, :8], str(p), a['last_obs_pos'][i]) for i, p in enumerate(a['ped_id'])]
        out = {k: np.stack([n[k] for n in neighbors]) for k in neighbors[0]}
        out.update({'target_history': a['obs_input'], 'future_target': a['future_target'], 'future_abs': a['future_abs'], 'last_obs_pos': a['last_obs_pos'], 'ped_id': a['ped_id'], 'frame_ids': a['frame_ids'], 'scene_id': a['scene_id'], 'obs_abs': a['obs_abs']})
        assert all(np.isfinite(v).all() for v in out.values() if v.dtype.kind in 'fbiu')
        # Full visibility is numerically identical to the old observation-only builder.
        with np.load(ROOT/'data/processed/eth_ucy_social'/scene/'sequences.npz', allow_pickle=False) as old:
            for k in old.files:
                np.testing.assert_array_equal(out[k], old[k])
        folder = FINAL_PROCESSED/scene
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/'sequences.npz', **out)
        q = neighbor_statistics(out['neighbor_mask'])
        counts = out['neighbor_mask'].sum(1)
        q['group_proportions'] = {'0': float((counts == 0).mean()), '1-2': float(((counts >= 1)&(counts <= 2)).mean()), '3-4': float(((counts >= 3)&(counts <= 4)).mean()), '5+': float((counts >= 5).mean())}
        q['pedestrian_count'] = len(set(a['ped_id'].tolist()))
        per_scene[scene] = q
        masks[scene] = out['neighbor_mask']
    audit = ROOT/'results/final_eth_ucy_benchmark/data_audit'
    fold_statistics = {f: {'train': neighbor_statistics(np.concatenate([masks[s] for s in SCENES if s != f])), 'test': per_scene[f]} for f in SCENES}
    dump(audit/'neighbor_statistics.json', {'per_scene': per_scene, 'folds': fold_statistics, 'N': 8, 'radius': None, 'policy': 'same recording, present in all 8 observed frames, exclude target, nearest 8; no future filtering', 'test_metrics_accessed': False})
    dump(audit/'train_scene_counts.json', {f: {'scene_list': [s for s in SCENES if s != f], 'scene_counts': {s: per_scene[s]['sample_count'] for s in SCENES if s != f}, 'total': sum(per_scene[s]['sample_count'] for s in SCENES if s != f)} for f in SCENES})
    dump(audit/'test_scene_counts.json', {f: {'scene_list': [f], 'total': per_scene[f]['sample_count']} for f in SCENES})
    print({s: q['sample_count'] for s, q in per_scene.items()})


if __name__ == '__main__':
    main()
