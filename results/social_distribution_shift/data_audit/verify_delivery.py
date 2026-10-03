"""Verify delivery using source-validation artifacts and Git metadata only."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scripts.social_shift_protocol import ROOT, RESULTS, PREVIOUS, verify_inputs, verify_history
from scripts.social_shift_statistics import collapse_windows

protocol = verify_inputs()
history = verify_history()
integrity = json.loads((RESULTS/'data_audit/result_integrity.json').read_text())
comparison = json.loads((RESULTS/'comparison.json').read_text())
for name, expected in integrity['output_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
all_files = {str(p.relative_to(ROOT)) for p in RESULTS.rglob('*') if p.is_file() and p != RESULTS/'data_audit/result_integrity.json'}
assert all_files == set(integrity['output_sha256'])
assert [int(n) for n in re.findall(r'^(\d+)\. ', (RESULTS/'brain_report.md').read_text(), re.M)] == list(range(1, 36))
for name in ('summary.md','brain_report.md'):
    for line in (RESULTS/name).read_text().splitlines():
        if line.startswith('|'):
            assert '|SMD|' not in line, 'Markdown literal pipes must be escaped'
assert comparison['heldout_test_accessed'] is False and comparison['historical_test_already_accessed'] is True
assert comparison['trajectory_training'] is False
assert len(history) == 1504
assert all(a['eval_no_grad'] and a['weights_unchanged'] and not a['optimizer_created'] and not a['heldout_test_accessed'] for a in integrity['forward_audits'])
assert len(integrity['forward_audits']) == 30
assert max(a['attention_context_max_abs_diff'] for a in integrity['forward_audits']) == 0
responses = {}
exact_columns = {'SR_ADE':'sample_ADE', 'SR_FDE':'sample_FDE', 'base_ADE':'base_ADE', 'base_FDE':'base_FDE', 'residual_norm':'residual_norm', 'correction_ratio':'correction_ratio'}
for model in ('emt_sr','ett_sr'):
    paths = sorted((RESULTS/model).glob('heldout_*/seed_*/samples.npz'))
    assert len(paths) == 15
    batches = []
    for p in paths:
        fold = p.parent.parent.name.removeprefix('heldout_')
        with np.load(p, allow_pickle=False) as archive:
            raw = dict(archive)
        assert fold not in set(raw['scene']), 'Held-out scene is excluded in each forward'
        old = PREVIOUS/p.parent.parent.name/model/p.parent.name/'validation_samples.npz'
        with np.load(old, allow_pickle=False) as archive:
            saved = dict(archive)
        assert np.array_equal(raw['scene'], saved['scene_id'])
        assert np.array_equal(raw['ped_id'], saved['target_ped_id'])
        assert np.array_equal(raw['observation_frame_ids'], saved['frame_ids'][:,:8])
        for current, previous in exact_columns.items():
            assert np.array_equal(raw[current], saved[previous]), (p, current)
        assert np.array_equal(raw['ADE_gain'], raw['base_ADE']-raw['SR_ADE'])
        assert np.array_equal(raw['FDE_gain'], raw['base_FDE']-raw['SR_FDE'])
        assert np.all((raw['neighbor_count']>=0)&(raw['neighbor_count']<=8))
        assert np.all((raw['attention_entropy']>=0)&(raw['attention_entropy']<=1))
        assert np.all(raw['attention_entropy'][raw['neighbor_count']<=1] == 0)
        batches.append(raw)
    stacked = {k:np.concatenate([b[k] for b in batches]) for k in batches[0]}
    assert len(stacked['scene']) == 48384
    collapsed = collapse_windows(stacked)
    assert len(collapsed['scene']) == 4032
    assert set(collapsed['prediction_count']) == {12}
    responses[model] = collapsed
assert np.array_equal(responses['emt_sr']['sample_uid'], responses['ett_sr']['sample_uid'])
for row in comparison['classifier']['folds']:
    assert row['pedestrian_overlap']==0 and row['train_scene_count']==5
assert set(comparison['classifier']['feature_names']) == {'neighbor_count','mean_neighbor_distance','nearest_neighbor_distance','mean_relative_speed','mean_closing_speed'}
verification = json.loads((RESULTS/'data_audit/verification.json').read_text())
assert verification['pytest_passed'] == 128 and verification['pytest_failed'] == 0
assert '128 passed' in (RESULTS/'data_audit/pytest.txt').read_text()
print(json.dumps({'source_input_files':protocol['source_input_files'], 'verified_output_hashes':len(all_files), 'historical_frozen_files':len(history), 'saved_response_columns_bitwise_equal_all_30_runs':True, 'unique_windows':4032, 'records_both_models':96768, 'replicates_per_window':12, 'brain_report_items':35, 'attention_context_max_abs_diff':0, 'pytest_passed':128}))
