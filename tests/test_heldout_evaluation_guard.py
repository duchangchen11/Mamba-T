import pytest
from scripts.final_protocol import dump, read, sha_file, require_heldout_evaluation, MODELS, SEEDS, SCENES
from test_final_protocol import fake_freeze


def complete_matrix(tmp_path):
    root, results = fake_freeze(tmp_path)
    protocol = read(results/'protocol_frozen.json')
    protocol['final_epochs'] = {f: {m: 2 for m in MODELS} for f in SCENES}
    dump(results/'protocol_frozen.json', protocol)
    psha = sha_file(results/'protocol_frozen.json')
    (results/'protocol_frozen.sha256').write_text(psha+'\n')
    runs = []
    for fold in SCENES:
        for model in MODELS:
            for seed in SEEDS:
                path = root/'checkpoints'/f'{fold}_{model}_{seed}.pt'
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b'fixed checkpoint')
                folder = results/'training'/f'heldout_{fold}'/model/f'seed_{seed}'
                report = {'protocol_sha256': psha, 'nan_inf': False, 'epochs_run': 2, 'checkpoint_sha256': sha_file(path), 'heldout_test_evaluated': False}
                dump(folder/'training_report.json', report)
                dump(folder/'training_history.json', [{'epoch': 1}, {'epoch': 2}])
                runs.append({'fold': fold, 'model': model, 'seed': seed, 'checkpoint_path': str(path.relative_to(root)), 'checkpoint_sha256': sha_file(path), 'training_report_sha256': sha_file(folder/'training_report.json'), 'training_history_sha256': sha_file(folder/'training_history.json')})
    dump(results/'FINAL_TRAINING_COMPLETE.json', {'completed_runs': 60, 'protocol_sha256': psha, 'test_metrics_accessed_before_freeze': False, 'runs': runs})
    return root, results


def test_guard_requires_explicit_enable():
    with pytest.raises(PermissionError):
        require_heldout_evaluation(False)


def test_guard_rejects_missing_completion(tmp_path):
    root, results = fake_freeze(tmp_path)
    with pytest.raises(FileNotFoundError):
        require_heldout_evaluation(True, root=root, results=results)


def test_guard_accepts_complete_matrix_and_rejects_missing_run(tmp_path):
    root, results = complete_matrix(tmp_path)
    require_heldout_evaluation(True, root=root, results=results)
    q = read(results/'FINAL_TRAINING_COMPLETE.json')
    q['runs'].pop()
    dump(results/'FINAL_TRAINING_COMPLETE.json', q)
    with pytest.raises(ValueError, match='Incomplete'):
        require_heldout_evaluation(True, root=root, results=results)


def test_guard_rejects_changed_checkpoint(tmp_path):
    root, results = complete_matrix(tmp_path)
    (root/'checkpoints/eth_ett_42.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        require_heldout_evaluation(True, root=root, results=results)


def test_guard_rejects_changed_history(tmp_path):
    root, results = complete_matrix(tmp_path)
    dump(results/'training/heldout_eth/ett/seed_42/training_history.json', [{'epoch': 100}])
    with pytest.raises(ValueError, match='training log changed'):
        require_heldout_evaluation(True, root=root, results=results)
