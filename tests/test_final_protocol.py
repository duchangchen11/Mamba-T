import json
import pytest
from scripts.freeze_final_eth_ucy_protocol import derive_epochs
from scripts.final_protocol import dump, sha_file, verify_protocol, MODELS, SCENES


def test_epoch_medians_use_validation_only():
    epochs, source = derive_epochs()
    assert set(epochs) == set(SCENES)
    for fold in SCENES:
        assert set(epochs[fold]) == set(MODELS)
        for model in MODELS:
            q = source[fold][model]
            assert epochs[fold][model] == sorted(q['validation_best_epochs'])[1]
            assert all('metrics_validation.json' in p for p in q['sources'])
    assert epochs['eth'] == {'ett': 38, 'emt': 10, 'ett_sr': 10, 'emt_sr': 14}


def fake_freeze(tmp_path):
    root = tmp_path/'root'
    results = root/'results'
    path = root/'configs/final.json'
    dump(path, {'lr': .001})
    protocol = {'config_sha256': {'configs/final.json': sha_file(path)}, 'code_sha256': {}, 'dataset_sha256': {}}
    dump(results/'protocol_frozen.json', protocol)
    (results/'protocol_frozen.sha256').write_text(sha_file(results/'protocol_frozen.json')+'\n')
    return root, results


def test_frozen_config_rejects_mutation(tmp_path):
    root, results = fake_freeze(tmp_path)
    verify_protocol(root, results)
    dump(root/'configs/final.json', {'lr': .01})
    with pytest.raises(ValueError, match='Frozen config'):
        verify_protocol(root, results)


def test_protocol_file_rejects_mutation(tmp_path):
    root, results = fake_freeze(tmp_path)
    with (results/'protocol_frozen.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='protocol hash'):
        verify_protocol(root, results)
