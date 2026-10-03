import numpy as np
import pytest
import torch
from src.data.eth_ucy_dataset import SCENES
from src.data.eth_ucy_dataset import ETHUCYDataset
from src.data.eth_ucy_training_regime_dataset import TrainingRegimeSourceDataset
from src.models.social_residual import SocialResidualPredictor
from scripts.eth_ucy_utils import make_model, state_hash
from scripts.training_regime_metrics import correction_metrics, neighbor_groups, correction_groups, strata
from scripts.training_regime_utils import run_fixed_epochs, diagnostic_best, save_last_checkpoint


@pytest.mark.parametrize('mode', ['heldout_test', 'final_train', 'test', 'source_validation'])
def test_diagnostic_loader_rejects_nonsource_modes_before_read(mode, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No file may be read for a rejected mode')
    monkeypatch.setattr(np, 'load', forbidden)
    with pytest.raises(PermissionError):
        TrainingRegimeSourceDataset('eth', mode)


@pytest.mark.parametrize('fold', SCENES)
def test_diagnostic_source_split_exactly_matches_old(fold, monkeypatch):
    from pathlib import Path
    original = np.load
    def guarded(path, *args, **kwargs):
        assert Path(path).parent.name != fold
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', guarded)
    for mode in ('train', 'val'):
        d = TrainingRegimeSourceDataset(fold, mode)
        legacy = ETHUCYDataset(fold, mode)
        assert d.tracks == legacy.tracks and len(d) == len(legacy)
        assert fold not in d.scenes
        for i in (0, len(d)//2, len(d)-1):
            assert torch.equal(d.arrays['target_history'][i], legacy[i]['obs_input'])
            assert torch.equal(d.arrays['future_target'][i], legacy[i]['future_target'])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_passive_validation_cannot_change_weights_lr_or_stop(device, tmp_path):
    torch.set_num_threads(1)
    x = torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3)/10
    y = torch.ones(4, 2, device=device)
    def experiment(noise):
        torch.manual_seed(71)
        torch.cuda.manual_seed_all(71)
        model = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Dropout(.3), torch.nn.Linear(6, 2)).to(device)
        calls, states = [], []
        def validation():
            calls.append(len(calls)+1)
            if noise:
                torch.rand(113)
                if device == 'cuda':
                    torch.rand(79, device=device)
            return {'validation_ADE': float(100-len(calls) if len(calls) < 7 else 1000), 'validation_FDE': float(noise*10000+len(calls))}
        history = run_fixed_epochs(model, 7, lambda epoch: [None], lambda batch: (model(x), y), validation, lambda h: states.append(state_hash(model.state_dict())))
        return model, history, states, calls
    a, ah, astates, calls = experiment(False)
    b, bh, bstates, bcalls = experiment(True)
    assert len(calls) == len(bcalls) == len(ah) == len(bh) == 7
    assert state_hash(a.state_dict()) == state_hash(b.state_dict())
    assert astates == bstates
    assert [q['train_loss'] for q in ah] == [q['train_loss'] for q in bh]
    assert all(q['lr'] == .001 for q in ah+bh)
    assert diagnostic_best(ah)['diagnostic_best_validation_epoch'] == 6
    path = tmp_path/'last.pt'
    save_last_checkpoint(path, a, 7, {'seed': 71, 'model': 'toy-fixed'})
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 7
    assert checkpoint['model_name'] == 'toy-fixed'
    assert state_hash(checkpoint['model']) == astates[-1] != astates[-2]


def test_validation_cannot_mutate_model():
    model = torch.nn.Linear(1, 1)
    def invalid():
        model.weight.add_(1)
        return {'validation_ADE': 0., 'validation_FDE': 0.}
    with pytest.raises(RuntimeError, match='mutated'):
        run_fixed_epochs(model, 1, lambda epoch: [None], lambda batch: (model(torch.ones(2, 1)), torch.zeros(2, 1)), invalid)


def test_correction_metric_definition_and_stationary_base():
    residual = np.zeros((2, 12, 2));base = np.zeros_like(residual)
    residual[0, :, :] = [3, 4];base[0, :, :] = [6, 8]
    residual[1, :, 0] = 2
    stats, samples = correction_metrics(residual, base, epsilon=1e-8)
    np.testing.assert_array_equal(samples['residual_norm'], [5., 2.])
    np.testing.assert_array_equal(samples['base_trajectory_norm'], [10., 0.])
    np.testing.assert_allclose(samples['correction_ratio'], [5/(10+1e-8), 2/1e-8])
    assert stats['residual_norm']['mean'] == 3.5 and np.isfinite(stats['correction_ratio']['max'])


def test_density_distance_and_correction_group_boundaries():
    count = np.array([0, 1, 2, 3, 4, 5, 8])
    distance = np.array([0, 1.99, 2., 4., 4.01, 1., 8.])
    groups = neighbor_groups(count, distance)
    assert [int(q.sum()) for q in groups['neighbor_count'].values()] == [1, 2, 2, 2]
    assert [int(q.sum()) for q in groups['neighbor_distance'].values()] == [1, 2, 2, 2]
    assert groups['neighbor_distance']['2-4m'][2:4].all()
    ratio = np.array([0, .05, .05001, .1, .10001, .2, .20001])
    grouped = correction_groups(ratio)
    assert [int(q.sum()) for q in grouped.values()] == [2, 2, 2, 1]
    assert np.all(sum(q.astype(int) for q in grouped.values()) == 1)


def test_strata_gains_and_improvement_rates():
    samples = {'sample_ADE': np.array([1., 3.]), 'sample_FDE': np.array([2., 4.]), 'base_ADE': np.array([2., 2.]), 'base_FDE': np.array([3., 3.])}
    q = strata(samples, {'all': np.array([True, True]), 'empty': np.array([False, False])}, True)
    assert q['all']['gain_ADE'] == 0 and q['all']['improvement_rate_ADE'] == .5
    assert q['empty']['sample_count'] == 0 and q['empty']['ADE'] is None


def test_analysis_rejects_test_before_protocol_or_file_access(monkeypatch):
    from scripts import summarize_training_regime_diagnostic as summary
    def forbidden():
        pytest.fail('Protocol/file access before mode rejection')
    monkeypatch.setattr(summary, 'verify_protocol', forbidden)
    with pytest.raises(PermissionError):
        summary.main('heldout_test')


def test_fold_equal_strata_aggregation_with_empty_fold():
    from scripts.summarize_training_regime_diagnostic import aggregate_strata, table
    rows = [{'fold': f, 'groups': {'a': {'sample_count': n, 'ADE': v}}} for f,n,v in [('eth',1,1.),('eth',9,3.),('hotel',100,5.),('univ',0,None),('zara1',0,None),('zara2',0,None)]]
    q = aggregate_strata(rows,'groups')['a']
    assert q['present_folds'] == 2 and q['sample_count_across_seed_predictions'] == 110
    assert q['ADE'] == pytest.approx((2.8+5)/2)
    assert '| ETT | 1 | 2 |' in table(['model','ADE','FDE'],[['ETT',1,2]])


@pytest.mark.parametrize('kind', ['ett', 'emt'])
def test_diagnostic_sr_backbone_frozen_zero_init(kind):
    base, _ = make_model(kind, 42)
    model = SocialResidualPredictor(base.cuda(), kind+'_sr', 42).cuda().train()
    before = state_hash(base.state_dict())
    x = torch.randn(2, 8, 4, device='cuda');n = torch.randn(2, 8, 8, 4, device='cuda');mask = torch.ones(2, 8, dtype=torch.bool, device='cuda');r = torch.randn(2, 8, 3, device='cuda')
    out = model(x, n, mask, r)
    assert torch.equal(out['future_pred'], out['base_prediction'])
    out['future_pred'].square().mean().backward()
    assert not base.training and all(p.grad is None and not p.requires_grad for p in base.parameters())
    assert before == state_hash(base.state_dict())
