import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.eth_ucy_utils import make_model, state_hash
from scripts.density_calibration_protocol import guard_path, require_source_mode
from scripts.social_residual_utils import new_state
from src.models.density_aware_social import DensityAwareSocialResidualPredictor, NEIGHBOR_LIMIT, density_scale_from_density


def test_fixed_count_density_zero_init_and_identity():
    torch.manual_seed(12)
    base, _ = make_model('emt', 42)
    model = DensityAwareSocialResidualPredictor(base, 42).cuda().eval()
    x = torch.randn(9, 8, 4, device='cuda')
    n = torch.randn(9, 8, 8, 4, device='cuda')
    mask = torch.arange(8, device='cuda').reshape(1, 8) < torch.arange(9, device='cuda').reshape(9, 1)
    relation = torch.randn(9, 8, 3, device='cuda')
    with torch.no_grad():
        out = model(x, n, mask, relation)
        old_out = model.shared(x, n, mask, relation)
        expected_density = torch.arange(9, device='cuda').float().reshape(9, 1)/8
        assert torch.equal(out['density'], expected_density)
        assert torch.equal(out['density_scale'], torch.ones(9, 1, device='cuda'))
        assert torch.equal(out['future_pred'], out['base_prediction'])
        assert torch.equal(out['future_pred'], old_out['future_pred'])
        assert torch.equal(out['social_context'], out['calibrated_social_context'])
        assert (out['future_pred']-out['base_prediction']).abs().max() < 1e-7
    assert NEIGHBOR_LIMIT == 8


def test_density_net_receives_only_normalized_mask_count(monkeypatch):
    base, _ = make_model('emt', 42)
    model = DensityAwareSocialResidualPredictor(base, 42).cuda().eval()
    received = []
    original = model.density_calibration.forward
    def capture(density):
        received.append(density.detach().clone())
        return original(density)
    monkeypatch.setattr(model.density_calibration, 'forward', capture)
    mask = torch.tensor([[1,1,0,0,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1]],device='cuda',dtype=torch.bool)
    x=torch.zeros(3,8,4,device='cuda');n=torch.zeros(3,8,8,4,device='cuda');r=torch.zeros(3,8,3,device='cuda')
    with torch.no_grad(): model(x,n,mask,r)
    torch.testing.assert_close(received[0],torch.tensor([[.25],[.5],[1.]],device='cuda'),rtol=0,atol=0)


def test_density_scales_bounded_after_parameter_changes():
    base, _=make_model('emt',42);model=DensityAwareSocialResidualPredictor(base,42).cuda().eval()
    with torch.no_grad():
        model.density_calibration[-1].weight.fill_(100)
        model.density_calibration[-1].bias.fill_(100)
        d=torch.linspace(0,1,101,device='cuda').reshape(-1,1)
        scale=density_scale_from_density(d, model.density_calibration)
    assert torch.isfinite(scale).all() and (scale>.5).all() and (scale<1.5).all()


def test_shared_initialization_exactly_matches_old_social_residual_model():
    base,_=make_model('emt',123)
    from src.models.social_residual import SocialResidualPredictor
    old=SocialResidualPredictor(base,'emt_sr',123)
    base2,_=make_model('emt',123)
    new=DensityAwareSocialResidualPredictor(base2,123)
    assert state_hash(new_state(old))==state_hash(new_state(new.shared))
    assert not any(p.requires_grad for p in new.backbone.parameters())
    assert not new.backbone.training


def test_density_forward_rejects_neighbor_limit_changes():
    base,_=make_model('emt',42);model=DensityAwareSocialResidualPredictor(base,42).cuda()
    with pytest.raises(ValueError,match='Expected neighbor mask'):
        model.forward_context(torch.zeros(1,128,device='cuda'),torch.zeros(1,8,128,device='cuda'),torch.zeros(1,8,3,device='cuda'),torch.ones(1,9,dtype=torch.bool,device='cuda'),torch.zeros(1,12,2,device='cuda'))


@pytest.mark.parametrize('mode',['test','heldout_test','final_train','final_validation'])
def test_source_mode_rejected_before_dataset_read(mode):
    with pytest.raises(PermissionError): require_source_mode(mode)


def test_formal_path_guard_rejects_path_and_symlink(tmp_path):
    with pytest.raises(RuntimeError):guard_path('/x/results/final_eth_ucy_benchmark/predictions/test.npz')
    target=tmp_path/'final_eth_ucy_benchmark';target.mkdir()
    link=tmp_path/'alias';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(RuntimeError):guard_path(link/'test.npz')


def test_protocol_and_model_do_not_add_density_inputs_or_change_frozen_components():
    config=json.loads(Path('configs/density_aware_social_calibration.json').read_text())
    assert config['neighbor_limit']==8
    assert config['density_inputs_only']==['neighbor_mask count divided by fixed limit 8']
    assert config['scheduler'] is None and config['early_stopping'] is False
    assert config['heldout_test_accessed'] is False and config['historical_test_already_accessed'] is True
