import numpy as np
import pytest
import torch
from scripts.eth_ucy_utils import make_model, state_hash
from src.models.social_residual import SocialResidualPredictor
from scripts.final_prediction_export import predict_export, save_prediction_artifact, displacement_metrics
from src.data.eth_ucy_social_final_dataset import ETHUCYSocialFinalDataset


@pytest.mark.parametrize('kind', ('ett', 'emt'))
def test_attention_export_preserves_nonzero_prediction_and_backbone(kind):
    torch.set_num_threads(1)
    base, _ = make_model(kind, 42)
    model = SocialResidualPredictor(base.cuda(), kind+'_sr', 42).cuda().eval()
    before = state_hash(model.backbone.state_dict())
    with torch.no_grad():
        torch.nn.init.normal_(model.residual[-1].weight, std=.01)
    batch = {'target_history': torch.randn(3, 8, 4, device='cuda'), 'neighbor_history': torch.randn(3, 8, 8, 4, device='cuda'), 'neighbor_relation': torch.randn(3, 8, 3, device='cuda'), 'neighbor_mask': torch.ones(3, 8, dtype=torch.bool, device='cuda')}
    batch['neighbor_mask'][0] = False
    batch['neighbor_mask'][1, 2:] = False
    a = predict_export(model, batch, True, False)
    b = predict_export(model, batch, True, True)
    assert a['future_pred'].shape == (3, 12, 2)
    assert float((a['future_pred']-b['future_pred']).abs().max()) < 1e-7
    assert not torch.equal(a['future_pred'], a['base_prediction'])
    weights = b['attention_weights']
    assert torch.all(weights[~batch['neighbor_mask']] == 0)
    torch.testing.assert_close(weights.sum(1), torch.tensor([0., 1., 1.], device='cuda'))
    assert before == state_hash(model.backbone.state_dict())
    assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())


@pytest.mark.parametrize('kind', ('ett', 'emt'))
def test_final_social_zero_init(kind):
    base, _ = make_model(kind, 123)
    model = SocialResidualPredictor(base.cuda(), kind+'_sr', 123).cuda().train()
    x = torch.randn(2, 8, 4, device='cuda')
    n = torch.randn(2, 8, 8, 4, device='cuda')
    mask = torch.ones(2, 8, dtype=torch.bool, device='cuda')
    r = torch.randn(2, 8, 3, device='cuda')
    out = model(x, n, mask, r)
    assert torch.equal(out['future_pred'], out['base_prediction'])
    out['future_pred'].square().mean().backward()
    assert not model.backbone.training and all(p.grad is None for p in model.backbone.parameters())


def test_source_prediction_artifact_recomputes_metrics(tmp_path):
    dataset = ETHUCYSocialFinalDataset('eth', 'final_train')
    class Small:
        audit = {k: v[:3] for k, v in dataset.audit.items()}
        arrays = {k: v[:3] for k, v in dataset.arrays.items()}
    target = Small.arrays['future_target'].numpy()
    out = {'future_pred': target+np.float32(.2), 'base_prediction': target.copy(), 'residual': np.full_like(target, .2), 'attention_weights': np.zeros((3, 8), dtype=np.float32)}
    path = tmp_path/'predictions.npz'
    metrics = save_prediction_artifact(path, Small, [out], True)
    with np.load(path, allow_pickle=False) as z:
        ade, fde = displacement_metrics(z['prediction_abs'], z['future_gt'])
        assert float(ade.mean()) == metrics['ADE'] and float(fde.mean()) == metrics['FDE']
        np.testing.assert_array_equal(ade, z['sample_ADE'])
        assert z['obs_abs'].shape == (3, 8, 2) and z['neighbor_abs'].shape == (3, 8, 8, 2)
        assert set(z['scene_id']).isdisjoint({'eth'})
