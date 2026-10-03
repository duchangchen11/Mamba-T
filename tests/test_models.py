import pytest
import torch
from scripts.eth_ucy_utils import make_model

@pytest.mark.parametrize('seed',[42,123,2024])
def test_decoder_matching(seed):
    a,ha=make_model('ett',seed);b,hb=make_model('emt',seed)
    assert ha==hb
    for key,value in a.decoder.state_dict().items():assert torch.equal(value,b.decoder.state_dict()[key])

@pytest.mark.parametrize('name',['ett','emt'])
def test_cuda_backward(name):
    m,_=make_model(name,42);m=m.cuda()
    y=m(torch.randn(2,8,4,device='cuda'))['future_pred']
    assert y.shape==(2,12,2) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
