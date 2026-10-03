import torch
from src.models.social_attention import SocialCrossAttention


def test_mask_and_zero_neighbor():
    m=SocialCrossAttention().eval();h=torch.randn(3,128);n=torch.randn(3,8,128);r=torch.randn(3,8,3)
    mask=torch.zeros(3,8,dtype=torch.bool);mask[1,:2]=True;mask[2,:]=True
    a=m(h,n,r,mask)
    assert torch.isfinite(a).all() and torch.equal(a[0],torch.zeros(128))
    n[~mask]=float('nan');r[~mask]=float('nan')
    b=m(h,n,r,mask)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    loss=b.square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_all_zero_neighbors():
    m=SocialCrossAttention();h=torch.randn(4,128,requires_grad=True)
    y=m(h,torch.full((4,8,128),float('nan')),torch.full((4,8,3),float('nan')),torch.zeros(4,8,dtype=torch.bool))
    assert torch.equal(y,torch.zeros_like(y));y.sum().backward();assert torch.equal(h.grad,torch.zeros_like(h))
