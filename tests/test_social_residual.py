import pytest,torch
from scripts.eth_ucy_utils import make_model
from src.models.social_residual import SocialResidualPredictor,VARIANTS

@pytest.mark.parametrize('name',['ett','emt'])
def test_encode_exact_legacy_predictions(name):
    torch.set_num_threads(1);m,_=make_model(name,42)
    from src.data.eth_ucy_dataset import ROOT
    ckpt=ROOT/'checkpoints/eth_ucy_mamba_baseline/heldout_eth'/name/'seed_42.pt'
    m.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['model'])
    m=m.cuda().eval();x=torch.randn(5,8,4,device='cuda')
    with torch.no_grad():
        if name=='ett':old=m.temporal_encoder(m.input_projection(x)+m.position_embedding[:,:8])[:,-1]
        else:_,old=m.encoder(x)
        pred=m.decoder(old).reshape(5,12,2);seq,context=m.encode(x);new=m(x)['future_pred']
    assert seq.shape==(5,8,128) and context.shape==(5,128)
    assert torch.equal(pred,new)

@pytest.mark.parametrize('variant',VARIANTS)
def test_frozen_zero_initialization_and_gradients(variant):
    b,_=make_model('ett' if variant=='ett_sr' else 'emt',42);b=b.cuda().eval()
    m=SocialResidualPredictor(b,variant,42).cuda().train()
    assert not m.backbone.training and not any(p.requires_grad for p in b.parameters())
    x=torch.randn(3,8,4,device='cuda');n=torch.randn(3,8,8,4,device='cuda');mask=torch.ones(3,8,device='cuda',dtype=torch.bool);mask[0]=False;r=torch.randn(3,8,3,device='cuda')
    with torch.no_grad():base=b(x)['future_pred']
    out=m(x,n,mask,r)
    assert torch.equal(out['future_pred'],base) and torch.isfinite(out['future_pred']).all()
    assert not out['base_prediction'].requires_grad
    if m.gate is not None:assert torch.all((out['gate']>=0)&(out['gate']<=1))
    # Zero last layer blocks upstream gradients on step one; after one update, new modules receive them.
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
    loss=(out['future_pred']-1).square().mean();loss.backward();optimizer.step();optimizer.zero_grad()
    out=m(x,n,mask,r);out['future_pred'].square().mean().backward()
    assert all(p.grad is None for p in b.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.residual.parameters())
    if m.social is not None:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.social.parameters())
    if m.gate is not None:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.gate.parameters())

@pytest.mark.parametrize('name',['ett','emt'])
def test_cached_context_matches_full_model(name):
    from scripts.social_residual_utils import cache_contexts
    from src.data.eth_ucy_social_dataset import ETHUCYSocialDataset
    d=ETHUCYSocialDataset('eth','val')
    class SmallSource:
        arrays={k:v[:4] for k,v in d.arrays.items()}
        def __len__(self):return 4
    b,_=make_model(name,42);b=b.cuda().eval()
    for p in b.parameters():p.requires_grad_(False)
    m=SocialResidualPredictor(b,'ett_sr' if name=='ett' else 'emt_sr',42).cuda().eval()
    with torch.no_grad():
        torch.nn.init.normal_(m.residual[-1].weight,std=.01)
        cached=cache_contexts(b,SmallSource())
        a=SmallSource.arrays
        full=m(a['target_history'].cuda(),a['neighbor_history'].cuda(),a['neighbor_mask'].cuda(),a['neighbor_relation'].cuda())
        from_cache=m.forward_context(cached['target_context'],cached['neighbor_context'],cached['neighbor_relation'],cached['neighbor_mask'],cached['base_prediction'])
    torch.testing.assert_close(full['future_pred'],from_cache['future_pred'],rtol=0,atol=1e-7)
