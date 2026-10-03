from pathlib import Path
import numpy as np
import pytest
import torch
from src.analysis.social_shift_features import observation_features,attention_features,response_features,group_masks,OBS_FEATURES
from scripts.social_shift_protocol import guard_path
from scripts.social_shift_statistics import scene_weights,correlate,collapse_windows,group_summary,pairwise_shift,scene_classifier
from scripts.eth_ucy_utils import state_hash


@pytest.mark.parametrize('path',['/tmp/results/final_eth_ucy_benchmark/predictions/x.npz','/tmp/a/../final_eth_ucy_benchmark/predictions/x.npz','C:\\results\\final_eth_ucy_benchmark\\predictions\\x.npz'])
def test_formal_prediction_guard_before_open(path,monkeypatch):
    from scripts.social_shift_protocol import read
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('Attempted formal file open'))
    with pytest.raises(RuntimeError):read(path)


def test_formal_guard_symlink(tmp_path):
    target=tmp_path/'final_eth_ucy_benchmark/predictions';target.mkdir(parents=True)
    p=target/'x.npz';p.write_bytes(b'forbidden')
    link=tmp_path/'alias';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(RuntimeError):guard_path(link/'x.npz')


@pytest.mark.parametrize('mode',['heldout_test','final_train','train'])
def test_run_rejects_nonvalidation_before_inputs(mode,monkeypatch):
    from scripts import run_social_distribution_shift as runner
    monkeypatch.setattr(runner,'freeze_protocol',lambda:pytest.fail('Inputs opened for forbidden mode'))
    with pytest.raises(PermissionError):runner.main(mode)


def test_observation_motion_closing_sign_and_no_future_inputs():
    h=np.zeros((3,8,4));n=np.zeros((3,8,8,4));r=np.zeros((3,8,3));mask=np.zeros((3,8),bool)
    mask[:2,0]=True;r[:2,0]=[2.,0.,2.]
    n[0,0,-2,0]=1.;n[1,0,-2,0]=-1.
    q=observation_features(h,n,r,mask)
    np.testing.assert_array_equal(q['neighbor_count'],[1,1,0])
    np.testing.assert_allclose(q['mean_relative_speed'][:2],[1.,1.])
    assert q['mean_signed_closing_speed'][0]>0 and q['mean_signed_closing_speed'][1]<0
    np.testing.assert_allclose(q['mean_closing_speed'],[2/(2+1e-8),0,0])
    assert q['max_closing_speed'][1]==0 and np.isnan(q['nearest_neighbor_distance'][2])
    # The extractor signature has only observation histories, relations, and masks.
    import inspect
    assert 'future' not in ' '.join(inspect.signature(observation_features).parameters)
    padded=n.copy();padded[:,1:]=1e8;bad_relation=r.copy();bad_relation[:,1:]=1e8
    other=observation_features(h,padded,bad_relation,mask)
    for key in q:np.testing.assert_allclose(q[key],other[key],equal_nan=True)


def test_distances_rank_padding_and_per_step_relative_speed():
    h=np.zeros((1,8,4));n=np.zeros((1,8,8,4));r=np.zeros((1,8,3));mask=np.zeros((1,8),bool)
    mask[0,[0,3,7]]=True;r[0,[0,3,7],0]=[8,2,4];r[0,[0,3,7],2]=[8,2,4]
    h[0,-2,0]=-.5;n[0,[0,3,7],-2,0]=-1.5
    q=observation_features(h,n,r,mask)
    assert q['neighbor_count'][0]==3 and q['nearest_neighbor_distance'][0]==2
    assert q['median_neighbor_distance'][0]==4 and q['max_neighbor_distance'][0]==8
    assert q['mean_neighbor_distance'][0]==pytest.approx(14/3)
    assert q['mean_relative_speed'][0]==1.
    assert [q[f'rank{i}_neighbor_distance'][0] for i in (1,2,3)]==[2,4,8]
    assert np.isnan(q['rank4_neighbor_distance'][0])


def test_attention_normalization_entropy_and_padding():
    mask=np.zeros((3,8),bool);mask[1,0]=True;mask[2,:4]=True
    weights=np.ones((3,4,8));weights[:,:,4:]=100
    q,alpha=attention_features(weights,mask)
    assert q['attention_entropy'][0]==q['attention_entropy'][1]==0
    assert q['attention_entropy'][2]==pytest.approx(1.,abs=1e-7)
    assert np.all((q['attention_entropy']>=0)&(q['attention_entropy']<=1))
    np.testing.assert_allclose(q['attention_max'],[0,1,.25])
    np.testing.assert_allclose(q['attention_top2_sum'],[0,1,.5])
    assert np.all(alpha[~mask]==0)


def test_attention_observer_preserves_module_and_matches_context():
    from src.models.social_attention import SocialCrossAttention
    from scripts.social_shift_inference import observe_attention
    torch.manual_seed(11);module=SocialCrossAttention().eval()
    target=torch.randn(3,128);neighbor=torch.randn(3,8,128);relation=torch.randn(3,8,3)
    mask=torch.zeros(3,8,dtype=torch.bool);mask[1,0]=True;mask[2]=True
    before=state_hash(module.state_dict())
    with torch.no_grad():original=module(target,neighbor,relation,mask)
    head,observed=observe_attention(module,target,neighbor,relation,mask)
    torch.testing.assert_close(observed,original,rtol=1e-5,atol=1e-6)
    assert head.shape==(3,4,8) and not head.requires_grad
    assert state_hash(module.state_dict())==before and all(p.grad is None for p in module.parameters())


def test_residual_final_ratio_and_gain_sign():
    residual=np.zeros((2,12,2));base=np.zeros_like(residual)
    residual[0]=[3,4];residual[1,-1]=[0,12];base[0]=[6,8]
    q=response_features(residual,base,np.array([2,1]),np.array([1,2]),np.array([4,3]),np.array([3,5]))
    np.testing.assert_allclose(q['residual_norm'],[5,1]);np.testing.assert_allclose(q['residual_final_norm'],[5,12])
    np.testing.assert_array_equal(q['ADE_gain'],[1,-1]);np.testing.assert_array_equal(q['FDE_gain'],[1,-2])
    assert q['correction_ratio'][1]==1e8


def test_scene_equal_weights_and_weighted_correlations():
    samples={'scene':np.array(['eth']+['univ']*9),'x':np.arange(10,dtype=float),'y':np.arange(10,dtype=float),'base_ADE':np.array([1.]+[3.]*9),'prediction_count':np.ones(10,dtype=int)}
    w=scene_weights(samples['scene'])
    assert w[0]==1 and w[1:].sum()==pytest.approx(1)
    for f in ('SR_ADE','ADE_gain','base_FDE','SR_FDE','FDE_gain','residual_norm','residual_final_norm','correction_ratio','attention_entropy','attention_max'):samples[f]=samples['base_ADE']
    assert group_summary(samples,np.ones(10,bool),True)['base_ADE']==pytest.approx(2)
    assert group_summary(samples,np.ones(10,bool),False)['base_ADE']==pytest.approx(2.8)
    c=correlate(samples,'x','y',True);assert c['Pearson']==pytest.approx(1) and c['Spearman']==pytest.approx(1)
    samples['y']=np.ones(10);assert correlate(samples,'x','y',True)['Pearson'] is None


def test_duplicate_windows_collapsed_and_observation_consistency():
    raw={'sample_uid':np.array(['a','a','b']),'scene':np.array(['eth']*3),'ped_id':np.array(['p','p','q']),'observation_frame_ids':np.tile(np.arange(8),(3,1)),'seed':np.array([42,123,42]),'residual_norm':np.array([1.,3.,4.])}
    for f in OBS_FEATURES:raw[f]=np.array([2.,2.,3.])
    q=collapse_windows(raw);np.testing.assert_array_equal(q['prediction_count'],[2,1]);np.testing.assert_array_equal(q['residual_norm'],[2.,4.])
    raw['neighbor_count'][1]=4
    with pytest.raises(AssertionError):collapse_windows(raw)


def test_group_boundaries_and_tied_closing_quartiles():
    samples={'neighbor_count':np.arange(9),'nearest_neighbor_distance':np.array([np.nan,1.9,2,3.9,4,8,8.1,1,1]),'max_closing_speed':np.zeros(9)}
    groups=group_masks(samples,[0,0,0])
    assert [v.sum() for v in groups['neighbor_count'].values()]==[1,2,2,2,2]
    assert [v.sum() for v in groups['nearest_distance'].values()]==[1,3,2,2,1]
    assert groups['closing_speed']['Q1'].sum()==9 and groups['closing_speed']['Q2'].sum()==0
    assert groups['saturation']['N=8'].sum()==1


def test_pairwise_smd_population_std_and_ten_equal_pairs():
    from scripts.social_shift_protocol import SCENES
    from src.analysis.social_shift_features import SHIFT_FEATURES
    samples={'scene':np.repeat(SCENES,2)}
    values=np.repeat(np.arange(5),2)+np.tile([0.,2.],5)
    for f in SHIFT_FEATURES+('ADE_gain','FDE_gain'):samples[f]=values
    q=pairwise_shift(samples)
    assert len(q['pairs'])==10
    first=q['pairs'][0]['features']['neighbor_count']
    assert first['mean_difference']==-1 and first['median_difference']==-1
    assert first['abs_SMD']==1 and first['Wasserstein']==1
    assert q['ranking'][0]['mean_abs_SMD']==2 and q['ranking'][0]['max_abs_SMD']==4


def test_classifier_cv_pedestrians_together_and_input_whitelist():
    from scripts.social_shift_protocol import SCENES,read,ROOT
    cfg=read(ROOT/'configs/social_distribution_shift.json')['classifier']
    scene=np.repeat(SCENES,12);ped=np.tile(np.repeat(np.arange(6),2),5).astype(str)
    data={'scene':scene,'ped_id':ped,'sample_uid':np.array([f'u{i}' for i in range(60)])}
    for j,f in enumerate(cfg['features']):data[f]=np.repeat(np.arange(5),12).astype(float)+np.tile(np.arange(12),5)*(.01+j*.002)
    result,pred=scene_classifier(data,cfg)
    assert result['executed'] and len(result['folds'])==5 and all(r['pedestrian_overlap']==0 for r in result['folds'])
    for s in SCENES:
        for p in np.unique(ped):assert len(np.unique(pred['cv_fold'][(scene==s)&(ped==p)]))==1
    bad={**cfg,'features':[f for f in cfg['features'] if f!='neighbor_count']+['future_trajectory']}
    with pytest.raises(AssertionError):scene_classifier(data,bad)
