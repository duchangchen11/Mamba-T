"""Frozen source-validation forwards; attention observation never alters predictions."""
import numpy as np
import torch
from src.data.eth_ucy_training_regime_dataset import TrainingRegimeSourceDataset
from src.models.social_residual import SocialResidualPredictor
from src.analysis.social_shift_features import observation_features, attention_features, response_features, SUMMARY_FEATURES
from scripts.eth_ucy_utils import make_model, state_hash
from scripts.social_residual_utils import cache_contexts, forward_cached, restore_new_state
from scripts.social_shift_protocol import ROOT, RESULTS, PREVIOUS, guard_path, read, sha_file, dump, require_validation


@torch.no_grad()
def observe_attention(social, target, neighbor, relation, mask):
    """Second eval-only MHA call returns weights; original need_weights=False path stays intact."""
    assert not social.training and not torch.is_grad_enabled()
    valid = mask.bool();has = valid.any(1)
    heads = torch.zeros((len(target),social.attention.num_heads,8),device=target.device)
    context = torch.zeros_like(target)
    if has.any():
        m=valid[has]
        n=neighbor[has].masked_fill(~m[...,None],0)
        r=relation[has].masked_fill(~m[...,None],0)
        kv=n+social.relation_embedding(r)
        c,w=social.attention(target[has,None],kv,kv,key_padding_mask=~m,need_weights=True,average_attn_weights=False)
        heads[has]=w[:,:,0]
        context[has]=c[:,0]
    return heads,context


def load_npz(path):
    with np.load(guard_path(path),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


@torch.no_grad()
def extract_run(fold,model_name,seed,mode='val'):
    require_validation(mode)
    folder=PREVIOUS/f'heldout_{fold}'/model_name/f'seed_{seed}'
    previous=read(folder/'metrics_validation.json')
    assert not previous['heldout_test_accessed'] and previous['historical_test_already_accessed']
    saved=load_npz(folder/'validation_samples.npz')
    base_name=model_name.split('_')[0]
    base_report=read(PREVIOUS/f'heldout_{fold}'/base_name/f'seed_{seed}/metrics_validation.json')
    base_checkpoint=guard_path(ROOT/base_report['checkpoint_path'])
    sr_checkpoint=guard_path(ROOT/previous['checkpoint_path'])
    assert str(base_checkpoint).startswith(str(ROOT/'checkpoints/training_regime_diagnostic')+'/')
    assert str(sr_checkpoint).startswith(str(ROOT/'checkpoints/training_regime_diagnostic')+'/')
    assert sha_file(base_checkpoint)==base_report['checkpoint_sha256']==previous['base_checkpoint_sha256']
    assert sha_file(sr_checkpoint)==previous['checkpoint_sha256']
    base_state=torch.load(base_checkpoint,map_location='cpu',weights_only=True)
    sr_state=torch.load(sr_checkpoint,map_location='cpu',weights_only=True)
    backbone,_=make_model(base_name,seed)
    backbone.load_state_dict(base_state['model']);backbone=backbone.cuda().eval()
    model=SocialResidualPredictor(backbone,model_name,seed).cuda()
    restore_new_state(model,sr_state['new_modules']);model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before=state_hash(model.state_dict())
    assert before==previous['final_model_state_sha256']
    dataset=TrainingRegimeSourceDataset(fold,'val',True)
    assert fold not in dataset.scenes
    for key,actual in (('scene_id',dataset.audit['scene_id']),('target_ped_id',dataset.audit['target_ped_id']),('frame_ids',dataset.audit['frame_ids'])):
        np.testing.assert_array_equal(actual,saved[key])
    obs=observation_features(dataset.arrays['target_history'].numpy(),dataset.arrays['neighbor_history'].numpy(),dataset.arrays['neighbor_relation'].numpy(),dataset.arrays['neighbor_mask'].numpy())
    np.testing.assert_array_equal(obs['neighbor_count'],saved['neighbor_count'])
    np.testing.assert_allclose(obs['mean_neighbor_distance'][obs['neighbor_count']>0],saved['mean_neighbor_distance'][obs['neighbor_count']>0],rtol=1e-6,atol=1e-6)
    cache=cache_contexts(backbone,dataset)
    residuals=[];weights=[];predictions=[];base_predictions=[];context_difference=0.
    for i in range(0,len(dataset),128):
        s=slice(i,i+128)
        original=forward_cached(model,cache,s)
        head,context=observe_attention(model.social,cache['target_context'][s],cache['neighbor_context'][s],cache['neighbor_relation'][s],cache['neighbor_mask'][s])
        context_difference=max(context_difference,float((context-original['social_context']).abs().max()))
        assert context_difference<1e-5
        residuals.append(original['residual'].cpu().numpy());weights.append(head.cpu().numpy())
        predictions.append(original['future_pred'].cpu().numpy());base_predictions.append(original['base_prediction'].cpu().numpy())
    residual,heads,pred,base = map(np.concatenate,(residuals,weights,predictions,base_predictions))
    attention,alpha=attention_features(heads,dataset.arrays['neighbor_mask'].numpy())
    responses=response_features(residual,base,saved['base_ADE'],saved['sample_ADE'],saved['base_FDE'],saved['sample_FDE'])
    reproduction={}
    for name in ('residual_norm','correction_ratio','base_trajectory_norm'):
        reproduction[name]=float(np.max(np.abs(responses[name]-saved[name])))
        np.testing.assert_allclose(responses[name],saved[name],rtol=1e-5,atol=1e-6)
        responses[name]=saved[name].copy()
    last=dataset.arrays['last_obs_pos'].numpy();future=dataset.arrays['future_abs'].numpy()
    for label,prediction in (('base',base),('SR',pred)):
        d=np.linalg.norm((prediction+last[:,None]).astype(np.float64)-future.astype(np.float64),axis=-1)
        for metric,value in (('ADE',d.mean(1)),('FDE',d[:,-1])):
            original=saved[f'base_{metric}' if label=='base' else f'sample_{metric}']
            reproduction[f'{label}_{metric}']=float(np.max(np.abs(value-original)))
            np.testing.assert_allclose(value,original,rtol=1e-5,atol=2e-6)
    after=state_hash(model.state_dict())
    assert before==after and not model.training and all(p.grad is None and not p.requires_grad for p in model.parameters())
    frames=dataset.audit['frame_ids'][:,:8]
    uids=np.array([str(scene)+'|'+str(ped)+'|'+','.join(map(str,frame)) for scene,ped,frame in zip(dataset.audit['scene_id'],dataset.audit['target_ped_id'],frames)])
    samples={'scene':dataset.audit['scene_id'],'fold':np.full(len(dataset),fold),'seed':np.full(len(dataset),seed,dtype=np.int64),'ped_id':dataset.audit['target_ped_id'],'sample_uid':uids,'observation_frame_ids':frames,**obs,**attention,**responses,'attention_weights':alpha,'attention_head_weights':heads}
    for name in SUMMARY_FEATURES:
        allowed_missing=name in ('nearest_neighbor_distance','mean_neighbor_distance','median_neighbor_distance','max_neighbor_distance','mean_relative_speed','max_relative_speed','mean_signed_closing_speed','max_signed_closing_speed')
        values=samples[name]
        assert not np.isinf(values).any()
        assert np.all(np.isnan(values)==(obs['neighbor_count']==0)) if allowed_missing else np.isfinite(values).all()
    destination=RESULTS/model_name/f'heldout_{fold}'/f'seed_{seed}'
    destination.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination/'samples.npz',**samples)
    audit={'fold':fold,'model':model_name,'seed':seed,'sample_count':len(dataset),'source_scenes':sorted(dataset.scenes),'model_state_sha256_before':before,'model_state_sha256_after':after,'weights_unchanged':True,'eval_no_grad':True,'optimizer_created':False,'trajectory_training':False,'attention_context_max_abs_diff':context_difference,'source_validation_reproduction_max_abs_diff':reproduction,'sample_sha256':sha_file(destination/'samples.npz'),'base_checkpoint_sha256':sha_file(base_checkpoint),'SR_checkpoint_sha256':sha_file(sr_checkpoint),'heldout_test_accessed':False,'historical_test_already_accessed':True}
    dump(destination/'forward_audit.json',audit)
    print(f'SOURCE FORWARD {fold} {model_name} seed={seed} n={len(dataset)} attention_context_diff={context_difference:.3g}',flush=True)
    del cache,model,backbone,dataset
    torch.cuda.empty_cache()
    return samples,audit
