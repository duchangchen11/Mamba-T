"""Frozen-context training. No held-out evaluation path; source split only."""
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from scripts.eth_ucy_utils import make_model,seed_all,state_hash
from src.data.eth_ucy_social_dataset import ROOT,SCENES,ETHUCYSocialDataset,BASE_RESULTS
from src.models.social_residual import SocialResidualPredictor,VARIANTS
RESULTS=ROOT/'results/social_residual'
CONFIG=json.loads((ROOT/'configs/social_residual.json').read_text())

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sha_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_backbone(kind,fold,seed):
    path=ROOT/'checkpoints/eth_ucy_mamba_baseline'/f'heldout_{fold}'/kind/f'seed_{seed}.pt'
    inventory=json.loads((RESULTS/'data_audit/stage1_freeze_inventory.json').read_text())
    expected=inventory[str(path.relative_to(ROOT))];actual=sha_file(path)
    if expected!=actual:raise ValueError('Frozen stage-one checkpoint modified')
    b,_=make_model(kind,seed);ckpt=torch.load(path,map_location='cpu',weights_only=True);b.load_state_dict(ckpt['model'])
    b=b.cuda().eval()
    for p in b.parameters():p.requires_grad_(False)
    return b,{'base_checkpoint':str(path),'base_checkpoint_sha256':actual,'base_checkpoint_epoch':ckpt['epoch'],'base_state_sha256':state_hash(b.state_dict()),'backbone_frozen':True,'backbone_eval':True}

@torch.no_grad()
def cache_contexts(backbone,dataset):
    source=dataset.arrays;out={}
    # Target batch=128 and order exactly match stage-one validation inference.
    h=[];pred=[]
    for i in range(0,len(dataset),128):
        x=source['target_history'][i:i+128].cuda();_,context=backbone.encode(x)
        h.append(context);pred.append(backbone.decoder(context).reshape(-1,12,2))
    out['target_context']=torch.cat(h);out['base_prediction']=torch.cat(pred)
    mask=source['neighbor_mask'].cuda();out['neighbor_mask']=mask
    n=torch.zeros((len(dataset),8,128),device='cuda')
    neighbor_x=source['neighbor_history'][source['neighbor_mask']]
    encoded=[]
    for i in range(0,len(neighbor_x),512):
        _,context=backbone.encode(neighbor_x[i:i+512].cuda());encoded.append(context)
    if encoded:n[mask]=torch.cat(encoded)
    out['neighbor_context']=n
    for k in ('neighbor_relation','future_target','last_obs_pos','future_abs'):out[k]=source[k].cuda()
    return out

def forward_cached(model,data,index):
    return model.forward_context(data['target_context'][index],data['neighbor_context'][index],data['neighbor_relation'][index],data['neighbor_mask'][index],data['base_prediction'][index])

BUCKETS={'0':lambda n:n==0,'1-2':lambda n:(n>=1)&(n<=2),'3-4':lambda n:(n>=3)&(n<=4),'5+':lambda n:n>=5}

def metric_report(ade,fde,counts,gates=None):
    report={'ADE':float(ade.mean()),'FDE':float(fde.mean()),'sample_count':len(ade),'neighbor_groups':{}}
    for name,rule in BUCKETS.items():
        selected=rule(counts);num=int(selected.sum())
        report['neighbor_groups'][name]={'sample_count':num,'ADE':float(ade[selected].mean()) if num else None,'FDE':float(fde[selected].mean()) if num else None}
    if gates is not None:
        report['gate_statistics']={'gate_mean':float(gates.mean()),'gate_std':float(gates.std(ddof=0)),'gate_p10':float(np.quantile(gates,.1)),'gate_p50':float(np.quantile(gates,.5)),'gate_p90':float(np.quantile(gates,.9)),'zero_neighbor_gate_mean':float(gates[counts==0].mean()) if (counts==0).any() else None,'has_neighbor_gate_mean':float(gates[counts>0].mean()) if (counts>0).any() else None,'std_definition':'population SD across validation samples'}
    return report

@torch.no_grad()
def evaluate_cached(model,data):
    model.eval();ades=[];fdes=[];gates=[]
    for i in range(0,len(data['target_context']),128):
        index=slice(i,i+128);out=forward_cached(model,data,index)
        pred=out['future_pred']+data['last_obs_pos'][index,None]
        d=torch.linalg.vector_norm(pred-data['future_abs'][index],dim=-1)
        if not torch.isfinite(d).all() or not torch.isfinite(out['gate']).all():raise FloatingPointError('nonfinite validation')
        ades.append(d.mean(1).cpu().numpy());fdes.append(d[:,-1].cpu().numpy());gates.append(out['gate'][:,0].cpu().numpy())
    report=metric_report(np.concatenate(ades).astype(np.float64),np.concatenate(fdes).astype(np.float64),data['neighbor_mask'].sum(1).cpu().numpy(),np.concatenate(gates).astype(np.float64) if model.gate is not None else None)
    return report

@torch.no_grad()
def baseline_report(data):
    d=torch.linalg.vector_norm(data['base_prediction']+data['last_obs_pos'][:,None]-data['future_abs'],dim=-1)
    return metric_report(d.mean(1).cpu().numpy().astype(np.float64),d[:,-1].cpu().numpy().astype(np.float64),data['neighbor_mask'].sum(1).cpu().numpy())

def parameter_report(model):
    return {'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad)}

def new_state(model):return {k:v for k,v in model.state_dict().items() if not k.startswith('backbone.')}

def restore_new_state(model,state):
    missing,extra=model.load_state_dict(state,strict=False)
    assert not extra and all(k.startswith('backbone.') for k in missing)

def train_variant(variant,fold,seed,backbone,metadata,train_data,val_data,epochs=50,smoke=False):
    model=SocialResidualPredictor(backbone,variant,seed).cuda();seed_all(seed)
    folder=RESULTS/('smoke' if smoke else f'heldout_{fold}')/variant/f'seed_{seed}'
    checkpoint=ROOT/'checkpoints/social_residual'/('smoke' if smoke else f'heldout_{fold}')/variant/f'seed_{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    init={**metadata,'variant':variant,'seed':seed,'residual_initialization_sha256':state_hash(model.residual.state_dict()),'trainable_initialization_sha256':state_hash(new_state(model))}
    model.eval()
    with torch.no_grad():
        output=forward_cached(model,val_data,slice(0,128))
        difference=float((output['future_pred']-val_data['base_prediction'][:128]).abs().max())
    assert difference<1e-7;init['initial_output_base_max_abs_diff']=difference
    dump(folder/'initialization_report.json',init);dump(folder/'parameter_count.json',parameter_report(model))
    manifest=BASE_RESULTS/'data_audit'/f'manifest_{fold}.json'
    dump(folder/'data_provenance.json',{'stage1_manifest_sha256':sha_file(manifest),'heldout_scene':fold,'heldout_test_accessed':False,'data_provenance':'results/social_residual/data_audit/data_provenance.json',**metadata})
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=4)
    criterion=nn.SmoothL1Loss(beta=1)
    generator=torch.Generator().manual_seed(seed)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();history=[];best=float('inf');bad=0;max_gradient=0.;best_epoch=None
    n=len(train_data['target_context'])
    for epoch in range(1,epochs+1):
        model.train();total=0.
        order=torch.randperm(n,generator=generator).cuda()
        for i in range(0,n,128):
            index=order[i:i+128];optimizer.zero_grad(set_to_none=True)
            out=forward_cached(model,train_data,index);loss=criterion(out['future_pred'],train_data['future_target'][index])
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();norm=nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);max_gradient=max(max_gradient,float(norm));optimizer.step()
            total+=float(loss)*len(index)
        metrics=evaluate_cached(model,val_data);scheduler.step(metrics['ADE'])
        history.append({'epoch':epoch,'train_loss':total/n,'validation_ADE':metrics['ADE'],'validation_FDE':metrics['FDE'],'lr':optimizer.param_groups[0]['lr']})
        if metrics['ADE']<best:
            best=metrics['ADE'];bad=0;best_epoch=epoch;torch.save({'new_modules':new_state(model),'epoch':epoch,'base_checkpoint_sha256':metadata['base_checkpoint_sha256'],'heldout_test_accessed':False},checkpoint)
        else:bad+=1
        dump(folder/'validation_history.json',history)
        if epoch==1 or epoch%10==0:print(f'{fold} {variant} {seed} epoch {epoch}: loss={total/n:.6f} ADE={metrics["ADE"]:.6f} FDE={metrics["FDE"]:.6f}',flush=True)
        if bad>=8:break
    torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
    assert state_hash(backbone.state_dict())==metadata['base_state_sha256']
    restore_new_state(model,torch.load(checkpoint,weights_only=True)['new_modules'])
    metrics=evaluate_cached(model,val_data)
    assert abs(metrics['ADE']-best)<1e-7
    report={'heldout_scene':fold,'model':variant,'seed':seed,'train_sample_count':n,'validation_sample_count':len(val_data['target_context']),'best_epoch':best_epoch,'epochs_run':len(history),'validation_ADE':metrics['ADE'],'validation_FDE':metrics['FDE'],'neighbor_groups':metrics['neighbor_groups'],'maximum_gradient_norm_before_clipping':max_gradient,'training_seconds':seconds,'gpu_peak_memory_bytes':peak,'parameter_count':parameter_report(model),'backbone_unchanged':True,'initial_output_base_max_abs_diff':difference,'nan_inf':False,'heldout_test_accessed':False,'smoke':smoke,'latency':None,'protocol':CONFIG,'training_method':'frozen eval context cache, GPU-resident; seed-fixed shuffled sample batches; gradients only through new modules',**metadata}
    if 'gate_statistics' in metrics:
        report['gate_statistics']=metrics['gate_statistics'];dump(folder/'gate_statistics.json',metrics['gate_statistics'])
    dump(folder/'metrics_validation.json',report)
    return report


def prepare_pair(kind,fold,seed):
    backbone,metadata=load_backbone(kind,fold,seed)
    train=ETHUCYSocialDataset(fold,'train');val=ETHUCYSocialDataset(fold,'val')
    assert not train.tracks&val.tracks and fold not in train.scenes|val.scenes
    start=time.perf_counter();tr=cache_contexts(backbone,train);va=cache_contexts(backbone,val)
    base=baseline_report(va)
    original=json.loads((BASE_RESULTS/f'heldout_{fold}'/kind/f'seed_{seed}/metrics_validation.json').read_text())
    differences={m:base[m]-original[f'validation_{m}'] for m in ('ADE','FDE')}
    assert all(abs(v)<1e-6 for v in differences.values())
    metadata.update({'context_cache_seconds':time.perf_counter()-start,'train_pedestrian_count':len(train.tracks),'validation_pedestrian_count':len(val.tracks),'baseline_validation_difference':differences})
    dump(RESULTS/'data_audit/base_validation'/f'{fold}_{kind}_{seed}.json',{'stage1_metrics':{m:original[f'validation_{m}'] for m in ('ADE','FDE')},'reproduced_validation':base,**metadata,'heldout_test_accessed':False})
    return backbone,metadata,tr,va

@torch.no_grad()
def benchmark_full(model):
    model.eval();out={}
    for batch in (1,128):
        x=torch.zeros(batch,8,4,device='cuda');n=torch.zeros(batch,8,8,4,device='cuda');r=torch.zeros(batch,8,3,device='cuda')
        for count in (0,8):
            mask=torch.zeros(batch,8,dtype=torch.bool,device='cuda');mask[:,:count]=True
            torch.cuda.reset_peak_memory_stats()
            for _ in range(20):model(x,n,mask,r)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(100):model(x,n,mask,r)
            torch.cuda.synchronize()
            out[f'batch_{batch}_neighbors_{count}_latency_ms']=(time.perf_counter()-start)*10
            out[f'batch_{batch}_neighbors_{count}_peak_memory_bytes']=torch.cuda.max_memory_allocated()
    out['method']='full uncached model incl shared frozen temporal encoder; exclusive GPU, zeros, eval/no_grad; 20 warmup +100 synchronized forwards; no data transfer'
    return out
