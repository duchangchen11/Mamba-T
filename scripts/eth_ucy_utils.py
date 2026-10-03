"""Shared matched training protocol. Held-out data has no evaluation entry point."""
import argparse, copy, hashlib, json, random, time, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data.eth_ucy_dataset import ROOT, SCENES, ETHUCYDataset
from src.models.trajectory_transformer import TargetOnlyTrajectoryTransformer, make_trajectory_decoder
from src.models.mamba_trajectory import MambaTrajectoryPredictor
RESULTS=ROOT/'results/eth_ucy_mamba_baseline'

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False

def make_model(name,seed):
    c=json.loads((ROOT/'configs/eth_ucy_mamba_baseline.json').read_text())
    seed_all(seed)
    # Isolated RNG ensures identical decoder regardless of backbone RNG consumption.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        decoder=make_trajectory_decoder(c['d_model'],c['pred_len'],c['dropout']).state_dict()
    common={k:c[k] for k in ('input_dim','d_model','num_layers','pred_len','dropout')}
    if name=='ett': model=TargetOnlyTrajectoryTransformer(**common,nhead=c['transformer']['nhead'],max_obs_len=c['obs_len'])
    elif name=='emt': model=MambaTrajectoryPredictor(**common,**c['mamba'])
    else: raise ValueError(name)
    model.decoder.load_state_dict(decoder)
    return model,state_hash(model.decoder.state_dict())

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ade=fde=0.;n=0
    for batch in loader:
        x=batch['obs_input'].to(device);last=batch['last_obs_pos'].to(device)
        pred=model(x)['future_pred']+last[:,None]
        d=torch.linalg.vector_norm(pred-batch['future_abs'].to(device),dim=-1)
        if not torch.isfinite(d).all():raise FloatingPointError('nonfinite validation')
        ade+=d.mean(1).sum().item();fde+=d[:,-1].sum().item();n+=len(x)
    return {'ADE':ade/n,'FDE':fde/n}

@torch.no_grad()
def benchmark(model,device):
    model.eval();out={}
    for b in (1,128):
        x=torch.zeros(b,8,4,device=device)
        for _ in range(20):model(x)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):model(x)
        torch.cuda.synchronize();out[f'batch_{b}_latency_ms']=(time.perf_counter()-start)*10
    out['method']='20 warmups, 100 forwards, synchronized CUDA wall time, eval/inference; batch latency'
    return out

def train(name,heldout,seed,epochs=100,smoke=False,defer_latency=False):
    torch.set_num_threads(1)
    device=torch.device('cuda')
    model,decoder_hash=make_model(name,seed);model=model.to(device)
    tr=ETHUCYDataset(heldout,'train');va=ETHUCYDataset(heldout,'val')
    assert not tr.tracks & va.tracks and heldout not in tr.scenes|va.scenes
    gen=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(tr,batch_size=128,shuffle=True,generator=gen,num_workers=0)
    val_loader=DataLoader(va,batch_size=128,shuffle=False,num_workers=0)
    path=RESULTS/('smoke' if smoke else f'heldout_{heldout}')/name/f'seed_{seed}'
    checkpoint=ROOT/'checkpoints/eth_ucy_mamba_baseline'/('smoke' if smoke else f'heldout_{heldout}')/name/f'seed_{seed}.pt'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
    criterion=nn.SmoothL1Loss(beta=1.)
    count=sum(p.numel() for p in model.parameters())
    dump(path/'parameter_count.json',{'total':count,'trainable':count})
    dump(path/'initialization_report.json',{'seed':seed,'decoder_initialization_sha256':decoder_hash,'model_initialization_sha256':state_hash(model.state_dict()),'decoder_rng_seed':seed})
    provenance=json.loads((RESULTS/'data_audit/data_provenance.json').read_text())
    manifest=ROOT/'data/processed/eth_ucy/manifests'/f'{heldout}.json'
    provenance.update({'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'heldout_test_accessed':False})
    dump(path/'data_provenance.json',provenance)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    history=[];best=float('inf');bad=0;max_grad=0.;best_epoch=0;best_metrics=None
    for epoch in range(1,epochs+1):
        model.train();total=0.;n=0
        for batch in train_loader:
            x=batch['obs_input'].to(device);y=batch['future_target'].to(device)
            optimizer.zero_grad(set_to_none=True);pred=model(x)['future_pred'];loss=criterion(pred,y)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            max_grad=max(max_grad,float(norm));optimizer.step();total+=float(loss)*len(x);n+=len(x)
        metrics=evaluate(model,val_loader,device);scheduler.step(metrics['ADE'])
        history.append({'epoch':epoch,'train_loss':total/n,'validation_ADE':metrics['ADE'],'validation_FDE':metrics['FDE'],'lr':optimizer.param_groups[0]['lr']})
        if metrics['ADE']<best:
            best=metrics['ADE'];bad=0;best_epoch=epoch;best_metrics=metrics
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation':metrics,'heldout_test_accessed':False},checkpoint)
        else:bad+=1
        dump(path/'validation_history.json',history)
        if epoch==1 or epoch%10==0:print(f'{heldout} {name} {seed} epoch {epoch}: loss={total/n:.4f} ADE={metrics["ADE"]:.4f} FDE={metrics["FDE"]:.4f}',flush=True)
        if bad>=12:break
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(checkpoint,weights_only=True)['model'])
    checked=evaluate(model,val_loader,device)
    assert all(abs(checked[k]-best_metrics[k])<1e-6 for k in checked)
    latency=None if defer_latency else benchmark(model,device)
    report={'heldout_scene':heldout,'model':name.upper(),'seed':seed,'train_sample_count':len(tr),'validation_sample_count':len(va),'train_pedestrian_count':len(tr.tracks),'validation_pedestrian_count':len(va.tracks),'best_epoch':best_epoch,'epochs_run':len(history),'validation_ADE':checked['ADE'],'validation_FDE':checked['FDE'],'parameter_count':count,'maximum_gradient_norm_before_clipping':max_grad,'training_seconds':seconds,'gpu_peak_memory_bytes':peak,'nan_inf':False,'heldout_test_accessed':False,'decoder_initialization_sha256':decoder_hash,'smoke':smoke,'latency':latency,'protocol':{'max_epochs':epochs,'batch_size':128,'optimizer':'AdamW','lr':1e-3,'weight_decay':1e-4,'gradient_clip_norm':5.,'loss':'SmoothL1Loss(beta=1)','scheduler':'ReduceLROnPlateau(min, factor=.5, patience=5)','early_stopping_patience':12,'checkpoint_selection':'minimum source validation ADE'}}
    dump(path/'metrics_validation.json',report)
    return report

def cli(name):
    p=argparse.ArgumentParser();p.add_argument('--heldout',choices=SCENES,default='eth');p.add_argument('--seed',type=int,choices=[42,123,2024],default=42);p.add_argument('--epochs',type=int,default=100);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    train(name,a.heldout,a.seed,a.epochs,a.smoke)
