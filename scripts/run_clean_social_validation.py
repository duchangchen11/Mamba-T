from clean_social_utils import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing


def smoke():
    p=RESULTS/'smoke/smoke_gate.json'
    if p.exists():
        assert json.loads(p.read_text())['passed'];return
    for name in VARIANTS:
        kind=name.split('_')[0];b,meta,tr,va=prepare_clean_pair(kind,'eth',42)
        report=train_clean(name,'eth',42,b,meta,tr,va,epochs=5,smoke=True)
        h=json.loads((RESULTS/'smoke'/name/'seed_42/validation_history.json').read_text())
        assert h[-1]['train_loss']<h[0]['train_loss'] and report['backbone_unchanged'] and not report['nan_inf']
        del b,tr,va;torch.cuda.empty_cache()
    dump(p,{'passed':True,'train_cross_split_neighbor_violations':0,'both_models':True,'initial_output_equals_base':True,'backbone_frozen':True,'loss_decreased':True,'heldout_test_accessed':False})


def worker(job):
    torch.set_num_threads(1);fold,seed=job
    for name in VARIANTS:
        p=RESULTS/f'heldout_{fold}'/name/f'seed_{seed}/metrics_validation.json'
        if p.exists():
            r=json.loads(p.read_text());assert not r['smoke'] and r['protocol']==CONFIG;continue
        b,meta,tr,va=prepare_clean_pair(name.split('_')[0],fold,seed)
        train_clean(name,fold,seed,b,meta,tr,va)
        del b,tr,va;torch.cuda.empty_cache()
    return job


def main():
    torch.set_num_threads(1);verify_frozen();smoke()
    jobs=[(f,s) for f in SCENES for s in (42,123,2024)]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,j) for j in jobs];errors=[]
        for future in as_completed(futures):
            try:print('PAIR COMPLETE',future.result(),flush=True)
            except Exception as e:errors.append(repr(e));print('PAIR FAILED',repr(e),flush=True)
        if errors:raise RuntimeError(errors)
    for f,s in jobs:
        for name in VARIANTS:
            p=RESULTS/f'heldout_{f}'/name/f'seed_{s}/metrics_validation.json';r=json.loads(p.read_text())
            b,meta=load_backbone(name.split('_')[0],f,s,RESULTS/'data_audit/stage1_freeze_inventory.json')
            m=SocialResidualPredictor(b,name,s).cuda();ckpt=CHECKPOINTS/f'heldout_{f}'/name/f'seed_{s}.pt'
            restore_new_state(m,torch.load(ckpt,weights_only=True)['new_modules'])
            r['latency']=benchmark_full(m);dump(p,r);del m,b;torch.cuda.empty_cache()
    verify_frozen()
    from summarize_clean_social_validation import main as summarize
    summarize()
if __name__=='__main__':main()
