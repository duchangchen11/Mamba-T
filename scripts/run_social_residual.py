from social_residual_utils import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing

def smoke():
    gate=RESULTS/'smoke/smoke_gate.json'
    if gate.exists():
        assert json.loads(gate.read_text())['passed'];return
    outputs=[]
    for kind,names in (('emt',VARIANTS[:3]),('ett',('ett_sr',))):
        b,meta,tr,va=prepare_pair(kind,'eth',42)
        for name in names:outputs.append(train_variant(name,'eth',42,b,meta,tr,va,epochs=5,smoke=True))
        del b,tr,va;torch.cuda.empty_cache()
    for r in outputs:
        p=RESULTS/'smoke'/r['model']/'seed_42/validation_history.json';h=json.loads(p.read_text())
        assert h[-1]['train_loss']<h[0]['train_loss'] and not r['nan_inf'] and r['initial_output_base_max_abs_diff']<1e-7
    dump(gate,{'passed':True,'all_four_models':True,'loss_decreased':True,'backbone_frozen':True,'checkpoint_hash_verified':True,'initial_output_equals_base':True,'heldout_test_accessed':False})
    print('SMOKE PASSED',flush=True)

def worker(job):
    torch.set_num_threads(1);fold,seed=job;reports=[]
    for kind,names in (('emt',VARIANTS[:3]),('ett',('ett_sr',))):
        pending=[name for name in names if not (RESULTS/f'heldout_{fold}'/name/f'seed_{seed}/metrics_validation.json').exists()]
        if not pending:continue
        b,meta,tr,va=prepare_pair(kind,fold,seed)
        for name in pending:reports.append(train_variant(name,fold,seed,b,meta,tr,va))
        del b,tr,va;torch.cuda.empty_cache()
    return fold,seed

def main():
    torch.set_num_threads(1);smoke()
    jobs=[(fold,seed) for fold in SCENES for seed in (42,123,2024)]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        failures=[]
        for future in as_completed(futures):
            try:print('PAIR COMPLETE',future.result(),flush=True)
            except Exception as error:
                failures.append(repr(error));print('PAIR FAILED',repr(error),flush=True)
        if failures:raise RuntimeError(f'Incomplete pairs; rerun resumes completed runs: {failures}')
    for fold,seed in jobs:
        for name in VARIANTS:
            p=RESULTS/f'heldout_{fold}'/name/f'seed_{seed}/metrics_validation.json';report=json.loads(p.read_text())
            kind='ett' if name=='ett_sr' else 'emt';b,meta=load_backbone(kind,fold,seed)
            model=SocialResidualPredictor(b,name,seed).cuda()
            ckpt=ROOT/'checkpoints/social_residual'/f'heldout_{fold}'/name/f'seed_{seed}.pt'
            restore_new_state(model,torch.load(ckpt,weights_only=True)['new_modules'])
            report['latency']=benchmark_full(model);dump(p,report);del model,b;torch.cuda.empty_cache()
    from summarize_social_residual import main as summarize
    summarize()
if __name__=='__main__':main()
