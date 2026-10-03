"""Smoke gate, independent parallel runs, then exclusive inference benchmarking."""
from eth_ucy_utils import *
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def worker(job):
    name,fold,seed=job
    return train(name,fold,seed,defer_latency=True)

def main():
    smoke_path=RESULTS/'smoke/smoke_gate.json'
    if not smoke_path.exists():
        smoke=[train(n,'eth',42,epochs=5,smoke=True) for n in ('ett','emt')]
        assert smoke[0]['decoder_initialization_sha256']==smoke[1]['decoder_initialization_sha256']
        for n,r in zip(('ett','emt'),smoke):
            h=json.loads((RESULTS/'smoke'/n/'seed_42/validation_history.json').read_text())
            assert h[-1]['train_loss']<h[0]['train_loss'] and not r['nan_inf'] and not r['heldout_test_accessed']
        dump(smoke_path,{'passed':True,'decoder_matched':True,'loss_decreased':True,'mamba_backward':True,'heldout_test_accessed':False})
    assert json.loads(smoke_path.read_text())['passed']
    jobs=[]
    for fold in SCENES:
        for seed in (42,123,2024):
            for n in ('ett','emt'):
                p=RESULTS/f'heldout_{fold}'/n/f'seed_{seed}/metrics_validation.json'
                if p.exists():
                    r=json.loads(p.read_text());assert r['protocol']['max_epochs']==100 and not r['smoke']
                else:jobs.append((n,fold,seed))
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for r in pool.map(worker,jobs):print(f'COMPLETE {r["heldout_scene"]} {r["model"]} {r["seed"]}',flush=True)
    # All workers have exited: measure each saved model without competing training.
    torch.set_num_threads(1)
    for fold in SCENES:
        for seed in (42,123,2024):
            for n in ('ett','emt'):
                p=RESULTS/f'heldout_{fold}'/n/f'seed_{seed}/metrics_validation.json'
                r=json.loads(p.read_text());model,_=make_model(n,seed)
                ckpt=ROOT/'checkpoints/eth_ucy_mamba_baseline'/f'heldout_{fold}'/n/f'seed_{seed}.pt'
                model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['model']);model=model.cuda()
                r['latency']=benchmark(model,torch.device('cuda'));dump(p,r);del model;torch.cuda.empty_cache()
    from summarize_eth_ucy_baselines import main as summarize
    summarize()
if __name__=='__main__':main()
