"""Clean datasets with identical frozen-model optimizer, loss and stopping rules."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.social_residual_utils import load_backbone,cache_contexts,baseline_report,train_variant,dump,sha_file,benchmark_full,restore_new_state,ROOT,SCENES,torch,SocialResidualPredictor
from src.data.eth_ucy_social_clean_dataset import ETHUCYSocialCleanDataset
RESULTS=ROOT/'results/clean_social_validation'
CHECKPOINTS=ROOT/'checkpoints/clean_social_validation'
CONFIG=json.loads((ROOT/'configs/clean_social_validation.json').read_text())
VARIANTS=('emt_sr','ett_sr')
BASE_RESULTS=ROOT/'results/eth_ucy_mamba_baseline'
OLD_RESULTS=ROOT/'results/social_residual'


def verify_frozen():
    inventory=json.loads((RESULTS/'data_audit/frozen_inventory.json').read_text())
    for path,sha in inventory.items():
        if sha_file(ROOT/path)!=sha:raise ValueError(f'Frozen artifact changed: {path}')
    return len(inventory)


def prepare_clean_pair(kind,fold,seed):
    b,meta=load_backbone(kind,fold,seed,RESULTS/'data_audit/stage1_freeze_inventory.json')
    train=ETHUCYSocialCleanDataset(fold,'train');val=ETHUCYSocialCleanDataset(fold,'val')
    assert not train.tracks&val.tracks and fold not in train.scenes|val.scenes
    start=time.perf_counter();tr=cache_contexts(b,train);va=cache_contexts(b,val)
    base=baseline_report(va);original=json.loads((BASE_RESULTS/f'heldout_{fold}'/kind/f'seed_{seed}/metrics_validation.json').read_text())
    differences={m:base[m]-original[f'validation_{m}'] for m in ('ADE','FDE')}
    assert all(abs(v)<1e-6 for v in differences.values())
    meta.update({'context_cache_seconds':time.perf_counter()-start,'train_pedestrian_count':len(train.tracks),'validation_pedestrian_count':len(val.tracks),'baseline_validation_difference':differences,'neighbor_policy':'train-target-set-only for train; all observation-visible for val','train_cross_split_neighbor_violations':0})
    dump(RESULTS/'data_audit/base_validation'/f'{fold}_{kind}_{seed}.json',{'reproduced_validation':base,'stage1_metrics':{m:original[f'validation_{m}'] for m in ('ADE','FDE')},'heldout_test_accessed':False,**meta})
    return b,meta,tr,va


def train_clean(name,fold,seed,b,meta,tr,va,epochs=50,smoke=False):
    if name not in VARIANTS:raise ValueError('Only ungated SR models are authorized')
    return train_variant(name,fold,seed,b,meta,tr,va,epochs,smoke,results_root=RESULTS,checkpoints_root=CHECKPOINTS,protocol=CONFIG)
