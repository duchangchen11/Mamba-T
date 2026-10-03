"""Descriptive scene-balanced statistics; no significance tests or trajectory training."""
import itertools
import numpy as np
from src.analysis.social_shift_features import OBS_FEATURES, SHIFT_FEATURES, SUMMARY_FEATURES, group_masks
from scripts.social_shift_protocol import SCENES


def collapse_windows(raw):
    uids,first,inverse=np.unique(raw['sample_uid'],return_index=True,return_inverse=True)
    count=np.bincount(inverse)
    out={k:raw[k][first] for k in ('scene','ped_id','sample_uid','observation_frame_ids')}
    out['prediction_count']=count
    for name,values in raw.items():
        if values.ndim!=1 or values.dtype.kind not in 'fiu' or name=='seed':
            continue
        finite=np.isfinite(values)
        totals=np.bincount(inverse[finite],weights=values[finite],minlength=len(uids))
        ns=np.bincount(inverse[finite],minlength=len(uids))
        out[name]=np.divide(totals,ns,out=np.full(len(uids),np.nan),where=ns>0)
    for name in OBS_FEATURES:
        np.testing.assert_allclose(raw[name],out[name][inverse],rtol=1e-6,atol=1e-6,equal_nan=True)
    return out


def subset(samples, selected):
    return {k:v[selected] for k,v in samples.items()}


def scene_weights(scene):
    labels,count=np.unique(scene,return_counts=True)
    return np.array([1./count[np.searchsorted(labels,s)] for s in scene])


def weighted_quantile(values,weights,quantiles):
    order=np.argsort(values,kind='stable');x=values[order];w=weights[order]
    midpoint=(np.cumsum(w)-.5*w)/w.sum()
    return np.interp(quantiles,midpoint,x,left=x[0],right=x[-1])


def summary(values,weights=None):
    finite=np.isfinite(values);v=np.asarray(values,dtype=np.float64)[finite]
    fields=('mean','std','median','p10','p25','p75','p90','p95')
    if not len(v):
        return {'sample_count':0,**{k:None for k in fields}}
    w=np.ones(len(v)) if weights is None else np.asarray(weights)[finite]
    mean=float(np.average(v,weights=w));std=float(np.sqrt(np.average((v-mean)**2,weights=w)))
    quantiles=np.quantile(v,[.5,.1,.25,.75,.9,.95]) if weights is None else weighted_quantile(v,w,[.5,.1,.25,.75,.9,.95])
    return {'sample_count':len(v),'mean':mean,'std':std,**dict(zip(fields[2:],map(float,quantiles)))}


def summarize_scenes(samples):
    features=SUMMARY_FEATURES+tuple(f'rank{i}_neighbor_distance' for i in range(1,9))
    return {s:{f:summary(samples[f][samples['scene']==s]) for f in features} for s in SCENES}


def overall_summary(samples,equal=False):
    result={}
    for f in SUMMARY_FEATURES:
        selected=np.isfinite(samples[f]);v=samples[f][selected]
        result[f]=summary(v,scene_weights(samples['scene'][selected]) if equal and len(v) else None)
    return result


def midranks(values,weights):
    unique,inverse=np.unique(values,return_inverse=True)
    mass=np.bincount(inverse,weights=weights,minlength=len(unique))
    ranks=(np.cumsum(mass)-.5*mass)/mass.sum()
    return ranks[inverse]


def pearson(x,y,w):
    x=x-np.average(x,weights=w);y=y-np.average(y,weights=w)
    variance=np.average(x*x,weights=w)*np.average(y*y,weights=w)
    return float(np.clip(np.average(x*y,weights=w)/np.sqrt(variance),-1.,1.)) if variance>1e-30 else None


def correlate(samples,x_name,y_name,equal=False,mask=None):
    selected=np.isfinite(samples[x_name])&np.isfinite(samples[y_name])
    if mask is not None:
        selected &= mask
    x,y=samples[x_name][selected],samples[y_name][selected]
    scenes=samples['scene'][selected]
    if len(x)<2:
        return {'sample_count':len(x),'present_scenes':len(np.unique(scenes)),'Pearson':None,'Spearman':None}
    w=scene_weights(scenes) if equal else np.ones(len(x))
    return {'sample_count':len(x),'present_scenes':len(np.unique(scenes)),'Pearson':pearson(x,y,w),'Spearman':pearson(midranks(x,w),midranks(y,w),w)}


def correlation_audits(samples):
    features=('neighbor_count','mean_neighbor_distance','nearest_neighbor_distance','mean_relative_speed','mean_closing_speed','max_closing_speed','attention_entropy','attention_max','residual_norm','correction_ratio')
    attention_pairs=(('neighbor_count','attention_entropy'),('neighbor_count','attention_max'),('max_closing_speed','attention_max'),('nearest_neighbor_distance','attention_max'),('mean_neighbor_distance','attention_max'))
    residual_pairs=tuple((x,'residual_norm') for x in ('neighbor_count','mean_neighbor_distance','nearest_neighbor_distance','mean_relative_speed','mean_closing_speed','max_closing_speed','attention_entropy','attention_max'))
    out={}
    for name,equal in (('pooled',False),('scene_equal',True)):
        out[name]={'gain':{x:correlate(samples,x,'ADE_gain',equal) for x in features},'attention':{f'{x} vs {y}':correlate(samples,x,y,equal) for x,y in attention_pairs},'residual':{x:correlate(samples,x,'residual_norm',equal) for x,_ in residual_pairs}}
        out[name]['attention_n_ge_2']={f'{x} vs {y}':correlate(samples,x,y,equal,samples['neighbor_count']>=2) for x,y in attention_pairs}
        out[name]['within_neighbor_count']={str(n):{f'{x} vs {y}':correlate(samples,x,y,equal,samples['neighbor_count']==n) for x,y in attention_pairs[2:]} for n in range(2,9)}
    return out


def pairwise_shift(samples):
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        wasserstein_distance=None
    pairs=[]
    # Performance gains are supplementary; primary factor can only use SHIFT_FEATURES.
    for a,b in itertools.combinations(SCENES,2):
        q={'scene_A':a,'scene_B':b,'direction':'A minus B','features':{}}
        for f in SHIFT_FEATURES+('ADE_gain','FDE_gain'):
            av=samples[f][(samples['scene']==a)&np.isfinite(samples[f])]
            bv=samples[f][(samples['scene']==b)&np.isfinite(samples[f])]
            if not len(av) or not len(bv):
                q['features'][f]={'available':False}
                continue
            difference=float(av.mean()-bv.mean());denom=float(np.sqrt((av.var()+bv.var())/2))
            smd=difference/denom if denom>0 else (0. if difference==0 else None)
            q['features'][f]={'available':True,'sample_count_A':len(av),'sample_count_B':len(bv),'mean_difference':difference,'median_difference':float(np.median(av)-np.median(bv)),'SMD':smd,'abs_SMD':abs(smd) if smd is not None else None,'zero_variance_denominator':denom==0,'Wasserstein':float(wasserstein_distance(av,bv)) if wasserstein_distance is not None else None}
        pairs.append(q)
    ranking=[]
    for f in SHIFT_FEATURES:
        smds=[p['features'][f]['abs_SMD'] for p in pairs if p['features'][f].get('abs_SMD') is not None]
        wd=[p['features'][f]['Wasserstein'] for p in pairs if p['features'][f].get('Wasserstein') is not None]
        ranking.append({'feature':f,'mean_abs_SMD':float(np.mean(smds)) if smds else None,'max_abs_SMD':float(np.max(smds)) if smds else None,'valid_pairs':len(smds),'mean_Wasserstein':float(np.mean(wd)) if wd else None})
    ranking.sort(key=lambda r:-(r['mean_abs_SMD'] if r['mean_abs_SMD'] is not None else -1))
    return {'pairs':pairs,'ranking':ranking,'Wasserstein_available':wasserstein_distance is not None}


def group_summary(samples,selected,equal=False):
    fields=('base_ADE','SR_ADE','ADE_gain','base_FDE','SR_FDE','FDE_gain','residual_norm','residual_final_norm','correction_ratio','attention_entropy','attention_max')
    s=subset(samples,selected)
    q={'sample_count_unique':int(selected.sum()),'sample_count_predictions':int(samples['prediction_count'][selected].sum()),'present_scenes':len(np.unique(s['scene']))}
    for f in fields:
        v=s[f]
        q[f]=float(np.average(v,weights=scene_weights(s['scene']) if equal else None)) if len(v) else None
    q['ADE_improvement_rate']=float(np.average(s['ADE_gain']>0,weights=scene_weights(s['scene']) if equal else None)) if len(s['scene']) else None
    return q


def grouped_performance(samples,quartiles):
    masks=group_masks(samples,quartiles)
    return {name:{category:{label:group_summary(samples,mask,equal) for label,mask in values.items()} for category,values in masks.items()} for name,equal in (('pooled',False),('scene_equal',True))}


def saturation_by_scene(samples):
    return {scene:{'fraction_N8':float((samples['neighbor_count'][samples['scene']==scene]==8).mean()),'N=8':group_summary(samples,(samples['scene']==scene)&(samples['neighbor_count']==8)),'N<8':group_summary(samples,(samples['scene']==scene)&(samples['neighbor_count']<8))} for scene in SCENES}


def paired_response(emt,ett):
    for key in ('sample_uid','scene','ped_id','observation_frame_ids','prediction_count'):
        np.testing.assert_array_equal(emt[key],ett[key])
    delta={key:emt[key] for key in ('scene','ped_id','sample_uid')}
    for f in ('residual_norm','attention_entropy','ADE_gain','FDE_gain','correction_ratio'):
        delta[f]=emt[f]-ett[f]
    return {'direction':'EMT minus ETT on the same unique observation window, matched available fold/seed predictions','per_scene':{s:{f:summary(delta[f][delta['scene']==s]) for f in ('residual_norm','attention_entropy','ADE_gain','FDE_gain','correction_ratio')} for s in SCENES},'pooled':{f:summary(delta[f]) for f in ('residual_norm','attention_entropy','ADE_gain','FDE_gain','correction_ratio')},'scene_equal':{f:summary(delta[f],scene_weights(delta['scene'])) for f in ('residual_norm','attention_entropy','ADE_gain','FDE_gain','correction_ratio')}}


def scene_classifier(samples,config):
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import accuracy_score,f1_score
    except ImportError:
        return {'executed':False,'reason':'scikit-learn unavailable; no packages installed'},None
    allowed={'neighbor_count','mean_neighbor_distance','nearest_neighbor_distance','mean_relative_speed','mean_closing_speed'}
    assert set(config['features'])==allowed
    x=np.column_stack([samples[f] for f in config['features']])
    y=np.array([SCENES.index(str(s)) for s in samples['scene']])
    groups=np.char.add(np.char.add(samples['scene'],'|'),samples['ped_id'])
    split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=config['seed'])
    pred={name:np.empty(len(y),dtype=np.int64) for name in ('logistic','majority','random_frequency')}
    cvfold=np.full(len(y),-1,dtype=np.int64);audits=[]
    for fold,(train,test) in enumerate(split.split(x,y,groups)):
        assert not set(groups[train])&set(groups[test])
        assert len(np.unique(y[train]))==5
        classifier=make_pipeline(SimpleImputer(strategy='constant',fill_value=0),StandardScaler(),LogisticRegression(C=config['C'],class_weight=config['class_weight'],max_iter=config['max_iter'],random_state=config['seed']))
        classifier.fit(x[train],y[train]);pred['logistic'][test]=classifier.predict(x[test])
        frequency=np.bincount(y[train],minlength=5)
        pred['majority'][test]=frequency.argmax()
        pred['random_frequency'][test]=np.random.default_rng(config['seed']+fold).choice(5,len(test),p=frequency/frequency.sum())
        cvfold[test]=fold
        audits.append({'fold':fold,'train_samples':len(train),'validation_samples':len(test),'train_groups':len(np.unique(groups[train])),'validation_groups':len(np.unique(groups[test])),'pedestrian_overlap':0,'train_scene_count':len(np.unique(y[train])),'validation_scene_count':len(np.unique(y[test])),'logistic_iterations':classifier[-1].n_iter_.tolist()})
    assert np.all(cvfold>=0)
    metrics={}
    for name,values in pred.items():
        metrics[name]={}
        for label,weights in (('pooled',None),('scene_equal',scene_weights(samples['scene']))):
            metrics[name][label]={'Accuracy':float(accuracy_score(y,values,sample_weight=weights)),'Macro_F1':float(f1_score(y,values,labels=np.arange(5),average='macro',sample_weight=weights,zero_division=0))}
    return {'executed':True,'feature_names':config['features'],'unique_window_samples':len(y),'groups':'scene,ped_id; repeated windows/folds/seeds deduplicated; all pedestrian windows remain together','cv':'5-fold StratifiedGroupKFold; standardization fitted inside each training fold; no tuning','metrics':metrics,'folds':audits,'no_significance_tests':True}, {'sample_uid':samples['sample_uid'],'scene':samples['scene'],'ped_id':samples['ped_id'],'true_class':y,'cv_fold':cvfold,**pred}
