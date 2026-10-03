"""Observation-only features and prediction-derived responses, with explicit masks."""
import numpy as np

EPSILON = 1e-8
OBS_FEATURES = ('neighbor_count', 'nearest_neighbor_distance', 'mean_neighbor_distance', 'median_neighbor_distance', 'max_neighbor_distance', 'mean_relative_speed', 'max_relative_speed', 'mean_closing_speed', 'max_closing_speed')
ATTENTION_FEATURES = ('attention_entropy', 'attention_max', 'attention_top2_sum')
RESPONSE_FEATURES = ('residual_norm', 'residual_final_norm', 'correction_ratio')
SHIFT_FEATURES = OBS_FEATURES + ATTENTION_FEATURES + RESPONSE_FEATURES
SUMMARY_FEATURES = SHIFT_FEATURES + ('mean_signed_closing_speed', 'mean_positive_closing_speed', 'max_signed_closing_speed', 'base_ADE', 'SR_ADE', 'ADE_gain', 'base_FDE', 'SR_FDE', 'FDE_gain')


def observation_features(target_history, neighbor_history, relation, mask, epsilon=EPSILON):
    h, n, r, valid = np.asarray(target_history, dtype=np.float64), np.asarray(neighbor_history, dtype=np.float64), np.asarray(relation, dtype=np.float64), np.asarray(mask, dtype=bool)
    count = valid.sum(1)
    out = {'neighbor_count': count}
    distance = np.where(valid, r[..., 2], np.nan)
    relative_v = (n[:, :, -1, :2]-n[:, :, -2, :2])-(h[:, -1, :2]-h[:, -2, :2])[:, None]
    speed = np.linalg.norm(relative_v, axis=-1)
    radial = -np.sum(r[..., :2]*relative_v, axis=-1)/(np.linalg.norm(r[..., :2], axis=-1)+epsilon)
    has = count > 0
    for name, values, method in (('nearest_neighbor_distance',distance,'min'),('mean_neighbor_distance',distance,'mean'),('median_neighbor_distance',distance,'median'),('max_neighbor_distance',distance,'max'),('mean_relative_speed',speed,'mean'),('max_relative_speed',speed,'max'),('mean_signed_closing_speed',radial,'mean'),('max_signed_closing_speed',radial,'max')):
        q = np.full(len(h), np.nan)
        if has.any():
            selected = np.where(valid[has], values[has], np.nan)
            q[has] = getattr(np, f'nan{method}')(selected, axis=1)
        out[name] = q
    positive = np.where(valid & (radial > 0), radial, 0.)
    positive_count = (valid & (radial > 0)).sum(1)
    out['mean_positive_closing_speed'] = np.divide(positive.sum(1),positive_count,out=np.zeros(len(h)),where=positive_count>0)
    out['mean_closing_speed'] = out['mean_positive_closing_speed'].copy()
    out['max_closing_speed'] = positive.max(1)
    for i in range(8):
        out[f'rank{i+1}_neighbor_distance'] = np.sort(distance,axis=1)[:, i]
    return out


def attention_features(weights, mask, epsilon=EPSILON):
    valid = np.asarray(mask,dtype=bool)
    w = np.asarray(weights,dtype=np.float64)
    if w.ndim == 3:
        w = w.mean(1)
    w = np.where(valid,w,0.)
    if np.any(w < -1e-7) or not np.isfinite(w).all():
        raise ValueError('Invalid attention weights')
    w = np.maximum(w,0.)
    mass = w.sum(1)
    count = valid.sum(1)
    if np.any((count>0)&(mass<=0)):
        raise ValueError('Valid neighbors have no attention mass')
    alpha = np.divide(w,mass[:,None],out=np.zeros_like(w),where=mass[:,None]>0)
    entropy = np.zeros(len(w))
    selected = count > 1
    entropy[selected] = -(alpha[selected]*np.log(alpha[selected]+epsilon)).sum(1)/np.log(count[selected])
    entropy = np.clip(entropy,0.,1.)
    sorted_alpha = np.sort(alpha,axis=1)
    return {'attention_entropy':entropy,'attention_max':alpha.max(1),'attention_top2_sum':sorted_alpha[:,-2:].sum(1)},alpha


def response_features(residual, base_prediction, base_ADE, SR_ADE, base_FDE, SR_FDE, epsilon=EPSILON):
    r = np.linalg.norm(np.asarray(residual,dtype=np.float64),axis=-1)
    b = np.linalg.norm(np.asarray(base_prediction,dtype=np.float64),axis=-1).mean(1)
    return {'residual_norm':r.mean(1),'residual_final_norm':r[:,-1],'correction_ratio':r.mean(1)/(b+epsilon),'base_trajectory_norm':b,'base_ADE':np.asarray(base_ADE),'SR_ADE':np.asarray(SR_ADE),'ADE_gain':np.asarray(base_ADE)-np.asarray(SR_ADE),'base_FDE':np.asarray(base_FDE),'SR_FDE':np.asarray(SR_FDE),'FDE_gain':np.asarray(base_FDE)-np.asarray(SR_FDE)}


def group_masks(samples, quartiles):
    n,d,c = samples['neighbor_count'],samples['nearest_neighbor_distance'],samples['max_closing_speed']
    q1,q2,q3 = quartiles
    return {'neighbor_count':{'0':n==0,'1-2':(n>=1)&(n<=2),'3-4':(n>=3)&(n<=4),'5-6':(n>=5)&(n<=6),'7-8':(n>=7)&(n<=8)},'nearest_distance':{'no neighbor':n==0,'<2m':(n>0)&(d<2),'2-4m':(n>0)&(d>=2)&(d<4),'4-8m':(n>0)&(d>=4)&(d<=8),'>8m':(n>0)&(d>8)},'closing_speed':{'Q1':c<=q1,'Q2':(c>q1)&(c<=q2),'Q3':(c>q2)&(c<=q3),'Q4':c>q3},'saturation':{'N=8':n==8,'N<8':n<8}}
