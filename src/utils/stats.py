from __future__ import annotations
import math
import numpy as np
import pandas as pd
try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

def summarize(values, bootstrap:int=10000, seed:int=123):
    x=np.asarray([v for v in values if np.isfinite(v)], dtype=float); n=len(x)
    if n==0: return dict(mean=np.nan,std=np.nan,se=np.nan,ci95_low=np.nan,ci95_high=np.nan,boot_ci95_low=np.nan,boot_ci95_high=np.nan,median=np.nan,iqr=np.nan,n=0)
    mean=float(x.mean()); std=float(x.std(ddof=1)) if n>1 else 0.0; se=std/math.sqrt(n) if n else np.nan
    tcrit=float(scipy_stats.t.ppf(0.975,n-1)) if scipy_stats and n>1 else 1.96
    rng=np.random.default_rng(seed); reps=min(bootstrap,10000)
    boots=rng.choice(x, size=(reps,n), replace=True).mean(axis=1) if n else np.array([np.nan])
    q75,q25=np.percentile(x,[75,25])
    return dict(mean=mean,std=std,se=se,ci95_low=mean-tcrit*se,ci95_high=mean+tcrit*se,
                boot_ci95_low=float(np.percentile(boots,2.5)), boot_ci95_high=float(np.percentile(boots,97.5)),
                median=float(np.median(x)), iqr=float(q75-q25), n=n)

def aggregate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    metrics=[c for c in raw.columns if c not in {'dataset','model','seed','status','error'} and pd.api.types.is_numeric_dtype(raw[c])]
    rows=[]
    for (ds,model), g in raw.groupby(['dataset','model']):
        for m in metrics:
            rows.append({'dataset':ds,'model':model,'metric':m, **summarize(g[m].to_numpy())})
    return pd.DataFrame(rows)

def pairwise_comparisons(raw: pd.DataFrame, metrics=None) -> pd.DataFrame:
    metrics = metrics or ['relative_l2','mse','magnetic_divergence_l2','mean_rollout_relative_l2']
    rows=[]
    for ds,gds in raw.groupby('dataset'):
        for base in ['unet','fno']:
            for metric in metrics:
                s=gds[gds.model=='sheaf_mhd'][['seed',metric]].merge(gds[gds.model==base][['seed',metric]], on='seed', suffixes=('_sheaf','_'+base)) if metric in gds else pd.DataFrame()
                if s.empty: continue
                diff=s[f'{metric}_sheaf']-s[f'{metric}_{base}']; mean_base=s[f'{metric}_{base}'].mean(); summ=summarize(diff)
                p=float(scipy_stats.ttest_rel(s[f'{metric}_sheaf'], s[f'{metric}_{base}'], nan_policy='omit').pvalue) if scipy_stats and len(s)>1 else np.nan
                rows.append({'dataset':ds,'comparison':f'sheaf_mhd_vs_{base}','metric':metric,'mean_difference':summ['mean'],'relative_percent_improvement':float(-100*summ['mean']/mean_base) if mean_base else np.nan,'p_value':p,**{f'diff_{k}':v for k,v in summ.items()}})
    return pd.DataFrame(rows)
