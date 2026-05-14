from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_loss_curve(train_log_csv, out_path):
    df=pd.read_csv(train_log_csv); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(df['epoch'], df['train_loss'], label='train')
    if 'valid_relative_l2' in df: plt.plot(df['epoch'], df['valid_relative_l2'], label='valid relative L2')
    plt.xlabel('epoch'); plt.legend(); plt.title('Sheaf Neural Operators for MHD loss'); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_prediction_example(pred, target, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); p=pred.detach().cpu()[0,0]; t=target.detach().cpu()[0,0]
    plt.figure(figsize=(8,3));
    for i,(a,title) in enumerate([(p,'prediction'),(t,'target'),((p-t).abs(),'abs error')]):
        plt.subplot(1,3,i+1); plt.imshow(a if a.ndim==2 else a[...,0]); plt.title(title); plt.colorbar(fraction=0.046)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_error_heatmap(pred, target, out_path):
    err=(pred-target).detach().abs().mean(dim=1).cpu()[0]; Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.imshow(err if err.ndim==2 else err[...,0]); plt.title('Mean absolute error'); plt.colorbar(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_divergence_map(div, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); d=div.detach().cpu()[0]
    plt.figure(); plt.imshow(d if d.ndim==2 else d[...,0]); plt.title('Magnetic divergence map'); plt.colorbar(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_rollout_error(metrics, out_path):
    y=metrics.get('rollout_relative_l2_by_step',[]); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(range(1,len(y)+1), y, marker='o'); plt.xlabel('rollout step'); plt.ylabel('relative L2'); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_aggregate_bars(agg, out_path, metric='relative_l2'):
    sub=agg[agg.metric==metric]; Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if sub.empty: return
    labels=[f"{r.dataset}\n{r.model}" for r in sub.itertuples()]; y=sub['mean'].to_numpy(); err=np.vstack([y-sub['ci95_low'].to_numpy(), sub['ci95_high'].to_numpy()-y])
    plt.figure(figsize=(max(6,len(labels)),4)); plt.bar(labels,y,yerr=err); plt.ylabel(metric); plt.xticks(rotation=30,ha='right'); plt.tight_layout(); plt.savefig(out_path); plt.close()
