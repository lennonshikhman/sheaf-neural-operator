from __future__ import annotations
from pathlib import Path
import pandas as pd

def _fmt(row): return f"{row['mean']:.4g} [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}]"

def write_tables(agg: pd.DataFrame, out_dir: str|Path) -> None:
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    metrics=['relative_l2','mse','magnetic_divergence_l2','mean_rollout_relative_l2','parameter_count','inference_time_per_batch']
    rows=[]
    for (ds,model), g in agg.groupby(['dataset','model']):
        r={'dataset':ds,'model':'Sheaf Neural Operator' if model=='sheaf_mhd' else model.upper()}
        for m in metrics:
            gm=g[g.metric==m]; r[m]=_fmt(gm.iloc[0]) if not gm.empty else ''
        rows.append(r)
    df=pd.DataFrame(rows)
    df.to_markdown(out/'main_results.md', index=False)
    df.to_latex(out/'main_results.tex', index=False, escape=False)
    for name, mets in {'divergence_results.tex':['magnetic_divergence_l2','magnetic_divergence_relative'], 'rollout_results.tex':['mean_rollout_relative_l2','final_step_relative_l2']}.items():
        sub=agg[agg.metric.isin(mets)]
        sub.to_latex(out/name, index=False)
