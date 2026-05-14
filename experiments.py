"""One-command experimental suite for Sheaf Neural Operators for MHD.

Run the complete default suite with:
    python experiments.py

Developers may manually set SMOKE_TEST=True below for a local 1-epoch pipeline check; no
command-line flags are required or used.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json, traceback

FULL_EXPERIMENT = True
SMOKE_TEST = False

MISSING_DEPENDENCY = None
try:
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from src.utils.config import load_yaml, save_json
    from src.utils.logging import setup_logger
    from src.utils.seed import seed_everything
    from src.datasets.orszag_tang import OrszagTangDataset
    from src.datasets.well_mhd import WellMHD64Dataset
    from src.models import UNet2D, UNet3D, FNO2D, FNO3D, SheafMHDOperator
    from src.training.trainer import Trainer
    from src.training.evaluator import evaluate
    from src.training.rollout import rollout_evaluate
    from src.utils.stats import aggregate_metrics, pairwise_comparisons
    from src.utils.tables import write_tables
    from src.utils.plotting import plot_loss_curve, plot_rollout_error, plot_aggregate_bars
except ModuleNotFoundError as exc:
    MISSING_DEPENDENCY = exc


def save_json_fallback(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def missing_dependency_main(exc: ModuleNotFoundError) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_root = Path("outputs") / ("full_experiment_" + timestamp)
    (out_root / "paper_tables").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    msg = (f"Missing Python dependency: {exc.name}. Install requirements with `pip install -r requirements.txt` "
           "in an environment with PyTorch/NumPy/SciPy/pandas/h5py/tqdm/matplotlib before running the full suite.")
    summary = {"outputs_saved": str(out_root), "completed_runs": 0, "failed_runs": 60, "failures": [{"status": "failed", "error": msg}],
               "aggregate_metrics_csv": str(out_root / "aggregate_metrics.csv"),
               "main_results_tex": str(out_root / "paper_tables/main_results.tex"), "figures_dir": str(out_root / "figures")}
    (out_root / "raw_metrics.csv").write_text("status,error\nfailed,"" + msg.replace('"', "'") + ""\n", encoding="utf-8")
    (out_root / "aggregate_metrics.csv").write_text("", encoding="utf-8")
    (out_root / "paper_tables/main_results.tex").write_text("% " + msg + "\n", encoding="utf-8")
    (out_root / "paper_tables/main_results.md").write_text(msg + "\n", encoding="utf-8")
    save_json_fallback(summary, out_root / "experiment_summary.json")
    save_json_fallback({"dependency_error": msg, "FULL_EXPERIMENT": FULL_EXPERIMENT, "SMOKE_TEST": SMOKE_TEST}, out_root / "resolved_experiment_config.json")
    (out_root / "experiment_log.txt").write_text(msg + "\n", encoding="utf-8")
    print(msg)
    print(f"Outputs saved: {out_root}")
    print("Completed runs: 0")
    print("Failed runs: 60")
    print(f"Aggregate metrics CSV: {out_root/'aggregate_metrics.csv'}")
    print(f"Main results TeX: {out_root/'paper_tables/main_results.tex'}")
    print(f"Figures directory: {out_root/'figures'}")

def apply_smoke(cfg: dict) -> dict:
    if not SMOKE_TEST:
        return cfg
    cfg = json.loads(json.dumps(cfg))
    cfg['seeds'] = [0]
    for d in cfg['datasets'].values():
        d['epochs'] = 1; d['max_train_samples'] = 4; d['max_valid_samples'] = 2; d['max_test_samples'] = 2
        d['hidden_channels'] = min(8, d.get('hidden_channels', 8)); d['num_layers'] = 1; d['modes'] = min(4, d.get('modes', 4))
    return cfg


def build_dataset(name: str, dc: dict, split: str):
    cap = dc.get({'train':'max_train_samples','valid':'max_valid_samples','test':'max_test_samples'}[split])
    if name == 'orszag_tang':
        return OrszagTangDataset(dc['data_root'], split, dc.get('n_input_frames',1), dc.get('target_frame',1), cap, normalize=True)
    if name == 'wells_mhd64':
        return WellMHD64Dataset(dc['data_root'], split, dc.get('n_input_frames',1), dc.get('n_output_frames',1), cap, True, None, dc.get('magnetic_field_indices'))
    raise KeyError(name)


def build_model(model_name: str, dim: int, in_ch: int, out_ch: int, dc: dict):
    kw = dict(in_channels=in_ch, out_channels=out_ch, hidden_channels=dc['hidden_channels'], num_layers=dc['num_layers'], modes=dc['modes'])
    if model_name == 'unet':
        return (UNet2D if dim == 2 else UNet3D)(**kw)
    if model_name == 'fno':
        return (FNO2D if dim == 2 else FNO3D)(**kw)
    if model_name == 'sheaf_mhd':
        return SheafMHDOperator(dim=dim, periodic=dc.get('periodic', True), dt=dc.get('dt',1.0), spacing=dc.get('spacing'),
                                magnetic_field_indices=dc.get('magnetic_field_indices'), fluid_field_indices=dc.get('fluid_field_indices'),
                                constrained_magnetic_update=(dim == 2), **kw)
    raise KeyError(model_name)


def run_prediction_figures(model, loader, device, run_dir: Path):
    # Lightweight per-run figures are best-effort; failures should not affect metrics.
    try:
        from src.utils.plotting import plot_prediction_example, plot_error_heatmap, plot_divergence_map
        from src.physics.divergence import periodic_divergence_2d
        batch = next(iter(loader)); x=batch['x'].to(device); y=batch['y'].to(device); pred=model(x)
        fig_dir=run_dir/'figures'; plot_prediction_example(pred,y,fig_dir/'prediction_example.png'); plot_error_heatmap(pred,y,fig_dir/'error_heatmap.png')
    except Exception:
        pass


def main() -> None:
    cfg = apply_smoke(load_yaml('configs/default_experiment.yaml'))
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_root = Path('outputs') / (('smoke_experiment_' if SMOKE_TEST else 'full_experiment_') + timestamp)
    for sub in ['paper_tables','figures/loss_curves','figures/prediction_examples','figures/divergence_maps','figures/rollout_curves','figures/aggregate_barplots','runs']:
        (out_root/sub).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root/'experiment_log.txt')
    save_json(cfg, out_root/'resolved_experiment_config.json')
    logger.info('Starting Sheaf Neural Operators for MHD suite. FULL_EXPERIMENT=%s SMOKE_TEST=%s', FULL_EXPERIMENT, SMOKE_TEST)
    raw_rows=[]; failures=[]; completed=0
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); logger.info('Using device: %s', device)

    for dataset_name, dc in cfg['datasets'].items():
        root=Path(dc['data_root'])
        if not root.exists():
            msg=f"Dataset root missing: {root}. Data are not downloaded by this codebase."
            logger.error('%s failed dataset check: %s', dataset_name, msg)
            for model_name in cfg['models']:
                for seed in cfg['seeds']:
                    row={'dataset':dataset_name,'model':model_name,'seed':seed,'status':'failed','error':msg}
                    raw_rows.append(row); failures.append(row)
            continue
        try:
            train_ds=build_dataset(dataset_name, dc, 'train'); valid_ds=build_dataset(dataset_name, dc, 'valid'); test_ds=build_dataset(dataset_name, dc, 'test')
            sample=train_ds[0]; in_ch=sample['x'].shape[0]; out_ch=sample['y'].shape[0]
            logger.info('%s shapes: x=%s y=%s n_train=%d n_valid=%d n_test=%d', dataset_name, tuple(sample['x'].shape), tuple(sample['y'].shape), len(train_ds), len(valid_ds), len(test_ds))
        except Exception as exc:
            msg=f"Dataset inspection failed: {exc}"
            logger.error('%s', msg)
            for model_name in cfg['models']:
                for seed in cfg['seeds']:
                    row={'dataset':dataset_name,'model':model_name,'seed':seed,'status':'failed','error':msg}
                    raw_rows.append(row); failures.append(row)
            continue

        for model_name in cfg['models']:
            for seed in cfg['seeds']:
                run_dir=out_root/'runs'/dataset_name/model_name/f'seed_{seed}'; run_dir.mkdir(parents=True, exist_ok=True); (run_dir/'figures').mkdir(exist_ok=True)
                run_cfg={**dc, **cfg.get('training', {}), 'dataset':dataset_name, 'model':model_name, 'seed':seed,
                         'lambda_div': dc.get('lambda_div_sheaf',0.0) if model_name=='sheaf_mhd' else dc.get('lambda_div_baseline',0.0)}
                save_json(run_cfg, run_dir/'config_resolved.json')
                try:
                    seed_everything(seed)
                    gen=torch.Generator().manual_seed(seed)
                    loaders={}
                    for split, ds in [('train',train_ds),('valid',valid_ds),('test',test_ds)]:
                        loaders[split]=DataLoader(ds, batch_size=dc['batch_size'], shuffle=(split=='train'), num_workers=cfg['training'].get('num_workers',0), generator=gen)
                    model=build_model(model_name, dc['dim'], in_ch, out_ch, dc)
                    trainer=Trainer(model, loaders['train'], loaders['valid'], run_dir, device, run_cfg); trainer.fit()
                    valid_metrics=evaluate(model, loaders['valid'], device, dc.get('magnetic_field_indices'), dc.get('spacing'))
                    test_metrics=evaluate(model, loaders['test'], device, dc.get('magnetic_field_indices'), dc.get('spacing'))
                    rollout=rollout_evaluate(model, loaders['test'], device, cfg['training'].get('rollout_steps',5), dc.get('magnetic_field_indices'), dc.get('spacing'))
                    save_json(valid_metrics, run_dir/'metrics_valid.json'); save_json(test_metrics, run_dir/'metrics_test.json'); save_json(rollout, run_dir/'rollout_metrics.json')
                    plot_loss_curve(run_dir/'train_log.csv', run_dir/'figures/loss_curve.png'); plot_rollout_error(rollout, run_dir/'figures/rollout_error.png'); run_prediction_figures(model, loaders['test'], device, run_dir)
                    row={'dataset':dataset_name,'model':model_name,'seed':seed,'status':'completed', **test_metrics,
                         'final_step_relative_l2': rollout.get('final_step_relative_l2'), 'mean_rollout_relative_l2': rollout.get('mean_rollout_relative_l2')}
                    raw_rows.append(row); completed += 1; logger.info('Completed %s/%s seed %s', dataset_name, model_name, seed)
                except Exception as exc:
                    err={'dataset':dataset_name,'model':model_name,'seed':seed,'status':'failed','error':str(exc), 'traceback':traceback.format_exc()}
                    save_json(err, run_dir/'failure.json'); raw_rows.append(err); failures.append(err); logger.error('Run failed: %s', err)

    raw=pd.DataFrame(raw_rows); raw.to_csv(out_root/'raw_metrics.csv', index=False)
    numeric_raw=raw[raw.get('status','')=='completed'].copy() if not raw.empty and 'status' in raw else pd.DataFrame()
    if not numeric_raw.empty:
        agg=aggregate_metrics(numeric_raw); agg.to_csv(out_root/'aggregate_metrics.csv', index=False); save_json({'records':agg.to_dict(orient='records')}, out_root/'aggregate_metrics.json')
        write_tables(agg, out_root/'paper_tables')
        pair=pairwise_comparisons(numeric_raw); pair.to_latex(out_root/'paper_tables'/'pairwise_comparisons.tex', index=False)
        plot_aggregate_bars(agg, out_root/'figures/aggregate_barplots/relative_l2.png', 'relative_l2')
    else:
        agg=pd.DataFrame(); agg.to_csv(out_root/'aggregate_metrics.csv', index=False); save_json({'records':[]}, out_root/'aggregate_metrics.json')
        (out_root/'paper_tables'/'main_results.tex').write_text('% No completed runs; see experiment_summary.json\n', encoding='utf-8')
        (out_root/'paper_tables'/'main_results.md').write_text('No completed runs; see experiment_summary.json.\n', encoding='utf-8')
        for name in ['divergence_results.tex','rollout_results.tex','pairwise_comparisons.tex']:
            (out_root/'paper_tables'/name).write_text('% No completed runs\n', encoding='utf-8')
    summary={'outputs_saved':str(out_root),'completed_runs':completed,'failed_runs':len(failures),'failures':failures,
             'aggregate_metrics_csv':str(out_root/'aggregate_metrics.csv'),'main_results_tex':str(out_root/'paper_tables/main_results.tex'), 'figures_dir':str(out_root/'figures')}
    save_json(summary, out_root/'experiment_summary.json')
    logger.info('Final summary: %s', summary)
    print(f"Outputs saved: {out_root}")
    print(f"Completed runs: {completed}")
    print(f"Failed runs: {len(failures)}")
    print(f"Aggregate metrics CSV: {out_root/'aggregate_metrics.csv'}")
    print(f"Main results TeX: {out_root/'paper_tables/main_results.tex'}")
    print(f"Figures directory: {out_root/'figures'}")

if __name__ == '__main__':
    if MISSING_DEPENDENCY is not None:
        missing_dependency_main(MISSING_DEPENDENCY)
    else:
        main()
