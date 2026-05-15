from __future__ import annotations

import json
from pathlib import Path

import experiments


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_time_dependent_legacy_complete_results_are_skippable(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "wells_mhd64" / "fno3d" / "seed_0"
    write_json(run_dir / "config_resolved.json", {"batch_size_effective": 4})
    write_json(run_dir / "metrics_test.json", {"relative_l2": 0.1})
    write_json(run_dir / "rollout_metrics.json", {"final_step_relative_l2": 0.2, "mean_rollout_relative_l2": 0.15})
    (run_dir / "train_log.csv").write_text("epoch,total_epoch_time\n0,1.5\n1,2.5\n", encoding="utf-8")

    completed = experiments.completed_run_from_disk(run_dir, "wells_mhd64", "fno3d", 0, "time_dependent")

    assert completed is not None
    row, effective_batch_size, epoch_times = completed
    assert row["status"] == "completed"
    assert row["relative_l2"] == 0.1
    assert row["final_step_relative_l2"] == 0.2
    assert row["mean_rollout_relative_l2"] == 0.15
    assert effective_batch_size == 4
    assert epoch_times == [1.5, 2.5]


def test_time_dependent_partial_results_are_not_skippable(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "wells_mhd64" / "fno3d" / "seed_0"
    write_json(run_dir / "config_resolved.json", {"batch_size_effective": 4})
    write_json(run_dir / "metrics_test.json", {"relative_l2": 0.1})

    assert experiments.completed_run_from_disk(run_dir, "wells_mhd64", "fno3d", 0, "time_dependent") is None


def test_select_output_root_resumes_latest_protocol_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    older = tmp_path / "outputs" / "final_experiment_20250101_000000"
    newer = tmp_path / "outputs" / "final_experiment_20250102_000000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older.touch()
    newer.touch()

    out_root, resumed = experiments.select_output_root()

    assert out_root == Path("outputs") / newer.name
    assert resumed is True


def test_cuda_loaders_use_spawn_worker_context() -> None:
    class TinyDataset:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, idx: int) -> dict:
            return {"x": experiments.torch.tensor([idx]), "y": experiments.torch.tensor([idx])}

    loaders = experiments.make_loaders(
        {"train": TinyDataset()},
        batch_size=1,
        loader_cfg={"num_workers": 1, "persistent_workers": True, "pin_memory": False},
        seed=0,
        device=experiments.torch.device("cuda"),
    )

    context = loaders["train"].multiprocessing_context
    assert context is not None
    assert context.get_start_method() == "spawn"
