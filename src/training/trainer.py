from __future__ import annotations

import csv
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .evaluator import evaluate
from .losses import mhd_loss
from src.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, train_loader, valid_loader, run_dir, device, cfg):
        self.original_model = model.to(device)
        self.model = self.original_model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.run_dir = Path(run_dir)
        self.device = device
        self.cfg = cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.compile_failed = False
        self.compile_error: str | None = None
        if bool(cfg.get("use_compile", False)) and device.type == "cuda" and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode=cfg.get("compile_mode", "reduce-overhead"))
            except Exception as exc:  # pragma: no cover - depends on local torch/compiler stack
                self.compile_failed = True
                self.compile_error = str(exc)
                print(f"WARNING: torch.compile failed; continuing uncompiled: {exc}")
        self.opt = torch.optim.AdamW(self.original_model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
        self.sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(1, cfg["epochs"]))
            if cfg.get("cosine_schedule", True)
            else None
        )
        self.use_amp = bool(cfg.get("use_amp", cfg.get("mixed_precision", False))) and device.type == "cuda"
        self.amp_dtype = self._resolve_amp_dtype(cfg.get("amp_dtype", "bf16"))
        self.use_scaler = self.use_amp and self.amp_dtype == torch.float16 and bool(cfg.get("use_grad_scaler", False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

    @staticmethod
    def _resolve_amp_dtype(name: str | torch.dtype) -> torch.dtype:
        if isinstance(name, torch.dtype):
            return name
        key = str(name).lower()
        if key in {"bf16", "bfloat16", "torch.bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half", "torch.float16"}:
            return torch.float16
        return torch.bfloat16

    def _amp_context(self):
        if self.use_amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def fit(self):
        log_path = self.run_dir / "train_log.csv"
        best = float("inf")
        rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()
            losses = []
            epoch_start = time.perf_counter()
            data_time = forward_time = backward_time = step_time = 0.0
            samples_seen = 0
            iter_ready = time.perf_counter()
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False):
                data_time += time.perf_counter() - iter_ready
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                samples_seen += int(x.shape[0])
                self.opt.zero_grad(set_to_none=True)
                t0 = time.perf_counter()
                with self._amp_context():
                    pred = self.model(x)
                    loss = mhd_loss(
                        pred,
                        y,
                        self.cfg.get("lambda_rel", 0.1),
                        self.cfg.get("lambda_div", 0.0),
                        self.cfg.get("magnetic_field_indices"),
                        self.cfg.get("spacing"),
                    )
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                forward_time += time.perf_counter() - t0
                if not torch.isfinite(loss):
                    raise FloatingPointError("NaN/Inf loss detected")
                t1 = time.perf_counter()
                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                else:
                    loss.backward()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                backward_time += time.perf_counter() - t1
                t2 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(self.original_model.parameters(), self.cfg.get("grad_clip_norm", 1.0))
                if self.use_scaler:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                step_time += time.perf_counter() - t2
                losses.append(float(loss.detach().cpu()))
                iter_ready = time.perf_counter()
            if self.sched:
                self.sched.step()
            val = (
                evaluate(self.model, self.valid_loader, self.device, self.cfg.get("magnetic_field_indices"), self.cfg.get("spacing"), use_amp=self.use_amp, amp_dtype=self.amp_dtype)
                if self.valid_loader
                else {"relative_l2": float("nan")}
            )
            total_epoch_time = time.perf_counter() - epoch_start
            samples_per_sec = samples_seen / max(total_epoch_time, 1e-12)
            mem_alloc = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
            mem_reserved = torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
            timing = {
                "epoch": epoch,
                "data_loading_time": data_time,
                "forward_time": forward_time,
                "backward_time": backward_time,
                "optimizer_step_time": step_time,
                "total_epoch_time": total_epoch_time,
                "samples_per_sec": samples_per_sec,
                "gpu_memory_allocated": mem_alloc,
                "gpu_memory_reserved": mem_reserved,
            }
            timing_rows.append(timing)
            print(
                f"Epoch {epoch} timing: data={data_time:.3f}s forward={forward_time:.3f}s backward={backward_time:.3f}s "
                f"step={step_time:.3f}s total={total_epoch_time:.3f}s samples/sec={samples_per_sec:.2f} "
                f"gpu_alloc={mem_alloc/1e9:.2f}GB gpu_reserved={mem_reserved/1e9:.2f}GB"
            )
            row = {
                "epoch": epoch,
                "train_loss": sum(losses) / max(1, len(losses)),
                "lr": self.opt.param_groups[0]["lr"],
                **timing,
                **{f"valid_{k}": v for k, v in val.items() if isinstance(v, (int, float))},
            }
            rows.append(row)
            if val.get("relative_l2", float("inf")) < best:
                best = val["relative_l2"]
                save_checkpoint(self.run_dir / "best_model.pt", self.original_model, self.opt, epoch, val)
        save_checkpoint(self.run_dir / "last_model.pt", self.original_model, self.opt, self.cfg["epochs"], rows[-1] if rows else {})
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
        with open(self.run_dir / "epoch_timing.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in timing_rows for k in r}))
            writer.writeheader()
            writer.writerows(timing_rows)
        return rows
