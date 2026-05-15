from __future__ import annotations

import csv
from pathlib import Path

import torch
from tqdm import tqdm

from .evaluator import evaluate
from .losses import mhd_loss
from src.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, train_loader, valid_loader, run_dir, device, cfg):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.run_dir = Path(run_dir)
        self.device = device
        self.cfg = cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
        self.sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(1, cfg["epochs"]))
            if cfg.get("cosine_schedule", True)
            else None
        )
        self.use_amp = bool(cfg.get("mixed_precision", False)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def fit(self):
        log_path = self.run_dir / "train_log.csv"
        best = float("inf")
        rows = []
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()
            losses = []
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False):
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                self.opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred = self.model(x)
                    loss = mhd_loss(
                        pred,
                        y,
                        self.cfg.get("lambda_rel", 0.1),
                        self.cfg.get("lambda_div", 0.0),
                        self.cfg.get("magnetic_field_indices"),
                        self.cfg.get("spacing"),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("NaN/Inf loss detected")
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("grad_clip_norm", 1.0))
                self.scaler.step(self.opt)
                self.scaler.update()
                losses.append(float(loss.detach().cpu()))
            if self.sched:
                self.sched.step()
            val = (
                evaluate(self.model, self.valid_loader, self.device, self.cfg.get("magnetic_field_indices"), self.cfg.get("spacing"))
                if self.valid_loader
                else {"relative_l2": float("nan")}
            )
            row = {
                "epoch": epoch,
                "train_loss": sum(losses) / max(1, len(losses)),
                "lr": self.opt.param_groups[0]["lr"],
                **{f"valid_{k}": v for k, v in val.items() if isinstance(v, (int, float))},
            }
            rows.append(row)
            if val.get("relative_l2", float("inf")) < best:
                best = val["relative_l2"]
                save_checkpoint(self.run_dir / "best_model.pt", self.model, self.opt, epoch, val)
        save_checkpoint(self.run_dir / "last_model.pt", self.model, self.opt, self.cfg["epochs"], rows[-1] if rows else {})
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
        return rows
