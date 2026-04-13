from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.cuda.amp as amp
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from recbole.trainer import Trainer
from recbole.utils import get_gpu_usage, set_color


class TrackedTrainer(Trainer):
    """RecBole trainer with lightweight JSONL batch-progress logging."""

    def __init__(self, config, model):
        super().__init__(config, model)
        progress_file = config["training_progress_file"] if "training_progress_file" in config else "training_progress.jsonl"
        self.progress_path = Path(progress_file)
        self.progress_interval = int(config["progress_log_interval"] if "progress_log_interval" in config else 10)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_progress(self, epoch_idx: int, batch_idx: int, total_batches: int, loss_value: float) -> None:
        payload = {
            "model": str(self.config["model"]),
            "seed": int(self.config["seed"]),
            "epoch": int(epoch_idx),
            "batch": int(batch_idx),
            "total_batches": int(total_batches),
            "loss": round(float(loss_value), 6),
            "ts": time.time(),
        }
        with self.progress_path.open("a") as f:
            json.dump(payload, f)
            f.write("\n")

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        total_batches = len(train_data)
        iter_data = (
            tqdm(
                train_data,
                total=total_batches,
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=self.enable_scaler)
        else:  # pragma: no cover - fallback for older torch versions
            scaler = amp.GradScaler(enabled=self.enable_scaler)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )

            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()

            if batch_idx % self.progress_interval == 0 or batch_idx + 1 == total_batches:
                self._append_progress(
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    loss_value=float(loss.item()),
                )

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

        return total_loss
