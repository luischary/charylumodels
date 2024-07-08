from typing import Any, Dict

import torch
import torch.nn as nn
import lightning.pytorch as pl

import bitsandbytes as bnb

# import deepspeed

from transformer.hierarchical.encoder import (
    HierarchicalEncoderMLM,
    RotaryHierarchicalEncoderMLM,
)
from transformer.schedulers import CosineWarmupScheduler


class HierarchicalEncoderMLMLightning(pl.LightningModule):
    def __init__(
        self,
        encoder_params: Dict,
        learning_rate: float = 1e-4,
        min_lr_percent: float = 0.1,
        warmup_steps: int = 500,
        total_training_steps: int = 1e6,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.min_lr_percent = min_lr_percent
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps

        self.model = RotaryHierarchicalEncoderMLM(**encoder_params)

    def training_step(self, batch, batch_idx):
        x, y, masked_mask = batch
        # se o shape tiver uma dimensao a mais a gente tira
        if len(x.shape) > 2:
            x = x[0]
            y = y[0]
            masked_mask = masked_mask[0]
        # first get the masked_ids to use later
        # flattens the masked id so its easier to deal with
        masked_ids = torch.flatten(masked_mask.reshape((-1,)).nonzero())
        # also flat the labels
        labels = y.reshape((-1,))[masked_ids]
        # runs through the model
        out = self.model(x, masked_mask, insert_global_token=True)
        loss = torch.nn.functional.cross_entropy(out, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=1e-2,
        )
        # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
        #     self.parameters(), lr=self.learning_rate, betas=(0.9, 0.98)
        # )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup=self.warmup_steps,
            max_iters=self.total_training_steps,
            min_percent=self.min_lr_percent,
        )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def forward(self, x, mask, insert_global_token: bool = True):
        return self.model(x, mask, insert_global_token=insert_global_token)
