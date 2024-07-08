from typing import Any, Dict

import torch
import torch.nn as nn
import lightning.pytorch as pl

from transformer.vanilla.encoder import EncoderMLM
from transformer.schedulers import CosineWarmupScheduler


class VanillaEncoderMLMLightning(pl.LightningModule):
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

        self.model = EncoderMLM(**encoder_params)

    def training_step(self, batch, batch_idx):
        x, y, masked_mask = batch
        # first get the masked_ids to use later
        # flattens the masked id so its easier to deal with
        masked_ids = torch.flatten(masked_mask.reshape((-1,)).nonzero())
        # also flat the labels
        labels = y.reshape((-1,))[masked_ids]
        # runs through the model
        out = self.model(x, masked_mask)
        loss = torch.nn.functional.cross_entropy(out, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = bnb.optim.AdamW8bit(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     betas=(0.9, 0.95),
        #     weight_decay=1e-2,
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

    def generate_text(self, **kwargs):
        return self.model.generate_text(**kwargs)

    def forward(self, x, mask):
        return self.model(x, mask)
