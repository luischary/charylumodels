from typing import Any, Dict
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import torch
import torch.nn as nn
import lightning.pytorch as pl
import bitsandbytes as bnb

from transformer.RetroMae.model import RetroMaeModel
from transformer.schedulers import CosineWarmupScheduler


class RetroMaeLightning(pl.LightningModule):
    def __init__(
        self,
        encoder_params: Dict,
        decoder_params: Dict,
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
        self.decoder_vocab_size = decoder_params["vocab_size"]

        self.model = RetroMaeModel(
            encoder_args=encoder_params, decoder_args=decoder_params
        )

    def training_step(self, batch, batch_idx):
        x, y, masked_mask = batch
        # flat the labels
        labels = y[masked_mask]
        # runs through the model
        encoder_logits, decoder_logits = self.model(x, masked_mask, y)
        mlm_loss = torch.nn.functional.cross_entropy(encoder_logits, labels)

        # para o decoder nao podemos considerar nem o primeiro indice, nem os tokens de padding
        y_no_cls = y[:, 1:]
        y_flat = y_no_cls.reshape((-1,))
        decoder_mask = torch.flatten(
            y_flat.nonzero()
        )  # retira o padding e o cls na maldade
        labels_decoder = y.reshape((-1, 1))
        labels_decoder = labels_decoder[decoder_mask, :].reshape((-1,))
        decoder_logits = decoder_logits.reshape((-1, self.decoder_vocab_size))
        decoder_logits = decoder_logits[decoder_mask, :]
        # pula o primeiro indice pq eh o cls token
        decoder_loss = torch.nn.functional.cross_entropy(decoder_logits, labels_decoder)

        loss = decoder_loss + mlm_loss
        self.log("train/decoder_loss", decoder_loss)
        self.log("train/mlm_loss", mlm_loss)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, masked_mask = batch
        # flat the labels
        labels = y[masked_mask]
        # runs through the model
        encoder_logits, decoder_logits = self.model(x, masked_mask, y)
        mlm_loss = torch.nn.functional.cross_entropy(encoder_logits, labels)

        # para o decoder nao podemos considerar nem o primeiro indice, nem os tokens de padding
        y_no_cls = y[:, 1:]
        y_flat = y_no_cls.reshape((-1,))
        decoder_mask = torch.flatten(
            y_flat.nonzero()
        )  # retira o padding e o cls na maldade
        labels_decoder = y.reshape((-1, 1))
        labels_decoder = labels_decoder[decoder_mask, :].reshape((-1,))
        decoder_logits = decoder_logits.reshape((-1, self.decoder_vocab_size))
        decoder_logits = decoder_logits[decoder_mask, :]
        # pula o primeiro indice pq eh o cls token
        decoder_loss = torch.nn.functional.cross_entropy(decoder_logits, labels_decoder)

        loss = decoder_loss + mlm_loss
        self.log("validation/decoder_loss", decoder_loss)
        self.log("validation/mlm_loss", mlm_loss)
        self.log("validationloss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=1e-2,
        )
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

    def forward(self, x, mask):
        return self.model(x, mask)

    def embed(self, x):
        return self.model.embed(x)
