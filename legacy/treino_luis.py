from typing import List
from pathlib import Path
import pickle
from datetime import datetime


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# from torch_optimizer.adafactor import Adafactor

import bitsandbytes as bnb

from tokenizers.tokenizer import MyTokenizer
from transformer import TransformerDecoder
from datasets import TokenizedDecoderMLMDataset
from alibi_transformer import AlibiDecoderLM

torch.backends.cuda.matmul.allow_tf32 = True


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, min_percent=0.01):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_percent = min_percent
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = min(self.get_lr_factor(epoch=self.last_epoch), self.min_percent)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_layers,
        num_heads,
        hidden_size,
        dropout=0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size)

        n_params = sum(p.numel() for p in self.decoder.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, non_pad_indexes=None):
        # we dont want to compute loss for padding tokens
        # get all hidden states
        logits = self.lm_head(self.decoder(x))
        # remove batch dimension
        logits = torch.reshape(logits, (-1, self.vocab_size))
        # get only the tokens that matter
        if non_pad_indexes is not None:
            logits = logits[non_pad_indexes, :]

        return logits


if __name__ == "__main__":
    vocab_size = 60000
    max_len = 512
    batch_size = 1
    acumulation_steps = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = MyTokenizer(tokenizer_path="artifacts/wiki_tokenizer")

    # dataset = DecoderMLMDataset(
    #     tokenizer=tokenizer,
    #     text_folders=[r"D:\DATASETS\NLP\Wikipedia\wiki"],
    #     max_len=max_len,
    #     limit=None,
    #     begin_sentence_token=1,
    #     end_of_sentence_token=2,
    #     pad_token_id=0,
    #     vocab_size=60000,
    # )

    dataset = TokenizedDecoderMLMDataset(
        tokenizer=tokenizer,
        text_folders=["tokens_wiki_512"],
        max_len=max_len,
        limit=None,
        begin_sentence_token=1,
        end_of_sentence_token=2,
        pad_token_id=0,
        vocab_size=60000,
    )

    # modelo base
    # modelo = DecoderLM(
    #     vocab_size=vocab_size,
    #     max_len=max_len,
    #     embed_dim=2048,
    #     num_layers=24,
    #     num_heads=16,
    #     hidden_size=2048 * 3,
    #     dropout=0.1,
    #     pad_token_id=0,
    # )
    modelo = AlibiDecoderLM(
        vocab_size=60000,
        embed_dim=2048,
        num_layers=24,
        num_heads=16,
        hidden_size=2048 * 4,
        dropout=0.2,
        pad_token_id=0,
        max_tokens=4096,
        tokens_per_sample=4096,
    )
    modelo.to(device)
    # modelo.load_state_dict(torch.load("wiki_decoder_2.pt"))
    # optimizer = torch.optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-2)
    # optimizer = Adafactor(modelo.parameters(), lr=1e-4)
    optimizer = bnb.optim.AdamW8bit(
        modelo.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-2
    )
    scheduler = CosineWarmupScheduler(
        optimizer, warmup=100, max_iters=25000, min_percent=0.1
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, shuffle=True, batch_size=batch_size
    )

    writer = SummaryWriter("logs/wiki_final")
    update_count = 0
    step_count = 0
    accum_loss = None
    checkpoint_count = 0

    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad(set_to_none=True)
    for epoca in range(5):
        for x, y in dataloader:
            x = x.to(device)
            y = y.long().to(device)

            pad_mask = (x != 0).type(torch.int).reshape((-1,))
            non_pad_indexes = torch.flatten(pad_mask.nonzero())

            # also flat the labels
            labels = y.reshape((-1,)).type(torch.long)
            labels = labels[non_pad_indexes]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # runs through the model
                out = modelo(x, non_pad_indexes)
                loss = (
                    torch.nn.functional.cross_entropy(out, labels) / acumulation_steps
                )
            scaler.scale(loss).backward()
            # loss.backward()
            step_count += 1

            if step_count % acumulation_steps == 0:
                # grad clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
                # agora normal
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_count += 1

                writer.add_scalar(
                    "Train/Loss",
                    loss * acumulation_steps,
                    global_step=update_count,
                )

                if accum_loss is not None:
                    accum_loss = (
                        0.99 * accum_loss
                        + 0.01 * loss.detach().cpu().item() * acumulation_steps
                    )
                else:
                    accum_loss = loss.detach().cpu().item() * acumulation_steps

                if update_count % 10 == 0:
                    print(
                        f"{datetime.now().strftime('%H:%M:%S')}\tEpoca: {epoca}\tUpdate: {update_count}/{len(dataloader) // acumulation_steps}\tLoss: {loss.detach().cpu().item()*acumulation_steps:.4f}\tAccum_loss: {accum_loss:.4f}",
                    )

                if update_count % 100 == 0:
                    torch.save(
                        modelo.state_dict(),
                        f"wiki_decoder_{checkpoint_count}.pt",
                    )
                    checkpoint_count += 1
                    if checkpoint_count >= 5:
                        checkpoint_count = 0

                if update_count >= 300000:
                    break

    torch.save(modelo.state_dict(), f"wiki_decoder_final.pt")
