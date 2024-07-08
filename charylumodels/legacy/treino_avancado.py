from typing import List
from pathlib import Path
import pickle


import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from tokenizers.tokenizer import MyTokenizer
from transformer import TransformerDecoder

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


class DecoderMLMDataset(Dataset):
    def __init__(
        self,
        tokenizer: MyTokenizer,
        text_folders: List[str],
        max_len: int = 512,
        limit: int = None,
        begin_sentence_token: int = 1,
        end_of_sentence_token: int = 0,
        pad_token_id: int = 2,
        vocab_size: int = 60000,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.begin_sentence_token = begin_sentence_token
        self.end_of_sentence_token = end_of_sentence_token
        self.pad_token_id = pad_token_id

        self.texts_refs = []
        for folder in text_folders:
            self.texts_refs += list(Path(folder).glob("**/*.txt"))

        if limit is not None:
            self.texts_refs = self.texts_refs[:limit]

    def __len__(self):
        return len(self.texts_refs)

    def __getitem__(self, index):
        text = self.texts_refs[index].read_text(encoding="utf8")
        tokenized = self.tokenizer.tokenize_text(text)
        # for this guys is important to put begin and end of sentence tokens
        tokenized = (
            [self.begin_sentence_token] + tokenized + [self.end_of_sentence_token]
        )
        # for the decoder we must have a maximum length of max_len + 1 for shifted values
        # its ok to pad but the padding must come from left ro rigth
        if len(tokenized) < self.max_len + 1:
            diff = (self.max_len + 1) - len(tokenized)
            tokenized = [self.pad_token_id for _ in range(diff)] + tokenized
        elif len(tokenized) > self.max_len + 1:
            tokenized = tokenized[: self.max_len + 1]

        # transform into tensor
        tokenized = torch.from_numpy(np.array(tokenized))

        decoder_input = tokenized[: self.max_len]
        desired_output = tokenized[1 : self.max_len + 1]

        return decoder_input, desired_output


class TokenizedDecoderMLMDataset(Dataset):
    def __init__(
        self,
        tokenizer: MyTokenizer,
        text_folders: List[str],
        max_len: int = 512,
        limit: int = None,
        begin_sentence_token: int = 1,
        end_of_sentence_token: int = 0,
        pad_token_id: int = 2,
        vocab_size: int = 60000,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.begin_sentence_token = begin_sentence_token
        self.end_of_sentence_token = end_of_sentence_token
        self.pad_token_id = pad_token_id

        self.texts_refs = []
        for folder in text_folders:
            self.texts_refs += list(Path(folder).glob("**/*.pkl"))

        if limit is not None:
            self.texts_refs = self.texts_refs[:limit]

    def __len__(self):
        return len(self.texts_refs)

    def __getitem__(self, index):
        tokenized = None
        with open(self.texts_refs[index], "rb") as file:
            tokenized = pickle.load(file)
        # for the decoder we must have a maximum length of max_len + 1 for shifted values
        # its ok to pad but the padding must come from left ro rigth
        if len(tokenized) < self.max_len + 1:
            diff = (self.max_len + 1) - len(tokenized)
            tokenized = [self.pad_token_id for _ in range(diff)] + tokenized
        elif len(tokenized) > self.max_len + 1:
            tokenized = tokenized[: self.max_len + 1]

        # transform into tensor
        tokenized = torch.from_numpy(np.array(tokenized))

        decoder_input = tokenized[: self.max_len]
        desired_output = tokenized[1 : self.max_len + 1]

        return decoder_input, desired_output


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
    max_len = 128
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer = MyTokenizer(tokenizer_path="artifacts/marketplace_tokenizer")
    tokenizer = MyTokenizer(tokenizer_path="artifacts/wiki_tokenizer")

    dataset = TokenizedDecoderMLMDataset(
        tokenizer=tokenizer,
        text_folders=["tokens_wiki"],
        max_len=512,
        limit=None,
        begin_sentence_token=1,
        end_of_sentence_token=2,
        pad_token_id=0,
        vocab_size=60000,
    )

    modelo = DecoderLM(
        vocab_size=vocab_size,
        max_len=512,
        embed_dim=1024,
        num_layers=24,
        num_heads=16,
        hidden_size=1024 * 2,
        dropout=0.1,
        pad_token_id=0,
    )
    modelo.to(device)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=1e-4)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup=10000, max_iters=2000000, min_percent=0.01
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, shuffle=True, batch_size=batch_size
    )
    writer = SummaryWriter("logs/decoder_wiki_base")
    update_count = 0
    accum_loss = None
    acumulation_steps = 4
    scaler = torch.cuda.amp.GradScaler()

    optimizer.zero_grad(set_to_none=True)
    for epoca in range(20):
        progress = tqdm(
            dataloader,
            total=len(dataloader),
        )
        for x, y in progress:
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

            update_count += 1
            if update_count % acumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            writer.add_scalar("Train/Loss", loss, global_step=update_count)

            if accum_loss is not None:
                accum_loss = (
                    0.99 * accum_loss
                    + 0.01 * loss.detach().cpu().item() * acumulation_steps
                )
            else:
                accum_loss = loss.detach().cpu().item() * acumulation_steps

            progress.set_description(
                f"Epoca: {epoca}\tUpdate: {update_count}\tLoss: {loss.detach().cpu().item() * acumulation_steps:.4f}\tAccum_loss: {accum_loss:.4f}"
            )

    torch.save(
        modelo.state_dict(),
        f"pre_treinados/base/wiki_final.pt",
    )
