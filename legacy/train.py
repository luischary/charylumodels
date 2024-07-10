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
from datasets import TokenizedDecoderMLMDataset
from alibi.alibi_transformer import AlibiDecoderLM
from t5.t5_transformer import T5DecoderLM

torch.backends.cuda.matmul.allow_tf32 = True

if __name__ == "__main__":
    vocab_size = 60000
    max_len = 128
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = MyTokenizer(tokenizer_path="artifacts/marketplace_tokenizer")
    # tokenizer = MyTokenizer(tokenizer_path="artifacts/movies_tokenizer")
    # tokenizer = MyTokenizer(tokenizer_path="artifacts/wiki_tokenizer")

    dataset = TokenizedDecoderMLMDataset(
        tokenizer=tokenizer,
        text_folders=["tokens_marketplace_128"],
        max_len=max_len,
        limit=None,
        begin_sentence_token=1,
        end_of_sentence_token=2,
        pad_token_id=0,
        vocab_size=60000,
    )

    modelo = AlibiDecoderLM(
        vocab_size=vocab_size,
        embed_dim=256,
        num_layers=12,
        num_heads=8,
        hidden_size=256 * 4,
        dropout=0.1,
        pad_token_id=0,
        tokens_per_sample=128,
    )
    # modelo = T5DecoderLM(
    #     vocab_size=vocab_size,
    #     embed_dim=256,
    #     num_layers=12,
    #     num_heads=8,
    #     hidden_size=256 * 4,
    #     dropout=0.1,
    #     pad_token_id=0,
    #     max_att_window=64,
    #     pos_buckets=64,
    # )
    # modelo.load_state_dict(torch.load("modelos/teste_alibi_marketplace_19.pt"))
    modelo.to(device)

    optimizer = torch.optim.AdamW(modelo.parameters(), lr=1e-5)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, shuffle=True, batch_size=batch_size
    )
    writer = SummaryWriter("logs/teste_alibi_exponential_marketplace")
    update_count = 0
    accum_loss = None
    acumulation_steps = 1

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
            # runs through the model
            out = modelo(x, non_pad_indexes)
            loss = torch.nn.functional.cross_entropy(out, labels) / acumulation_steps
            loss.backward()

            update_count += 1
            if update_count % acumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

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
            f"modelos/teste_alibi_exponential_marketplace_{epoca}.pt",
        )
