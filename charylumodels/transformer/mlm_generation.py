from typing import List
from pathlib import Path


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from tokenizers.tokenizer import MyTokenizer


@torch.no_grad()
class MLMPipeline:
    def __init__(
        self,
        mask_token_id: int,
        tokenizer,
        model: torch.nn.Module,
        sos_token: int = 0,
        eos_token: int = 1,
    ):
        self.mask_token_id = mask_token_id
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.tokenizer = tokenizer

        self.model = model
        self.model.eval()

    def predict_mask(
        self,
        sentence: str,
        device,
        top_k: int = 5,
        max_length=512,
        padding="max_length",
        truncation=True,
        pad_input: bool = False,
        pad_token_id: int = 0,
        pad_multiple_of: int = 31,
    ):
        tokenized = self.tokenizer.tokenize_text(
            sentence, padding=padding, max_length=max_length, truncation=truncation
        )
        if pad_input:
            desired_size = (len(tokenized) // pad_multiple_of + 1) * pad_multiple_of
            diff = desired_size - len(tokenized)
            tokenized = [pad_token_id for _ in range(diff)] + tokenized
        # print(self.tokenizer.untokenize_tokens(tokenized))
        tokenized_t = torch.Tensor(tokenized).type(torch.int).to(device)
        masked = tokenized_t == self.mask_token_id
        # masked_indexes = torch.flatten(masked.nonzero())
        # print(tokenized[masked_indexes[0]])
        output = self.model(tokenized_t.unsqueeze(dim=0), masked).squeeze()
        if len(output.shape) < 2:
            output = torch.unsqueeze(output, dim=0)
        # pega o top 5 preditado
        ordered = torch.argsort(torch.softmax(output, dim=1), descending=True)
        if len(ordered.shape) > 1:
            top = [ordered[i][:top_k].tolist() for i in range(ordered.shape[0])]
        else:
            top = [torch.argsort(output, descending=True)[:top_k].tolist()]

        respostas = []
        tokens = []
        probas = []
        for i in range(len(top)):
            tokens.append([])
            probas.append([])
            for k in range(top_k):
                proba = torch.softmax(output, dim=1)[i, top[i][k]]
                token_preditado = self.tokenizer.untokenize_tokens([top[i][k]])
                respostas.append(f"token {i} - {k}: {token_preditado} - {proba:0.5f}")
                tokens[i].append(token_preditado)
                probas[i].append(proba.detach().cpu())
        return respostas, tokens, probas

    def print_pretty(self, input_text, respostas, tokens, probas, top_ans: int = 5):
        probas = np.array(probas)
        tokens = np.array(tokens)
        quantidade_masks = len(tokens)

        for top in range(top_ans):
            nova_frase = input_text
            for m_id in range(quantidade_masks):
                nova_frase = nova_frase.replace("<mask>", tokens[m_id, top], 1)

            print(f"PROBAS: {probas[:, top]}")
            print(nova_frase)

        for r in respostas:
            print(r)
