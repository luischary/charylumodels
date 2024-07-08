from typing import List, Dict
from pathlib import Path
import pickle
import math


import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class TokenizedEncoderMLMDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        max_len: int = 512,
        mask_prob: float = 0.15,
        limit: int = None,
        mask_token_id: int = 1,
        pad_token_id: int = 0,
        special_tokens: List[int] = [],
        vocab_size: int = 60000,
    ):
        super().__init__()

        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

        self.metadata = pd.read_parquet(metadata_path)

        if limit is not None:
            self.metadata = self.metadata[self.metadata.indice <= limit]

    def __len__(self):
        return self.metadata.indice.max()

    def __getitem__(self, index):
        # primeiro localiza qual particao tokenizada
        linha_metadata = self.metadata[self.metadata.indice > index].reset_index(
            drop=True
        )
        matriz = np.load(linha_metadata.loc[0, "tokenized_512_ref"])["arr_0"]
        # qual o indice do cara na matriz?
        final_matriz = linha_metadata.loc[0, "indice"]
        tras_para_frente = final_matriz - index
        tokenized = matriz[-tras_para_frente]
        # agora so seguir

        # before masking its nice to have everything in the right size
        if len(tokenized) < self.max_len:
            diff = self.max_len - len(tokenized)
            tokenized += [self.pad_token_id for _ in range(diff)]
        elif len(tokenized) > self.max_len:
            tokenized = tokenized[: self.max_len]
        # transform into tensor
        tokenized = torch.from_numpy(tokenized)
        # mask time
        probas = torch.rand(tokenized.shape)
        mask = (probas < self.mask_prob) * (tokenized != self.pad_token_id)
        # special tokens
        for special_token in self.special_tokens:
            mask = mask * (tokenized != special_token)

        # now mask tokenized with the mask we just created
        masked = torch.clone(tokenized).type(torch.int)
        masked_ids = torch.flatten(mask.nonzero())
        masked_ids_list = masked_ids.tolist()
        # 80% will be replaced by the mask token
        # 10% no change
        # 10% replaced by random token
        original_masked_tokens = tokenized[masked_ids_list]
        replace_masked_tokens = self.generate_mlm_tokens(
            original_masked_tokens.tolist()
        )
        masked[masked_ids_list] = replace_masked_tokens

        return masked, tokenized, mask

    def generate_mlm_tokens(self, original_tokens: List[int]):
        # 80% will be replaced by the mask token
        # 10% no change
        # 10% replaced by random token
        replace_tokens = np.random.rand(len(original_tokens))
        for i in range(len(original_tokens)):
            if replace_tokens[i] <= 0.8:
                replace_tokens[i] = self.mask_token_id
            elif replace_tokens[i] <= 0.9:
                replace_tokens[i] = np.random.randint(self.vocab_size)
            else:
                replace_tokens[i] = original_tokens[i]
        return torch.from_numpy(replace_tokens).type(torch.int)
