from dataclasses import dataclass

import torch
import torch.nn as nn

from charylumodels.transformer.attention_blocks import RotaryMultiHeadFlashAttention
from charylumodels.transformer.transformer_blocks import FeedFowardBlock
from charylumodels.transformer.rotary import precompute_freqs_cis


@dataclass
class ModelArgs:
    vocab_size: int
    num_layers: int
    embed_dim: int
    hidden_size: int
    num_heads: int
    dropout: float = 0.1
    window_size: int = -1
    rotary_theta: int = 10_000

    max_seq_len: int = 2048


class EncoderBlock(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.attention = RotaryMultiHeadFlashAttention(
            num_heads=params.num_heads,
            embed_dim=params.embed_dim,
            dropout=params.dropout,
            causal=False,
            window_size=params.window_size,
        )
        self.feedforward = FeedFowardBlock(
            embed_dim=params.embed_dim,
            hidden_size=params.hidden_size,
            dropout=params.dropout,
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=params.embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=params.embed_dim)
        self.drop_skip_1 = nn.Dropout(params.dropout)
        self.drop_skip_2 = nn.Dropout(params.dropout)

    def forward(self, x, rotary_freqs):
        x = self.drop_skip_1(x) + self.attention(self.norm_1(x), rotary_freqs)
        x = self.drop_skip_2(x) + self.feedforward(self.norm_2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.rotary_freqs = precompute_freqs_cis(
            dim=params.embed_dim // params.num_heads,
            end=params.max_seq_len,
            theta=params.rotary_theta,
        )
        self.embedding = nn.Embedding(
            num_embeddings=params.vocab_size, embedding_dim=params.embed_dim
        )

        self.layers = nn.ModuleList()
        for _ in range(params.num_layers):
            self.layers.append(EncoderBlock(params))

        self.output_norm = nn.LayerNorm(normalized_shape=params.embed_dim)

    def forward(self, x):
        B, seq_len = x.shape
        rotary_freqs = self.rotary_freqs[:seq_len].to(x.device)

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, rotary_freqs)

        # ja retorna uma saida normalizada
        # nenhum wrapper precisa normalizar a saida
        return self.output_norm(x)


class EncoderMLM(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.embed_dim = params.embed_dim

        self.encoder = Encoder(params=params)

        self.mlm_head = nn.Linear(params.embed_dim, params.vocab_size)

    def forward(self, x, mask, return_hidden_states: bool = False):
        # first get the masked_ids to use later
        # flattens the masked id so its easier to deal with
        masked_ids = torch.flatten(mask.reshape((-1,)).nonzero())
        # get all hidden states (they are already normalized)
        last_hidden_states = self.encoder(x)
        # flatten everything so we can compute everything at once
        all_hidden_states = last_hidden_states.reshape((-1, self.embed_dim))
        # get only the masked hidden states
        masked_hidden_states = all_hidden_states[masked_ids, :]
        # predicts only the masked tokens
        logits = self.mlm_head(masked_hidden_states)

        if return_hidden_states:
            return logits, last_hidden_states

        return logits

    def embed(self, x):
        return self.encoder(x)
