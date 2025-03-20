from dataclasses import dataclass

import torch
import torch.nn as nn

from charylumodels.transformer.attention_blocks import (
    RotaryMultiHeadFlashAttention,
    RotaryMHFlashCrossAttention,
)
from charylumodels.transformer.transformer_blocks import FeedFowardBlock
from charylumodels.transformer.rotary import precompute_freqs_cis
from charylumodels.transformer.utils import TransformerCache


@dataclass
class ModelArgs:
    vocab_size: int
    num_layers: int
    embed_dim: int
    hidden_size: int
    num_heads: int
    dropout: float = 0.1
    window_size: int = -1
    max_seq_len: int = 2048
    rotary_theta: int = 10_000
    make_cross_attention: bool = False
    use_cache: bool = False


class DecoderBlock(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.attention = RotaryMultiHeadFlashAttention(
            num_heads=params.num_heads,
            embed_dim=params.embed_dim,
            dropout=params.dropout,
            causal=True,
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

        if params.make_cross_attention:
            self.norm_3 = nn.LayerNorm(normalized_shape=params.embed_dim)
            self.drop_skip_3 = nn.Dropout(params.dropout)
            self.cross_attention = RotaryMHFlashCrossAttention(
                num_heads=params.num_heads,
                embed_dim=params.embed_dim,
                dropout=params.dropout,
                causal=False,
                window_size=-1,  # nao faz sentido para o cross attention.... pode dar ruim se mudar isso
            )

        if params.use_cache:
            self.cross_attention_cache = TransformerCache()
            self.attention_cache = TransformerCache()
        else:
            self.cross_attention_cache = None
            self.attention_cache = None

        self.make_cross_attention = params.make_cross_attention

    def forward(self, x, rotary_freqs, x_cross=None, rotary_freqs_cross=None):
        x = self.drop_skip_1(x) + self.attention(
            self.norm_1(x), rotary_freqs, self.attention_cache
        )

        # caso tenha cross attention
        if self.make_cross_attention:
            x = self.drop_skip_3(x) + self.cross_attention(
                self.norm_3(x),
                x_cross,
                rotary_freqs,
                self.cross_attention_cache,
                rotary_freqs_cross,
            )

        x = self.drop_skip_2(x) + self.feedforward(self.norm_2(x))
        return x

    def clear_cache(self):
        if self.attention_cache is not None:
            self.attention_cache.clear_cache()

        if self.cross_attention_cache is not None:
            self.cross_attention_cache.clear_cache()


class Decoder(nn.Module):
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
        self.cache = []
        for _ in range(params.num_layers):
            self.layers.append(DecoderBlock(params))

        self.output_norm = nn.LayerNorm(normalized_shape=params.embed_dim)

    def forward(self, x, x_cross=None, pos=0):
        B, seq_len = x.shape
        if seq_len > 1:
            x = x[:, -1:]

        rotary_freqs = self.rotary_freqs[pos:].to(x.device)

        if x_cross is not None:
            rotary_freqs_cross = self.rotary_freqs[:].to(
                x.device
            )  # cross sempre vai ser 1 vez so
        else:
            rotary_freqs_cross = None

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, rotary_freqs, x_cross, rotary_freqs_cross)

        # ja retorna uma saida normalizada
        # nenhum wrapper precisa normalizar a saida
        return self.output_norm(x)

    def reset_cache(self):
        for l in self.layers:
            l.clear_cache()


class DecoderLM(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        self.embed_dim = params.embed_dim
        self.decoder = Decoder(params=params)
        self.lm_head = nn.Linear(params.embed_dim, params.vocab_size)

    def forward(self, x, y=None, x_cross=None):
        last_hidden_states = self.decoder(x, x_cross)
        logits = self.lm_head(last_hidden_states)

        if y is not None:
            B, L, V = logits.shape
            loss = nn.functional.cross_entropy(logits.reshape((-1, V)), y)
            return logits, loss
        else:
            return logits
