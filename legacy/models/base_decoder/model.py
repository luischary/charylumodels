from typing import List

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from charylumodels.transformer.attention_blocks import RotaryMultiHeadFlashAttention
from transformer.transformer_blocks import FeedFowardBlock
from transformer.utils import TransformerCache


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_size,
        rotary,
        dropout: float = 0.1,
        window_size: int = -1,
    ):
        super().__init__()

        self.attention = RotaryMultiHeadFlashAttention(
            rotary=rotary,
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout=dropout,
            causal=True,
            window_size=window_size,
        )
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.drop_skip_1 = nn.Dropout(dropout)
        self.drop_skip_2 = nn.Dropout(dropout)

    def forward(self, x, cache: TransformerCache = None):
        if cache is not None:
            x_att, cache = self.attention(self.norm_1(x), cache)
            x = self.drop_skip_1(x) + x_att
            x = self.drop_skip_2(x) + self.feedforward(self.norm_2(x))
            return x, cache
        else:
            x = self.drop_skip_1(x) + self.attention(self.norm_1(x))
            x = self.drop_skip_2(x) + self.feedforward(self.norm_2(x))
            return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        window_size: int = -1,
        use_xpos: bool = False,
        rotary_frequency: int = 10_000,
    ):
        super().__init__()

        self.rotary = RotaryEmbedding(
            dim=embed_dim // num_heads, theta=rotary_frequency, use_xpos=use_xpos
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                DecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    rotary=self.rotary,
                    dropout=dropout,
                    window_size=window_size,
                )
            )

    def forward(self, x):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        return x

    def forward_with_cache(self, x, caches: List[TransformerCache]):
        x = self.embedding(x)
        for i in range(len(self.layers)):
            x, layer_cache = self.layers[i](x, caches[i])
            caches[i] = layer_cache

        return x, caches


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        window_size: int,
        use_xpos: bool = False,
        rotary_frequency: int = 10_000,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.decoder = Decoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            window_size=window_size,
            rotary_frequency=rotary_frequency,
            use_xpos=use_xpos,
        )

        self.last_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, next_only: bool = False):
        """
        next_only - returns only de logits for predicting the next token
        """
        last_hidden_states = self.decoder(x)
        # normalize everything first
        last_hidden_states_norm = self.last_norm(last_hidden_states)
        if not next_only:
            logits = self.lm_head(last_hidden_states_norm)
        else:
            logits = self.lm_head(last_hidden_states_norm[:, -1, :])

        return logits

    def forward_with_cache(
        self, x, caches: List[TransformerCache], next_only: bool = True
    ):
        last_hidden_states, caches = self.decoder.forward_with_cache(x, caches)
        # normalize everything first
        last_hidden_states_norm = self.last_norm(last_hidden_states)
        if not next_only:
            logits = self.lm_head(last_hidden_states_norm)
        else:
            logits = self.lm_head(last_hidden_states_norm[:, -1, :])

        return logits, caches

    def train_step(self, x, y):
        # flat the labels
        labels = y.reshape((-1,))
        # runs through the model
        out = self.forward(x)
        # flattens the output
        out = out.reshape((-1, self.vocab_size))
        loss = torch.nn.functional.cross_entropy(out, labels)

        return loss
