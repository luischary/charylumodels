import math

import torch
import torch.nn as nn
import numpy as np


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(
            n
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int = 8, embed_dim: int = 512, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert (
            embed_dim % num_heads == 0
        ), "The number of dimensions must be divible by the number of heads"

        self.head_dim = embed_dim // num_heads
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)

        self.proj_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_projection = nn.Dropout(dropout)

    def reshape_for_attention(self, x):
        B, L, E = x.shape
        # shape x = [batch, len, embed_dim]
        # precisa virar [batch * heads, len, head_dim]
        x = x.contiguous().view((B, L, self.num_heads, self.head_dim)).transpose(1, 2)
        # virou [batch, heads, len, head_dim]
        # x = x.contiguous().view((B * self.num_heads, L, self.head_dim))
        return x

    def reshape_from_attention(self, x):
        B, H, L, HD = x.shape
        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x = x.transpose(1, 2)
        # virou [batch, len, heads, head_dim]
        x = x.contiguous().view((B, L, self.embed_dim))
        # virou [batch, len, embed_dim]
        return x

    def QKVattention(self, q, k, v, mask=None):
        b, heads, len_tokens, embed_dim = q.shape
        k_t = torch.transpose(k, -1, -2)
        # shapes for q, k, v are [B, HEADS, SEQ, HEAD_DIM]
        # for K_t we have [B, HEADS, HEAD_DIM, SEQ]
        qk = torch.einsum("bhsd, bhde -> bhse", q, k_t)
        # qk = torch.bmm(q, k_t)
        # shape of qk is [B, SEQ, SEQ]
        if mask is not None:
            qk = qk + mask
        attention = torch.softmax(qk / np.sqrt(embed_dim), dim=-1)
        attention = self.dropout_attention(attention)
        # [batch, heads, decoder_len, head_dim] * [batch, heasd, encoder_len, head_dim]
        full_attention = torch.einsum("bhde, bher -> bhdr", attention, v)
        return self.dropout_projection(full_attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        q = self.reshape_for_attention(self.proj_q(x))
        k = self.reshape_for_attention(self.proj_k(x))
        v = self.reshape_for_attention(self.proj_v(x))

        x_att = self.QKVattention(q, k, v, mask)

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att = self.reshape_from_attention(x_att)

        # projecao final
        x_att = self.out_projection(x_att)
        return x_att


class FeedFowardBlock(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout: int = 0.1):
        super().__init__()

        self.ff_1 = nn.Linear(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ff_2 = nn.Linear(hidden_size, embed_dim)
        self.activation = NewGELU()

    def forward(self, x):
        x = self.ff_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff_2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, dropout: int = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x, mask):
        x = x + self.attention(self.norm_1(x), mask)
        x = x + self.feedforward(self.norm_2(x))
        return x


class T5Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        hidden_size,
        dropout: int = 0.1,
        max_att_window: int = 128,
        pos_buckets: int = 32,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.max_attention_window = max_att_window
        self.pos_buckets = pos_buckets

        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_blocks.append(
                DecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
            )

        self.last_norm = nn.LayerNorm(normalized_shape=embed_dim)

        self.relative_embedding = nn.Embedding(
            num_embeddings=self.pos_buckets, embedding_dim=self.num_heads
        )

    def forward(self, x):
        causal_mask = self.create_future_mask(x)
        x = self.embedding(x)

        for block in self.decoder_blocks:
            x = block(x, causal_mask)

        return self.last_norm(x)

    def create_future_mask(self, tensor):
        # positional encoding
        context_position = torch.arange(tensor.shape[1], dtype=torch.long)[:, None]
        memory_position = torch.arange(tensor.shape[1], dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position = relative_position.abs()
        # como e decoder
        relative_pos_indexes = torch.max(
            relative_position, torch.zeros_like(relative_position)
        )
        # base da escala log
        base = (self.max_attention_window) ** (1 / (self.pos_buckets - 1))
        n_buckets = torch.max(
            torch.log((relative_pos_indexes.float())) / math.log(base),
            torch.zeros_like(relative_pos_indexes),
        ).to(torch.long)
        # limita o indice do bucket
        n_buckets = torch.min(
            relative_pos_indexes,
            torch.ones_like(relative_pos_indexes) * (self.pos_buckets - 1),
        ).to(tensor.device)
        embedded = self.relative_embedding(n_buckets)
        # arruma o shape para [heads, q, k]
        embedded = embedded.permute([2, 0, 1])
        # do jeito que esta pode ser aplicado num encoder, precisamos tornar causal
        causal_mask = (
            torch.tril(torch.ones((tensor.shape[1], tensor.shape[1])))
            .view((1, tensor.shape[1], tensor.shape[1]))
            .repeat((self.num_heads, 1, 1))
        ).to(tensor.device)
        future_mask = embedded.masked_fill(causal_mask == 0, float("-inf"))
        future_mask = future_mask.unsqueeze(0).to(tensor.device)
        return future_mask


class T5DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        hidden_size,
        dropout=0.1,
        pad_token_id: int = 0,
        max_att_window: int = 128,
        pos_buckets: int = 32,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.decoder = T5Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout=dropout,
            max_att_window=max_att_window,
            pos_buckets=pos_buckets,
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
