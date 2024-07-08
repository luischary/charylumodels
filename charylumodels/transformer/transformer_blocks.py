import torch
import torch.nn as nn
import numpy as np

from charylumodels.transformer.activations import NewGELU
from charylumodels.transformer.attention import qkv_attention


class PositionalEncoding(nn.Module):
    def __init__(
        self, model_dimension, dropout_probability, expected_max_sequence_length=5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(
            10000.0,
            -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension,
        )

        positional_encodings_table = torch.zeros(
            expected_max_sequence_length, model_dimension
        )
        positional_encodings_table[:, 0::2] = torch.sin(
            position_id * frequencies
        )  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(
            position_id * frequencies
        )  # cosine on odd positions

        self.register_buffer("positional_encodings_table", positional_encodings_table)

    def forward(self, embeddings_batch):
        assert (
            embeddings_batch.ndim == 3
            and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1]
        ), f"Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}"

        positional_encodings = self.positional_encodings_table[
            : embeddings_batch.shape[1]
        ]

        return self.dropout(embeddings_batch + positional_encodings)


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        q = self.reshape_for_attention(self.proj_q(x))
        k = self.reshape_for_attention(self.proj_k(x))
        v = self.reshape_for_attention(self.proj_v(x))

        x_att = qkv_attention(q, k, v, mask)
        # dropout
        x_att = self.dropout_attention(x_att)

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att = self.reshape_from_attention(x_att)

        # projecao final
        x_att = self.out_projection(x_att)
        x_att = self.dropout_projection(x_att)
        return x_att


class MultiHeadCrossAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        q = self.reshape_for_attention(self.proj_q(x))
        k = self.reshape_for_attention(self.proj_k(y))
        v = self.reshape_for_attention(self.proj_v(y))

        x_att = qkv_attention(q, k, v, mask)
        x_att = self.dropout_attention(x_att)

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att = self.reshape_from_attention(x_att)

        # projecao final
        x_att = self.out_projection(x_att)
        x_att = self.dropout_projection(x_att)
        return x_att


class FeedFowardBlock(nn.Module):
    def __init__(self, embed_dim, hidden_size, dropout: float = 0.1):
        super().__init__()

        self.ff_1 = nn.Linear(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.ff_2 = nn.Linear(hidden_size, embed_dim)
        self.activation = NewGELU()

    def forward(self, x):
        x_1 = self.ff_1(x)
        x_act = self.activation(x_1)
        x_drop = self.dropout(x_act)
        x_2 = self.ff_2(x_drop)
        return x_2


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim)
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.drop_skip_1 = nn.Dropout(dropout)
        self.drop_skip_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.drop_skip_1(x) + self.attention(self.norm_1(x), mask)
        x = self.drop_skip_2(x) + self.feedforward(self.norm_2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads, embed_dim=embed_dim, dropout=dropout
        )
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.drop_skip_1 = nn.Dropout(dropout)
        self.drop_skip_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm_1(self.drop_skip_1(x) + self.attention(x, mask))
        x = self.norm_2(self.drop_skip_2(x) + self.feedforward(x))
        return x
