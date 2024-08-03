from dataclasses import dataclass

import torch
import torch.nn as nn
from charylumodels.transformer.rotary import precompute_freqs_cis

from charylumodels.transformer.attention_blocks import RotaryMultiHeadFlashAttention
from charylumodels.transformer.transformer_blocks import FeedFowardBlock


@dataclass
class ConformerParams:
    embed_dim: int = 512
    n_layers: int = 8
    dropout: float = 0.1
    ff_hidden_size: int = 2048
    num_attn_heads: int = 8
    deepthwise_kernel_size: int = 128

    max_seq_len: int = 2048
    rotary_theta: int = 10_000
    window_size: int = -1


class ConvModule(nn.Module):
    def __init__(self, model_dim: int, deepthwise_kernel: int, dropout: float):
        super().__init__()

        self.pointwise_conv_1 = nn.Conv1d(
            in_channels=model_dim, out_channels=model_dim * 2, kernel_size=1, bias=False
        )

        self.norm = nn.LayerNorm(model_dim)

        self.pointwise_1_activation = nn.GLU(dim=1)

        self.deepthwise_conv = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=deepthwise_kernel,
            padding="same",
            groups=model_dim,  # deepthwise is this
            bias=False,
        )

        self.deepthwise_norm = nn.BatchNorm1d(model_dim)

        self.pointwise_conv_2 = nn.Conv1d(
            in_channels=model_dim, out_channels=model_dim, kernel_size=1
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x -> (Batch, Length, Dim)
        """

        y = self.norm(x)
        y = x.transpose(-2, -1)  # (B, D, L)

        y = self.pointwise_conv_1(y)
        y = self.pointwise_1_activation(y)

        y = self.deepthwise_conv(y)
        y = self.deepthwise_norm(y)
        y = nn.functional.silu(y)

        y = self.pointwise_conv_2(y)
        y = self.dropout(y)

        y = y.transpose(-2, -1)  # (B, L, D)
        return x + y


class ConformerBlock(nn.Module):
    def __init__(self, params: ConformerParams):
        super().__init__()

        self.ff1 = FeedFowardBlock(
            embed_dim=params.embed_dim,
            hidden_size=params.ff_hidden_size,
            dropout=params.dropout,
        )

        self.attn = RotaryMultiHeadFlashAttention(
            num_heads=params.num_attn_heads,
            embed_dim=params.embed_dim,
            dropout=params.dropout,
            causal=False,
            window_size=params.window_size,
        )

        self.conv = ConvModule(
            model_dim=params.embed_dim,
            deepthwise_kernel=params.deepthwise_kernel_size,
            dropout=params.dropout,
        )

        self.ff2 = FeedFowardBlock(
            embed_dim=params.embed_dim,
            hidden_size=params.ff_hidden_size,
            dropout=params.dropout,
        )

        self.norm = nn.LayerNorm(params.embed_dim)

    def forward(self, x, rotary_freqs):
        x = x + self.ff1(x) * 0.5
        x = x + self.attn(x, rotary_freqs)
        x = x + self.conv(x)
        x = x + self.ff2(x) * 0.5
        x = self.norm(x)
        return x


class ConformerEncoder(nn.Module):
    def __init__(self, params: ConformerParams):
        super().__init__()

        self.rotary_freqs = precompute_freqs_cis(
            dim=params.embed_dim // params.num_attn_heads,
            end=params.max_seq_len,
            theta=params.rotary_theta,
        )

        self.layers = nn.ModuleList()
        for _ in range(params.n_layers):
            self.layers.append(ConformerBlock(params=params))

    def forward(self, x):
        B, seq_len, embed_dim = x.shape
        rotary_freqs = self.rotary_freqs[:seq_len].to(x.device)

        for i in range(len(self.layers)):
            x = self.layers[i](x, rotary_freqs)

        return x
