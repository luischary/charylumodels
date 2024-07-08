import torch
import torch.nn as nn
import numpy as np

from flash_attn import flash_attn_func

from charylumodels.transformer.rotary import apply_rotary_emb, apply_cross_rotary_emb
from charylumodels.transformer.attention import qkv_attention, make_causal_mask
from charylumodels.transformer.transformer_blocks import FeedFowardBlock


class RotaryMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        embed_dim: int = 512,
        dropout=0.1,
    ):
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

    def forward(
        self, x: torch.Tensor, rotary_freqs: torch.Tensor, mask: torch.Tensor = None
    ):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.reshape_for_attention(self.proj_v(x))

        # rotary embeddings
        q, k = apply_rotary_emb(q, k, rotary_freqs)

        q = self.reshape_for_attention(q)
        k = self.reshape_for_attention(k)

        x_att = qkv_attention(q, k, v, mask)
        x_att = self.dropout_attention(x_att)

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att = self.reshape_from_attention(x_att)

        # projecao final
        x_att = self.out_projection(x_att)
        x_att = self.dropout_projection(x_att)
        return x_att


class RotaryMultiHeadFlashAttention(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        embed_dim: int = 512,
        dropout=0.1,
        causal: bool = False,
        window_size: int = -1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.causal = causal
        self.window_size = window_size  # new feature from flash attention

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
        x_view = x.contiguous().view((B, L, self.num_heads, self.head_dim))
        # virou [batch, len, heads, head_dim]
        return x_view

    def reshape_from_attention(self, x):
        B, L, H, HD = x.shape
        # virou [batch, len, heads, head_dim]
        x_view = x.contiguous().view(B, L, self.embed_dim)
        # virou [batch, len, embed_dim]
        return x_view

    def forward(
        self,
        x: torch.Tensor,
        rotary_freqs: torch.Tensor,
    ):
        q = self.reshape_for_attention(self.proj_q(x.clone()))
        k = self.reshape_for_attention(self.proj_k(x.clone()))
        v = self.reshape_for_attention(self.proj_v(x.clone()))

        # rotary embeddings
        q, k = apply_rotary_emb(q, k, rotary_freqs)

        if next(self.parameters()).is_cuda:
            x_att = flash_attn_func(
                q=q.half(),  # flash requer float16
                k=k.half(),
                v=v.half(),
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=1 / np.sqrt(self.head_dim),
                causal=self.causal,
                return_attn_probs=False,
                window_size=(self.window_size, self.window_size),
            )
        else:
            x_att = qkv_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                mask=make_causal_mask(q) if self.causal else None,
            ).transpose(1, 2)
        # para inferencia
        if not self.training or q.dtype != x_att.dtype:
            # se o modelo esta em float32 precisa voltar para float32
            for parametro in self.out_projection.parameters():
                x_att_ajustado = x_att.type(parametro.dtype)
                break
        else:
            x_att_ajustado = x_att

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att_reshaped = self.reshape_from_attention(x_att_ajustado)

        # projecao final
        x_att_projected = self.out_projection(self.dropout_projection(x_att_reshaped))

        return x_att_projected


class RotaryMHFlashCrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        embed_dim: int = 512,
        dropout=0.1,
        causal: bool = False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.causal = causal

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
        x_view = x.contiguous().view((B, L, self.num_heads, self.head_dim))
        # virou [batch, len, heads, head_dim]
        return x_view

    def reshape_from_attention(self, x):
        B, L, H, HD = x.shape
        # virou [batch, len, heads, head_dim]
        x_view = x.contiguous().view(B, L, self.embed_dim)
        # virou [batch, len, embed_dim]
        return x_view

    def forward(
        self,
        x: torch.Tensor,
        rotary_freqs: torch.Tensor,
    ):
        q = self.reshape_for_attention(self.proj_q(x.clone()))
        k = self.reshape_for_attention(self.proj_k(x.clone()))
        v = self.reshape_for_attention(self.proj_v(x.clone()))

        # rotary embeddings
        q, k = apply_cross_rotary_emb(q, k, rotary_freqs)

        x_att = flash_attn_func(
            q=q.half(),  # flash requer float16
            k=k.half(),
            v=v.half(),
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=1 / np.sqrt(self.head_dim),
            causal=self.causal,
            return_attn_probs=False,
            window_size=(self.window_size, self.window_size),
        )
        # para inferencia
        if not self.training or q.dtype != x_att.dtype:
            # se o modelo esta em float32 precisa voltar para float32
            for parametro in self.out_projection.parameters():
                x_att_ajustado = x_att.type(parametro.dtype)
                break
        else:
            x_att_ajustado = x_att

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att_reshaped = self.reshape_from_attention(x_att_ajustado)

        # projecao final
        x_att_projected = self.out_projection(self.dropout_projection(x_att_reshaped))

        return x_att_projected


class RotaryFlashDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_size, rotary, dropout: int = 0.1):
        super().__init__()

        self.attention = RotaryMultiHeadFlashAttention(
            rotary=rotary,
            num_heads=num_heads,
            embed_dim=embed_dim,
            dropout=dropout,
            causal=True,
        )
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.drop_skip_1 = nn.Dropout(dropout)
        self.drop_skip_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop_skip_1(x) + self.attention(self.norm_1(x))
        x = self.drop_skip_2(x) + self.feedforward(self.norm_2(x))
        return x
