from typing import List
import math

import torch
import torch.nn as nn
import numpy as np
from rotary_embedding_torch import RotaryEmbedding

from transformer.transformer_blocks import (
    FeedFowardBlock,
    PositionalEncoding,
)
from charylumodels.transformer.attention_blocks import RotaryMultiHeadFlashAttention


class SegmentEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_size,
        segment_size,
        rotary,
        dropout: int = 0.1,
    ):
        super().__init__()
        self.segment_size = segment_size
        self.attention = RotaryMultiHeadFlashAttention(
            num_heads=num_heads, embed_dim=embed_dim, dropout=dropout, rotary=rotary
        )
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.drop_skip_1 = nn.Dropout(dropout)
        self.drop_skip_2 = nn.Dropout(dropout)

    def forward(self, x):
        # muda de [batch, len, dim] para [batch * segments, segment, dim]
        B, L, D = x.size()
        x_reshaped = x.contiguous().view(-1, self.segment_size, D)

        x_1 = self.norm_1(self.drop_skip_1(x_reshaped) + self.attention(x_reshaped))
        x_2 = self.norm_2(self.drop_skip_2(x_1) + self.feedforward(x_1))

        # volta para o shape original
        x_final = x_2.contiguous().view(B, L, D)
        return x_final

    def reshape_for_segmented_attention(self, x: torch.Tensor):
        # muda de [batch, len, dim] para [batch * segments, segment, dim]
        B, L, D = x.size()
        return x.contiguous().view(-1, self.segment_size, D)


class GlobalEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        hidden_size,
        segment_size,
        rotary,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.segment_size = segment_size
        self.attention = RotaryMultiHeadFlashAttention(
            num_heads=num_heads, embed_dim=embed_dim, dropout=dropout, rotary=rotary
        )
        self.feedforward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.drop_skip_1 = nn.Dropout(dropout)
        self.drop_skip_2 = nn.Dropout(dropout)

    def forward(self, x):
        globals = self.extract_globals(x)
        globals_1 = self.norm_1(self.drop_skip_1(globals) + self.attention(globals))
        globals_2 = self.norm_2(
            self.drop_skip_2(globals_1) + self.feedforward(globals_1)
        )

        # substitui os globais na matriz original
        novo_x = self.update_globals(globals_2, x)
        return novo_x

    def extract_globals(self, x: torch.Tensor):
        """
        Devolve apenas os tokens globais
        """
        B, L, D = x.size()
        globals = x.contiguous().view(B, -1, self.segment_size, D)[:, :, 0, :]
        return globals

    def update_globals(self, new_globals: torch.Tensor, x: torch.Tensor):
        B, L, D = x.size()
        x.contiguous().view(B, -1, self.segment_size, D)[:, :, 0, :] = new_globals
        # TODO verificar se precisa fazer reshape no x ou nao
        return x


class HierarchicalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        max_context_size: int = 1024,
        segment_size: int = 128,
        global_token_id: int = 4,
        layers_type: List[str] = [
            "segment",
            "segment",
            "global",
            "segment",
            "segment",
            "global",
        ],
    ):
        super().__init__()

        self.segment_size = segment_size
        self.global_token_id = global_token_id

        self.pos_encoding = PositionalEncoding(
            model_dimension=embed_dim,
            dropout_probability=dropout,
            expected_max_sequence_length=max_context_size,
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        self.layers = nn.ModuleList()
        for layer_type in layers_type:
            if layer_type == "global":
                self.layers.append(
                    GlobalEncoderBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        segment_size=segment_size,
                        dropout=dropout,
                    )
                )
            else:
                self.layers.append(
                    SegmentEncoderBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        segment_size=segment_size,
                        dropout=dropout,
                    )
                )

    def forward(self, x, insert_global_token: bool = False):
        if insert_global_token:
            x = self.insert_global_tokens(x)

        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        # retorna os embeddings normais e globais separadamente
        return self.separate_globals(x)

    def insert_global_tokens(self, x: torch.Tensor):
        B, L = x.size()
        num_pre_segments = L // (self.segment_size - 1)
        x_segmented = x.reshape((B, num_pre_segments, -1))
        to_insert = (
            (torch.ones((B, num_pre_segments, 1)) * self.global_token_id)
            .long()
            .to(x.device)
        )
        x_inserted = torch.cat([to_insert, x_segmented], dim=-1).reshape((B, -1))
        return x_inserted

    def separate_globals(self, x: torch.Tensor):
        B, L, D = x.size()
        globals = x.contiguous().view(B, -1, self.segment_size, D)[:, :, 0, :]
        normals = x.contiguous().view(B, -1, self.segment_size, D)[:, :, 1:, :]
        normals = normals.reshape(B, -1, D)
        return normals, globals


class HierarchicalEncoderMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        max_context_size: int = 1024,
        segment_size: int = 128,
        global_token_id: int = 4,
        layers_type: List[str] = [
            "segment",
            "segment",
            "global",
            "segment",
            "segment",
            "global",
        ],
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.encoder = HierarchicalEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            max_context_size=max_context_size,
            segment_size=segment_size,
            global_token_id=global_token_id,
            layers_type=layers_type,
        )

        self.mlm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask, insert_global_token: bool = False):
        # first get the masked_ids to use later
        # flattens the masked id so its easier to deal with
        masked_ids = torch.flatten(mask.reshape((-1,)).nonzero())
        # get all hidden states
        last_hidden_states, _ = self.encoder(x, insert_global_token=insert_global_token)
        # flatten everything so we can compute everything at once
        all_hidden_states = last_hidden_states.reshape(-1, self.embed_dim)
        # get only the masked hidden states
        masked_hidden_states = all_hidden_states[masked_ids, :]
        # predicts only the masked tokens
        logits = self.mlm_head(masked_hidden_states)

        return logits


# versao Rotary positional encoding
class RotaryHierarchicalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        segment_size: int = 128,
        global_token_id: int = 4,
        layers_type: List[str] = [
            "segment",
            "segment",
            "global",
            "segment",
            "segment",
            "global",
        ],
    ):
        super().__init__()

        self.segment_size = segment_size
        self.global_token_id = global_token_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        self.rotary = RotaryEmbedding(embed_dim // num_heads)

        layers = nn.ModuleList()
        for layer_type in layers_type:
            if layer_type == "global":
                layers.append(
                    GlobalEncoderBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        segment_size=segment_size,
                        dropout=dropout,
                        rotary=self.rotary,
                    )
                )
            else:
                layers.append(
                    SegmentEncoderBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        segment_size=segment_size,
                        dropout=dropout,
                        rotary=self.rotary,
                    )
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, x, insert_global_token: bool = False):
        if insert_global_token:
            x_inserted = self.insert_global_tokens(x)
        else:
            x_inserted = x

        x_embedded = self.embedding(x_inserted)

        hidden_states = self.layers(x_embedded)

        # retorna os embeddings normais e globais separadamente
        return self.separate_globals(hidden_states)

    def insert_global_tokens(self, x: torch.Tensor):
        B, L = x.size()
        num_pre_segments = L // (self.segment_size - 1)
        x_segmented = x.reshape((B, num_pre_segments, -1))
        to_insert = (
            (torch.ones((B, num_pre_segments, 1)) * self.global_token_id)
            .int()
            .to(x.device)
        )
        x_inserted = torch.cat([to_insert, x_segmented], dim=-1).reshape((B, -1))
        return x_inserted

    def separate_globals(self, x: torch.Tensor):
        B, L, D = x.size()
        globals = x.contiguous().view(B, -1, self.segment_size, D)[:, :, 0, :]
        normals = x.contiguous().view(B, -1, self.segment_size, D)[:, :, 1:, :]
        normals_reshaped = normals.reshape(B, -1, D)
        return normals_reshaped, globals


class RotaryHierarchicalEncoderMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        segment_size: int = 128,
        global_token_id: int = 4,
        layers_type: List[str] = [
            "segment",
            "segment",
            "global",
            "segment",
            "segment",
            "global",
        ],
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.encoder = RotaryHierarchicalEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            segment_size=segment_size,
            global_token_id=global_token_id,
            layers_type=layers_type,
        )

        self.mlm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask, insert_global_token: bool = False):
        # first get the masked_ids to use later
        # flattens the masked id so its easier to deal with
        masked_ids = torch.flatten(mask.reshape((-1,)).nonzero())
        # get all hidden states
        last_hidden_states, _ = self.encoder(x, insert_global_token=insert_global_token)
        # flatten everything so we can compute everything at once
        all_hidden_states = last_hidden_states.reshape(-1, self.embed_dim)
        # get only the masked hidden states
        masked_hidden_states = all_hidden_states[masked_ids, :]
        # predicts only the masked tokens
        logits = self.mlm_head(masked_hidden_states)

        return logits
