import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_exponential_slopes(n):
    sizes = [64, 128, 255, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    slopes = []
    for i in range(n):
        usavel = i
        while usavel >= len(sizes):
            usavel -= len(sizes)
        slopes.append(math.log(0.1) / sizes[usavel])
    return slopes


def _chunk(x, w):
    """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""

    # non-overlapping chunks of size = 2w
    x = x.view(x.size(0), x.size(1), x.size(2) // (w * 2), w * 2, x.size(3))

    # use `as_strided` to make the chunks overlap with an overlap size = w
    chunk_size = list(x.size())
    chunk_size[2] = chunk_size[2] * 2 - 1

    chunk_stride = list(x.stride())
    chunk_stride[2] = chunk_stride[2] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)


def _skew(x, direction, padding_value):
    """Convert diagonals into columns (or columns into diagonals depending on `direction`"""
    x_padded = F.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(
        *x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2)
    )
    return x_padded


def _skew2(x, padding_value):
    """shift every row 1 step to right converting columns into diagonals"""
    # X = B x C x M x L
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
    x = x.view(B, C, -1)  # B x C x ML+MM+M
    x = x[:, :, :-M]  # B x C x ML+MM
    x = x.view(B, C, M, M + L)  # B x C, M x L+M
    x = x[:, :, :, :-1]
    return x


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


class SlidingCuhnksMultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int = 8, embed_dim: int = 512, w: int = 128, dropout=0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.w = w

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

    def sliding_chunks_matmul_qk(
        self, q: torch.Tensor, k: torch.Tensor, padding_value: int = 0
    ):
        """
        q - (batch, heads, seqlen, head_dim)
        k - (batch, heads, seqlen, head_dim)
        retorna a matriz de atencoes janelada com pading w nas duas pontas
        (batch, heads, seqlen, 2 * w + 1)

        """
        bsz, num_heads, seqlen, head_dim = q.size()
        chunks_count = seqlen // self.w - 1
        assert seqlen % (self.w * 2) == 0, "The seqlen must be multiple of the window"

        q_chunk = _chunk(q, self.w)
        k_chunk = _chunk(k, self.w)
        chunk_att = torch.einsum("bhcxd,bhcyd->bhcxy", (q_chunk, k_chunk))

        diagonal_chunk_att = _skew(
            chunk_att, direction=(0, 0, 0, 1), padding_value=padding_value
        )
        diagonal_attn = diagonal_chunk_att.new_empty(
            (bsz, num_heads, chunks_count + 1, self.w, self.w * 2 + 1)
        )

        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :, :-1, :, self.w :] = diagonal_chunk_att[
            :, :, :, : self.w, : self.w + 1
        ]
        diagonal_attn[:, :, -1, :, self.w :] = diagonal_chunk_att[
            :, :, -1, self.w :, : self.w + 1
        ]
        # - copying the lower triangle
        diagonal_attn[:, :, 1:, :, : self.w] = diagonal_chunk_att[
            :, :, :, -(self.w + 1) : -1, self.w + 1 :
        ]
        diagonal_attn[:, :, 0, 1 : self.w, 1 : self.w] = diagonal_chunk_att[
            :, :, 0, : self.w - 1, 1 - self.w :
        ]

        diagonal_attn = diagonal_attn.reshape(bsz, num_heads, seqlen, 2 * self.w + 1)
        return diagonal_attn

    def sliding_chunks_matmul_pv(
        self, prob: torch.Tensor, v: torch.Tensor, padding_value: int = 0
    ):
        """
        faz a multiplicacao da probabilidade pelo value, ambos janelados.
        prob - (batch * num_heads, seqlen // w, w, 2 * w + 1)
        v - (batch, seqlen, num_heads, head_dim)
        """
        bsz, seqlen, num_heads, head_dim = v.size()
        v = v.transpose(2, 1).reshape((bsz * num_heads, seqlen, head_dim))

        # comeca colocando um padding no value. W no comeco e W no final
        padded_v = F.pad(v, (0, 0, self.w, self.w), value=-1)

        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunks_count = seqlen // self.w - 1
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * self.w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = (
            chunk_v_stride[0],
            self.w * chunk_v_stride[1],
            chunk_v_stride[1],
            chunk_v_stride[2],
        )
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

        chunk_prob = prob.reshape(
            bsz * num_heads, seqlen // self.w, self.w, 2 * self.w + 1
        )
        skewed_prob = _skew2(chunk_prob, padding_value=padding_value)

        context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))
        return context.view((bsz, num_heads, seqlen, head_dim))

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

    def slidingChunksQKVattention(self, q, k, v, mask=None):
        b, heads, len_tokens, embed_dim = q.shape
        qk = self.sliding_chunks_matmul_qk(q, k, padding_value=0)
        # qk tem valores de padding no comeco e no final que precisam virar -inf
        # mascara atencoes invalidas
        for i in range(self.w):
            # tira da esquerda
            qk[:, :, i, : self.w - i] = float("-inf")
            # tira da direita
            qk[:, :, -(i + 1), -(self.w - i) :] = float("-inf")

        attention = torch.softmax(qk / np.sqrt(embed_dim), dim=-1)
        attention = self.dropout_attention(attention)

        # [batch, heads, seqlen, 2 * w + 1]

        full_attention = self.sliding_chunks_matmul_pv(
            prob=attention, v=v, padding_value=0
        )
        # [batch, heads, seqlen, head_dim]

        return self.dropout_projection(full_attention)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        q = self.reshape_for_attention(self.proj_q(x))
        k = self.reshape_for_attention(self.proj_k(x))
        v = self.proj_v(x)
        B, L, E = v.shape
        v = v.reshape(
            (B, L, self.num_heads, self.head_dim)
        )  # v fica com shape diferente mesmo

        x_att = self.slidingChunksQKVattention(q, k, v, mask)

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


class LongformerBlock(nn.Module):
    def __init__(self, num_heads, embed_dim, hidden_size, window_size, dropout):
        super().__init__()

        self.attn_layer = SlidingCuhnksMultiHeadAttention(
            num_heads=num_heads, embed_dim=embed_dim, w=window_size, dropout=dropout
        )

        self.feed_forward = FeedFowardBlock(
            embed_dim=embed_dim, hidden_size=hidden_size, dropout=dropout
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.skip_dropout_1 = nn.Dropout(dropout)
        self.skip_dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.skip_dropout_1(x) + self.attn_layer(self.norm_1(x), mask)
        x = self.skip_dropout_2(x) + self.feed_forward(self.norm_2(x))
        return x


class Longformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_heads,
        num_layers,
        window_size,
        max_context_size,
        dropout,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.max_context_size = max_context_size

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                LongformerBlock(
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    hidden_size=hidden_size,
                    window_size=window_size,
                    dropout=dropout,
                )
            )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        # ja cria a matriz do positional encoding
        context_position = torch.arange(
            end=self.window_size + 1, dtype=torch.long, start=-self.window_size
        )[None, :]
        memory_position = torch.arange(self.max_context_size, dtype=torch.long)[:, None]
        relative_position = memory_position + context_position
        relative_position = relative_position.abs() * -1

        pos_encoding = relative_position.unsqueeze(0).expand(num_heads, -1, -1)
        slopes = torch.Tensor(get_exponential_slopes(num_heads))
        slopes = slopes.unsqueeze(1).unsqueeze(1)
        self.pos_encoding_mask = torch.exp(pos_encoding * torch.abs(slopes))

    def forward(self, x):
        x = self.embedding(x)
        attention_mask = self.create_attention_mask(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x

    def create_attention_mask(self, x: torch.Tensor):
        len = x.shape[1]

        return self.pos_encoding_mask[:, :len, :].to(x.device)


class LongformerMLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_heads,
        num_layers,
        window_size,
        max_context_size,
        dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.encoder = Longformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            max_context_size=max_context_size,
            dropout=dropout,
        )

        self.mlm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask):
        # first get the masked_ids to use later
        # flattens the masked id so its easier to deal with
        masked_ids = torch.flatten(mask.reshape((-1,)).nonzero())
        # get all hidden states
        last_hidden_states = self.encoder(x)
        # flatten everything so we can compute everything at once
        all_hidden_states = last_hidden_states.reshape(-1, self.embed_dim)
        # get only the masked hidden states
        masked_hidden_states = all_hidden_states[masked_ids, :]
        # predicts only the masked tokens
        logits = self.mlm_head(masked_hidden_states)

        return logits


if __name__ == "__main__":
    # teste
    device = "cuda"
    modelo = Longformer(
        vocab_size=100,
        embed_dim=16,
        hidden_size=16 * 4,
        num_heads=2,
        num_layers=2,
        window_size=3,
        max_context_size=1024,
        dropout=0.1,
        pad_token_id=0,
    )
    modelo.to(device)

    x_teste = np.random.randint(1, 100, size=(10))
    x_teste = np.concatenate([x_teste, [0, 0]])
    x_teste = torch.from_numpy(x_teste).unsqueeze(0).to(device)
    with torch.no_grad():
        resposta = modelo(x_teste)
    print(resposta)
    print(resposta.shape)
