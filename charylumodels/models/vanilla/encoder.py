import torch
import torch.nn as nn

from transformer.transformer_blocks import EncoderBlock


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


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        max_context_size: int = 1024,
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(
            model_dimension=embed_dim,
            dropout_probability=dropout,
            expected_max_sequence_length=max_context_size,
        )
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
            )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x)

        return x


class EncoderMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        embed_dim: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        max_context_size: int = 1024,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.encoder = Encoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            max_context_size=max_context_size,
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
