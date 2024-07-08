# implementacao do paper https://arxiv.org/pdf/2205.12035

import torch
import torch.nn as nn

from models.base_encoder.model import EncoderMLM
from transformer.transformer_blocks import (
    PositionalEncoding,
    MultiHeadCrossAttention,
    FeedFowardBlock,
)


class RetroMaeDecoder(nn.Module):
    """
    Por conta da forma como eh feita a atencao nele (nao eh causal), vamos
    usar um bloco de encoder
    """

    def __init__(self, decoder_args):
        super().__init__()

        self.masking_ratio = decoder_args["masking_ratio"]

        self.pos_encoding = PositionalEncoding(
            model_dimension=decoder_args["embed_dim"], dropout_probability=0.0
        )

        self.embeddings = nn.Embedding(
            num_embeddings=decoder_args["vocab_size"],
            embedding_dim=decoder_args["embed_dim"],
        )

        self.attention = MultiHeadCrossAttention(
            num_heads=decoder_args["num_heads"],
            embed_dim=decoder_args["embed_dim"],
            dropout=decoder_args["dropout"],
        )

        self.ff = FeedFowardBlock(
            embed_dim=decoder_args["embed_dim"],
            hidden_size=decoder_args["hidden_size"],
            dropout=decoder_args["dropout"],
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=decoder_args["embed_dim"])
        self.norm_2 = nn.LayerNorm(normalized_shape=decoder_args["embed_dim"])

        self.lm_proj = nn.Linear(decoder_args["embed_dim"], decoder_args["vocab_size"])

    def forward(self, enc_emb: torch.Tensor, decoder_input):
        """
        enc_emb - mean embedding from encoder [B, embed_dim]
        decoder_input - raw tokens [B, input_len]
        """
        batch_size, input_len = decoder_input.shape
        # muda o shape para [B, 1, embed_dim] e depois [B, input_len, embed_dim]
        h1 = enc_emb.unsqueeze(1).repeat((1, input_len, 1))
        # adiciona o positional encoding
        h1 = self.pos_encoding(h1)

        h2 = self.embeddings(decoder_input)
        # nao usa a primeira posicao, substituimos pelo embedding do encoder
        h2[:, 0, :] = enc_emb

        # por ultimo a mascara da atencao
        # so nao presta atencao em si mesmo
        att_mask = torch.rand((batch_size, input_len, input_len))
        # precisa ter certeza que nenhum deles vai prestar atencao em si mesmo
        att_mask -= torch.eye(input_len)
        # precisa garantir que todos menos a primeira linha prestem atencao no token 0 (encoder embedding)
        att_mask[:, 1:, 0] = 1.0
        att_mask = torch.masked_fill(
            att_mask, att_mask <= self.masking_ratio, float("-inf")
        )
        att_mask = torch.masked_fill(att_mask, att_mask > float("-inf"), 0)
        # shape [input_len, input_len] -> [1, 1, input_len, input_len]
        att_mask = att_mask.reshape((batch_size, 1, input_len, input_len))
        att_mask = att_mask.to(decoder_input.device)

        # agora roda
        x = self.norm_1(h1 + self.attention(h1, h2, att_mask))
        x = self.norm_2(x + self.ff(x))
        x = self.lm_proj(x)
        return x  # lm logits


class RetroMaeModel(nn.Module):
    def __init__(self, encoder_args, decoder_args):
        super().__init__()

        self.encoder = EncoderMLM(**encoder_args)
        self.decoder = RetroMaeDecoder(decoder_args)

    def forward(self, x_masked, x_mask, x_unmasked):
        encoder_logits, embeddings = self.encoder(
            x_masked, x_mask, return_hidden_states=True
        )
        sentence_embedding = embeddings[:, 0, :]

        decoder_logits = self.decoder(sentence_embedding, x_unmasked)

        return encoder_logits, decoder_logits

    def embed(self, x):
        return self.encoder.embed(x)
