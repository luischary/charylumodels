import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from charylumodels.transformer.attention_blocks import RotaryFlashDecoderBlock
from transformer.utils import count_parameters
from transformer.text_generation import LMPipeline


class MyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.rotary = RotaryEmbedding(embed_dim // num_heads)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_blocks.append(
                RotaryFlashDecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    rotary=self.rotary,
                )
            )

        self.last_norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x):
        x = self.embedding(x)

        for block in self.decoder_blocks:
            x = block(x)

        return self.last_norm(x)


class MyGPTLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.decoder = MyGPT(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.lm_head = nn.Linear(embed_dim, vocab_size)
        count_parameters(self)

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

    def generate_text(
        self,
        input_text,
        tokenizer,
        max_tokens: int = 100,
        sos_token: int = 1,
        eos_token: int = 2,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = None,
        num_beams: int = 1,
        p: float = None,
        repetition_penalty: float = 1.0,
        device: str = "cpu",
        decoder_max_len: int = 512,
    ):
        pipe = LMPipeline(
            tokenizer=tokenizer,
            sos_token=sos_token,
            eos_token=eos_token,
            vocab_size=self.vocab_size,
        )

        if p is not None:
            generated_text = pipe.decoder_nucleus_generation(
                model=self,
                input_text=input_text,
                max_tokens=max_tokens,
                decoder_max_len=decoder_max_len,
                p=p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                device=device,
            )
        else:
            generated_text = pipe.decoder_standard_generation(
                model=self,
                input_text=input_text,
                max_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                num_breams=num_beams,
                repetition_penalty=repetition_penalty,
                device=device,
                decoder_max_len=decoder_max_len,
            )

        return generated_text
