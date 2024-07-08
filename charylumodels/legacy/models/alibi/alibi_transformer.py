import math

import torch
import torch.nn as nn
import numpy as np

from transformer.transformer_blocks import DecoderBlock
from transformer.utils import count_parameters
from transformer.text_generation import LMPipeline


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


def get_exponential_slopes(n):
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    slopes = []
    for i in range(n):
        usavel = i
        while usavel >= len(sizes):
            usavel -= len(sizes)
        slopes.append(math.log(0.1) / sizes[usavel])
    return slopes


class AlibiDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        hidden_size,
        dropout: int = 0.1,
        tokens_per_sample: int = 1024,
        limit_alibi_window: int = None,
        encoding_type: str = "exponential",
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.tokens_per_sample = tokens_per_sample

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

        # alibi related
        if encoding_type == "exponential":
            self.alibi_mask = self.create_alibi_exponential()
        else:
            self.alibi_mask = self.create_alibi(limit_alibi_window)

        self._future_mask = torch.empty(0)

    def forward(self, x):
        causal_mask = self.create_future_mask(x)
        x = self.embedding(x)

        for block in self.decoder_blocks:
            x = block(x, causal_mask)

        return self.last_norm(x)

    def create_future_mask(self, tensor):
        dim = tensor.shape[1]
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(1) < self.tokens_per_sample
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(
                    torch.zeros([self.tokens_per_sample, self.tokens_per_sample])
                ),
                1,
            )
            self._future_mask = self._future_mask.unsqueeze(0) + self.alibi_mask
        self._future_mask = self._future_mask.to(tensor.device)
        # return self._future_mask[: tensor.shape[0] * self.num_heads, :dim, :dim]
        return self._future_mask[:, :dim, :dim]

    def create_alibi_exponential(self):
        slopes = torch.Tensor(get_exponential_slopes(self.num_heads))
        context_position = torch.arange(self.tokens_per_sample, dtype=torch.long)[
            :, None
        ]
        memory_position = torch.arange(self.tokens_per_sample, dtype=torch.long)[
            None, :
        ]
        relative_position = memory_position - context_position
        relative_position = relative_position.abs() * -1

        alibi_mask = relative_position.unsqueeze(0).expand(self.num_heads, -1, -1)
        slopes = slopes.unsqueeze(1).unsqueeze(1)
        alibi_mask = torch.exp(alibi_mask * torch.abs(slopes))
        return alibi_mask

    def create_alibi(self, limit_alibi_window):
        slopes = torch.Tensor(get_slopes(self.num_heads))

        # normal alibi
        context_position = torch.arange(self.tokens_per_sample, dtype=torch.long)[
            :, None
        ]
        memory_position = torch.arange(self.tokens_per_sample, dtype=torch.long)[
            None, :
        ]
        relative_position = memory_position - context_position
        relative_position = relative_position.abs() * -1
        if limit_alibi_window is not None:
            limits = torch.ones_like(relative_position) * limit_alibi_window
            relative_position = torch.max(relative_position, limits)

        alibi_mask = relative_position.unsqueeze(0).expand(
            self.num_heads, -1, -1
        ) * slopes.unsqueeze(1).unsqueeze(1)
        return alibi_mask


class AlibiDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        hidden_size,
        dropout=0.1,
        pad_token_id: int = 0,
        tokens_per_sample: int = 1024,
        limit_alibi_window: int = None,
        encoding_type: str = "exponential",
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.decoder = AlibiDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout=dropout,
            tokens_per_sample=tokens_per_sample,
            limit_alibi_window=limit_alibi_window,
            encoding_type=encoding_type,
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
                decoder_max_len=self.decoder.tokens_per_sample,
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
            )

        return generated_text
