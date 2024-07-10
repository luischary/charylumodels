from typing import Dict, List

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from transformer.hierarchical.encoder import RotaryHierarchicalEncoder
from charylumodels.transformer.attention_blocks import RotaryFlashCrossDecoderBlock
from transformer.text_generation import LMPipeline
from tokenizers.tokenizer import MyTokenizer


class GPTCross(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_size: int,
        dropout: float = 0.1,
        flash_cross_attention: bool = True,
        use_xpos: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.rotary = RotaryEmbedding(embed_dim // num_heads, use_xpos=use_xpos)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_blocks.append(
                RotaryFlashCrossDecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    rotary=self.rotary,
                    use_cross_flash=flash_cross_attention,
                )
            )

        self.last_norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x_decoder, x_encoder):
        x_decoder = self.embedding(x_decoder)

        for block in self.decoder_blocks:
            x_decoder = block(x_decoder, x_encoder)

        return self.last_norm(x_decoder)


class HBART(nn.Module):
    def __init__(self, encoder_params: Dict, decoder_params: Dict):
        super().__init__()

        self.encoder = RotaryHierarchicalEncoder(**encoder_params)
        self.decoder = GPTCross(**decoder_params)

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

    def forward(self, x_encoder, x_decoder, insert_global_token: bool = False):
        # pega so os tokens globais do encoder hierarquico
        x_encoded = self.encode(x_encoder, insert_global_token=insert_global_token)
        if x_encoded.dtype not in [torch.float16, torch.bfloat16]:
            x_encoded = x_encoded.type(torch.float16)
        x = self.decoder(x_decoder, x_encoded)
        return x

    def encode(self, x, insert_global_token: bool = False):
        _, x_encoded = self.encoder(x, insert_global_token=insert_global_token)
        return x_encoded

    def decode(self, x_decoder, x_cross):
        return self.decoder(x_decoder, x_cross)


class HBARTLM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, encoder_params: Dict, decoder_params: Dict
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.model = HBART(encoder_params=encoder_params, decoder_params=decoder_params)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        x_encoder,
        x_decoder,
        non_pad_indexes=None,
        insert_global_token: bool = False,
    ):
        # we dont want to compute loss for padding tokens
        # get all hidden states
        logits = self.lm_head(
            self.model(x_encoder, x_decoder, insert_global_token=insert_global_token)
        )
        # remove batch dimension
        logits = torch.reshape(logits, (-1, self.vocab_size))
        # get only the tokens that matter
        if non_pad_indexes is not None:
            logits = logits[non_pad_indexes, :]

        return logits

    def get_logits_next_token(self, x_decoder, x_cross_attention):
        last_hidden_state = self.model.decode(x_decoder, x_cross_attention)[:, -1, :]
        logits = self.lm_head(last_hidden_state)
        return logits

    def generate_text(
        self,
        cross_attention_text: str,
        decoder_text: str,
        tokenizer: MyTokenizer,
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
        segment_size = self.model.encoder.segment_size
        # faz tokenizacao e padding
        encoder_tokenized = (
            [sos_token] + tokenizer.tokenize_text(cross_attention_text) + [eos_token]
        )

        # padding da parte do encoder
        desired_len = (len(encoder_tokenized) // (segment_size - 1) + 1) * (
            segment_size - 1
        )
        diff = desired_len - len(encoder_tokenized)
        encoder_tokenized = [0 for _ in range(diff)] + encoder_tokenized
        # transforma em tensor
        cross_attention_tokens = (
            torch.tensor(encoder_tokenized).type(torch.int).unsqueeze(0).to(device)
        )
        # pega os tokens globais do encoder para cross attention
        cross_attention_tokens = self.model.encode(
            cross_attention_tokens, insert_global_token=True
        )

        pipe = LMPipeline(
            tokenizer=tokenizer,
            sos_token=sos_token,
            eos_token=eos_token,
            vocab_size=self.vocab_size,
        )

        if p is not None:
            generated_text = pipe.decoder_nucleus_generation(
                model=self,
                input_text=decoder_text,
                max_tokens=max_tokens,
                decoder_max_len=decoder_max_len,
                p=p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                device=device,
                cross_attention_tokens=cross_attention_tokens,
            )
        else:
            generated_text = pipe.decoder_standard_generation(
                model=self,
                input_text=decoder_text,
                max_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                num_breams=num_beams,
                repetition_penalty=repetition_penalty,
                device=device,
                decoder_max_len=decoder_max_len,
                cross_attention_tokens=cross_attention_tokens,
                insert_bos_decoder=True,
            )

        return generated_text
