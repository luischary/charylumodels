from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np


class GeneratedText:
    def __init__(
        self,
        current_tokens: List[int],
        initial_logprob: float = 0,
        vocab_size: int = 60000,
        device=torch.device("cpu"),
    ):
        self.tokens = current_tokens
        self.generated_tokens = []
        self.logprob = initial_logprob
        self.vocab_size = vocab_size
        self.repetition_penalizer = torch.ones((vocab_size,)).to(device)

    def add_new_token(self, token, logprob=0, repetition_penalty: float = 1.0):
        # if token != self.tokens[-1]:
        self.tokens.append(token)
        self.generated_tokens.append(token)
        self.logprob += logprob
        self.repetition_penalizer[token] += repetition_penalty

    def __len__(self):
        return len(self.tokens)

    def get_tokens_for_decoder(self, decoder_max_len):
        if len(self.tokens) > decoder_max_len:
            # half = decoder_max_len // 2
            # return self.tokens[:half] + self.tokens[-half:]
            return self.tokens[-decoder_max_len:]
        else:
            return self.tokens

    def get_repetition_penalizer(self):
        return self.repetition_penalizer


class LMPipeline:
    def __init__(
        self,
        tokenizer,
        sos_token: int = 0,
        eos_token: int = 1,
        vocab_size: int = 60000,
    ):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    @torch.no_grad()
    def new_decoder_sample(
        self,
        model: torch.nn.Module,
        tokenized_input,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = None,
        num_beams: int = 1,
        repetition_penalizer: torch.Tensor = torch.ones(
            1,
        ),
        device=torch.device("cpu"),
        cross_attention_tokens: torch.Tensor = None,
    ):
        decoder_input = (
            torch.Tensor(tokenized_input).type(torch.int).unsqueeze(0).to(device)
        )
        if cross_attention_tokens is not None:
            logits = model.get_logits_next_token(decoder_input, cross_attention_tokens)
        else:
            logits = model.get_logits_next_token(decoder_input)
        last = logits.reshape((-1,))
        last = last / (temperature * repetition_penalizer)
        if num_beams > 1:
            v, beam_top_indices = torch.topk(last, k=top_k, sorted=False, largest=True)
            last[last < v.min()] = -float("Inf")
        elif top_k is not None:
            v, _ = torch.topk(last, k=top_k, sorted=False, largest=True)
            last[last < v.min()] = -float("Inf")

        probas = nn.functional.softmax(last, dim=0)

        chosen = []
        logprob = []
        for i in range(num_beams):
            if do_sample:
                token_chosed = torch.multinomial(probas, 1).detach().cpu().item()
            else:
                token_chosed = torch.argmax(probas).detach().cpu().item()

            logprob_token = 1 - torch.log(probas[token_chosed]).detach().cpu().item()
            chosen.append(token_chosed)
            logprob.append(logprob_token)

        return chosen, logprob

    def decoder_standard_generation(
        self,
        model: torch.nn.Module,
        input_text: str,
        max_tokens: int,
        decoder_max_len: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = None,
        num_breams: int = 1,
        repetition_penalty: float = 1,
        device=torch.device("cpu"),
        cross_attention_tokens: torch.Tensor = None,
        insert_bos_decoder: bool = True,
    ):
        tokenized = self.tokenizer.tokenize_text(
            input_text, padding=False, truncation=False
        )
        if insert_bos_decoder:
            tokenized = [self.sos_token] + tokenized

        generated_text = GeneratedText(
            current_tokens=tokenized, vocab_size=self.vocab_size, device=device
        )
        generated_texts = [generated_text]
        for _ in range(max_tokens):
            new_generated_texts = []
            for current_text in generated_texts:
                # se ja tiver chegado no final esse cara nao adiciona mais nada
                if current_text.tokens[-1] == self.eos_token:
                    new_generated_texts.append(deepcopy(current_text))
                    continue

                chosen, logprob = self.new_decoder_sample(
                    model=model,
                    tokenized_input=current_text.get_tokens_for_decoder(
                        decoder_max_len=decoder_max_len
                    ),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    num_beams=num_breams,
                    repetition_penalizer=current_text.get_repetition_penalizer(),
                    device=device,
                    cross_attention_tokens=cross_attention_tokens,
                )

            for new_token, token_logprob in zip(chosen, logprob):
                if new_token != self.eos_token:
                    new_text = deepcopy(current_text)
                    new_text.add_new_token(
                        new_token, token_logprob, repetition_penalty=repetition_penalty
                    )
                    new_generated_texts.append(new_text)
                else:
                    new_generated_texts.append(current_text)

            generated_texts = self.filter_beam_generated(
                new_generated_texts, num_beams=num_breams**3, normalize_prob=True
            )

        generated_text = self.filter_beam_generated(
            generated_texts, num_beams=1, normalize_prob=True
        )[0]
        return self.tokenizer.untokenize_tokens(generated_text.tokens)

    def filter_beam_generated(
        self, texts: List[GeneratedText], num_beams: int, normalize_prob: bool = True
    ):
        if len(texts) <= num_beams:
            return texts

        probas = [t.logprob for t in texts]
        if normalize_prob:
            probas = [p / (len(t.tokens) ** 0.7) for p, t in zip(probas, texts)]

        filtered = []
        for i in range(num_beams):
            biggest = np.argmax(probas)
            filtered.append(texts[biggest])
            probas.pop(biggest)
            texts.pop(biggest)

        return filtered

    @torch.no_grad()
    def decoder_nucleus_generation(
        self,
        model: torch.nn.Module,
        input_text: str,
        max_tokens: int,
        decoder_max_len: int = 512,
        p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        device=torch.device("cpu"),
        cross_attention_tokens: torch.Tensor = None,
    ):
        tokenized = [self.sos_token] + self.tokenizer.tokenize_text(
            input_text, padding=False, truncation=False
        )
        generated_text = GeneratedText(
            current_tokens=tokenized, vocab_size=self.vocab_size, device=device
        )

        for i in range(max_tokens):
            decoder_input = (
                torch.Tensor(
                    generated_text.get_tokens_for_decoder(
                        decoder_max_len=decoder_max_len
                    )
                )
                .type(torch.int)
                .unsqueeze(0)
                .to(device)
            )
            if cross_attention_tokens is not None:
                logits = model.get_logits_next_token(
                    decoder_input, cross_attention_tokens
                )
            else:
                logits = model.get_logits_next_token(decoder_input)
            last = logits.reshape((-1,))
            probas = nn.functional.softmax(
                last / (temperature * generated_text.get_repetition_penalizer()), dim=0
            )
            # v, top_indices = torch.topk(probas, k=1000, sorted=True, largest=True)
            sorted_probas, sorted_indices = torch.sort(probas, descending=True, dim=-1)
            cumulative_sum = torch.cumsum(sorted_probas, dim=-1)
            # pode acontecer da primeira probabilidade ja ser maior que p
            if cumulative_sum[0] > p:
                p_aplicado = 1
            else:
                p_aplicado = p

            selected = cumulative_sum <= p_aplicado

            nucleus_probas = sorted_probas.masked_fill(selected == 0, 0)
            nucleus_probas = nucleus_probas / torch.sum(nucleus_probas)
            # nucleus_probas = torch.softmax(nucleus_probas, dim=-1)

            # print(self.tokenizer.untokenize_tokens(nucleus))
            token_chosed = torch.multinomial(nucleus_probas, 1).detach().cpu().item()
            new_token = sorted_indices[token_chosed].detach().cpu().item()
            generated_text.add_new_token(
                new_token, repetition_penalty=repetition_penalty
            )

            if new_token == self.eos_token:
                break

        return self.tokenizer.untokenize_tokens(generated_text.tokens)
