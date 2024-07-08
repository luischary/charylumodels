from typing import Any, Dict, List
import os
from copy import deepcopy

import torch
import torch.nn as nn
import lightning.pytorch as pl
import bitsandbytes as bnb
import deepspeed
import numpy as np

from transformer.models.base_decoder.model import DecoderLM
from transformer.schedulers import CosineWarmupScheduler, LinearWarmupScheduler
from transformer.text_generation import GeneratedText
from transformer.utils import TransformerCache


class BaseDecoderLmLT(pl.LightningModule):
    def __init__(
        self,
        decoder_params: Dict,
        learning_rate: float = 1e-4,
        min_lr_percent: float = 0.1,
        warmup_steps: int = 500,
        total_training_steps: int = 1e6,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.min_lr_percent = min_lr_percent
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps

        self.model = DecoderLM(**decoder_params)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # flat the labels
        labels = y.reshape((-1,))
        # runs through the model
        out = self.model(x)
        # flattens the output
        out = out.reshape((-1, self.model.vocab_size))
        loss = torch.nn.functional.cross_entropy(out, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # flat the labels
        labels = y.reshape((-1,))
        # runs through the model
        out = self.model(x)
        # flattens the output
        out = out.reshape((-1, self.model.vocab_size))
        loss = torch.nn.functional.cross_entropy(out, labels)
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=1e-2,
        )
        # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     betas=(0.9, 0.95),
        #     weight_decay=1e-2
        # )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup=self.warmup_steps,
            max_iters=self.total_training_steps,
            min_percent=self.min_lr_percent,
        )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def forward(self, x):
        return self.model(x)

    def get_logits_next_token(self, x):
        with torch.no_grad():
            return self.model(x, next_only=True)

    def get_logits_next_token_cached(self, x, caches):
        with torch.no_grad():
            return self.model.forward_with_cache(x, caches, next_only=True)

    def decoder_nucleus_generation(
        self,
        tokenizer,
        input_text: str,
        sos_token: int,
        eos_token: int,
        end_ia_token: int,
        max_tokens: int,
        vocab_size: int,
        context: str = None,
        decoder_max_len: int = 512,
        p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        device=torch.device("cpu"),
        update_output: bool = False,
        add_sos_token: bool = True,
    ):
        if context is None:
            if add_sos_token:
                tokenized = [sos_token] + tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
            else:
                tokenized = tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
        else:
            if add_sos_token:
                tokenized = (
                    [sos_token]
                    + tokenizer.tokenize_text(context)
                    + tokenizer.tokenize_text(
                        input_text, padding=False, truncation=False
                    )
                )
            else:
                tokenized = tokenizer.tokenize_text(context) + tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
        # print(len(tokenized))
        generated_text = GeneratedText(
            current_tokens=tokenized, vocab_size=vocab_size, device=device
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

            logits = self.get_logits_next_token(decoder_input)
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

            if new_token == eos_token or new_token == end_ia_token:
                break

            if update_output:
                os.system("clear")
                # print(nucleus_probas[:20])
                # print(cumulative_sum[:20])
                print(tokenizer.untokenize_tokens(generated_text.generated_tokens))

        return tokenizer.untokenize_tokens(
            generated_text.tokens
        ), tokenizer.untokenize_tokens(generated_text.generated_tokens)

    def new_decoder_sample(
        self,
        tokenized_input,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = None,
        num_beams: int = 1,
        repetition_penalizer: torch.Tensor = torch.ones(
            1,
        ),
        device=torch.device("cpu"),
    ):
        decoder_input = (
            torch.Tensor(tokenized_input).type(torch.int).unsqueeze(0).to(device)
        )

        logits = self.get_logits_next_token(decoder_input)
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
        tokenizer,
        input_text: str,
        max_tokens: int,
        sos_token: int,
        eos_token: int,
        end_ia_token: int,
        context: str = None,
        decoder_max_len: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = None,
        num_breams: int = 1,
        repetition_penalty: float = 1,
        device=torch.device("cpu"),
        add_sos_token: bool = True,
        vocab_size: int = 50_000,
    ):
        if context is None:
            if add_sos_token:
                tokenized = [sos_token] + tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
            else:
                tokenized = tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
        else:
            if add_sos_token:
                tokenized = (
                    [sos_token]
                    + tokenizer.tokenize_text(context)
                    + tokenizer.tokenize_text(
                        input_text, padding=False, truncation=False
                    )
                )
            else:
                tokenized = tokenizer.tokenize_text(context) + tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
        print(len(tokenized))
        generated_text = GeneratedText(
            current_tokens=tokenized, vocab_size=vocab_size, device=device
        )
        generated_texts = [generated_text]
        for _ in range(max_tokens):
            new_generated_texts = []
            for current_text in generated_texts:
                # se ja tiver chegado no final esse cara nao adiciona mais nada
                if (
                    current_text.tokens[-1] == eos_token
                    or current_text.tokens[-1] == end_ia_token
                ):
                    new_generated_texts.append(deepcopy(current_text))
                    continue

                chosen, logprob = self.new_decoder_sample(
                    tokenized_input=current_text.get_tokens_for_decoder(
                        decoder_max_len=decoder_max_len
                    ),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    num_beams=num_breams,
                    repetition_penalizer=current_text.get_repetition_penalizer(),
                    device=device,
                )

            for new_token, token_logprob in zip(chosen, logprob):
                if new_token != eos_token and new_token != end_ia_token:
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
        return tokenizer.untokenize_tokens(
            generated_text.tokens
        ), tokenizer.untokenize_tokens(generated_text.generated_tokens)

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

    def decoder_nucleus_generation_with_cache(
        self,
        tokenizer,
        input_text: str,
        sos_token: int,
        eos_token: int,
        end_ia_token: int,
        max_tokens: int,
        vocab_size: int,
        caches: List[TransformerCache] = None,
        context: str = None,
        decoder_max_len: int = 512,
        p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        device=torch.device("cpu"),
        update_output: bool = False,
        add_sos_token: bool = True,
    ):
        if caches is None:
            caches = [
                TransformerCache(cache_max_len=decoder_max_len) for _ in range(20)
            ]

        if context is None:
            if add_sos_token:
                tokenized = [sos_token] + tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
            else:
                tokenized = tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
        else:
            if add_sos_token:
                tokenized = (
                    [sos_token]
                    + tokenizer.tokenize_text(context)
                    + tokenizer.tokenize_text(
                        input_text, padding=False, truncation=False
                    )
                )
            else:
                tokenized = tokenizer.tokenize_text(context) + tokenizer.tokenize_text(
                    input_text, padding=False, truncation=False
                )
        # print(len(tokenized))
        generated_text = GeneratedText(
            current_tokens=tokenized, vocab_size=vocab_size, device=device
        )

        for i in range(max_tokens):
            if i == 0:
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
            else:
                decoder_input = (
                    torch.Tensor(generated_text.generated_tokens[-1:])
                    .type(torch.int)
                    .unsqueeze(0)
                    .to(device)
                )

            logits, caches = self.get_logits_next_token_cached(decoder_input, caches)
            last = logits.reshape((-1,))
            probas = nn.functional.softmax(
                last / (temperature * generated_text.get_repetition_penalizer()), dim=0
            )
            # v, top_indices = torch.topk(probas, k=1000, sorted=True, largest=True)
            sorted_probas, sorted_indices = torch.sort(probas, descending=False, dim=-1)
            cumulative_sum = torch.cumsum(sorted_probas, dim=-1)
            # pode acontecer da primeira probabilidade ja ser maior que p
            if cumulative_sum[-1] > p:
                p_aplicado = cumulative_sum[-1]
            else:
                p_aplicado = p

            selected = cumulative_sum >= p_aplicado

            nucleus_probas = sorted_probas.masked_fill(selected == 0, 0)
            nucleus_probas = nucleus_probas / torch.sum(nucleus_probas)
            # nucleus_probas = torch.softmax(nucleus_probas, dim=-1)

            # print(self.tokenizer.untokenize_tokens(nucleus))
            token_chosed = torch.multinomial(nucleus_probas, 1).detach().cpu().item()
            new_token = sorted_indices[token_chosed].detach().cpu().item()
            generated_text.add_new_token(
                new_token, repetition_penalty=repetition_penalty
            )

            if new_token == eos_token or new_token == end_ia_token:
                break

            if update_output:
                os.system("clear")
                # print(nucleus_probas[:20])
                # print(cumulative_sum[:20])
                print(tokenizer.untokenize_tokens(generated_text.generated_tokens))

        caches = None
        del caches
        return tokenizer.untokenize_tokens(
            generated_text.tokens
        ), tokenizer.untokenize_tokens(generated_text.generated_tokens)


class FineTunedDecoderLT(BaseDecoderLmLT):
    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        # flat the labels
        labels = y.reshape((-1,))
        # mask
        masked_ids = torch.flatten(mask.reshape((-1,)).nonzero())
        # runs through the model
        out = self.model(x)
        # flattens the output
        out = out.reshape((-1, self.model.vocab_size))
        loss = torch.nn.functional.cross_entropy(out[masked_ids], labels[masked_ids])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        # flat the labels
        labels = y.reshape((-1,))
        # mask
        masked_ids = torch.flatten(mask.reshape((-1,)).nonzero())
        # runs through the model
        out = self.model(x)
        # flattens the output
        out = out.reshape((-1, self.model.vocab_size))
        loss = torch.nn.functional.cross_entropy(out[masked_ids], labels[masked_ids])
        self.log("validation_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = bnb.optim.AdamW8bit(
        #     self.parameters(),
        #     lr=self.learning_rate,
        #     betas=(0.9, 0.95),
        #     weight_decay=1e-2,
        # )
        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
            model_params=self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=1e-5,
        )
        scheduler = LinearWarmupScheduler(
            optimizer=optimizer,
            warmup=self.warmup_steps,
            max_iters=self.total_training_steps,
            min_percent=self.min_lr_percent,
        )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]
