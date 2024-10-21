import torch
import numpy as np


def make_causal_mask(x: torch.Tensor):
    B, seq_len = x.shape
    mask = torch.triu(torch.ones((seq_len, seq_len))) - torch.eye(seq_len)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    mask = mask.reshape((1, 1, seq_len, seq_len)).expand(B, -1, -1, -1)
    mask = mask.to(x.device)

    return mask


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def qkv_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask=None,
    return_attention: bool = False,
):
    """
    Recebe tensores no shape [B, H, L, D] para calculo da atencao
    """
    b, heads, len_tokens, embed_dim = q.shape
    k_t = torch.transpose(k, -1, -2)
    # shapes for q, k, v are [B, HEADS, SEQ_, HEAD_DIM]
    # for K_t we have [B, HEADS, HEAD_DIM, SEQ_K]
    qk = torch.einsum("bhsd, bhde -> bhse", q, k_t)
    # qk = torch.bmm(q, k_t)
    # shape of qk is [B, SEQ_Q, SEQ_K]
    if mask is not None:
        qk = qk + mask
    attention = torch.softmax(qk / np.sqrt(embed_dim), dim=-1)
    # [batch, heads, decoder_len, head_dim] * [batch, heasd, encoder_len, head_dim]
    full_attention = torch.einsum("bhde, bher -> bhdr", attention, v)
    if return_attention:
        return full_attention, attention
    else:
        return full_attention


def block_wise_parallel_attention(
    query, key, value, attention_mask, query_chunk_size, key_chunk_size
):
    """
    query, key, value - [batch, len, heads, head dim]
    """

    q = query / np.sqrt(query.shape[-1])

    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape

    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size

    q = q.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head)).half()
    k = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head)).half()
    v = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head)).half()

    q = q.transpose(1, 0)  # mudando o batch de lugar e colocando o indice antes
    k = k.transpose(1, 0)
    v = v.transpose(1, 0)

    outputs = []

    for query_idx in range(q.shape[0]):
        query_chunk = q[query_idx]

        prev_max_score = torch.tensor(float("-Inf")) * torch.ones(
            batch, query_chunk_size, num_heads, 1
        ).to(q.device).to(q.dtype)
        numerator = (
            torch.zeros((batch, query_chunk_size, num_heads, dim_per_head))
            .to(q.device)
            .to(q.dtype)
        )
        denominator = (
            torch.zeros((batch, query_chunk_size, num_heads, dim_per_head))
            .to(q.device)
            .to(q.dtype)
        )

        for kv_idx in range(k.shape[0]):
            key_chunk = k[kv_idx]
            value_chunk = v[kv_idx]

            attn_weights = torch.einsum("bqhd,bkhd->bqhk", query_chunk, key_chunk)
            # print(attn_weights.shape)
            mask_chunk = attention_mask[
                :,
                :,
                query_idx * query_chunk_size : (query_idx + 1) * query_chunk_size,
                kv_idx * key_chunk_size : (kv_idx + 1) * key_chunk_size,
            ]
            mask_chunk = mask_chunk.transpose(1, 2)

            # precisa checar se a mascara toda nao eh invalida para este bloco
            if mask_chunk.max() < 0:
                continue
            # print(mask_chunk.shape)

            attn_weights = attn_weights + mask_chunk

            max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]
            max_score = torch.maximum(prev_max_score, max_score).detach()

            exp_weights = torch.exp(attn_weights - max_score)
            exp_values = torch.einsum("bqhk,bkhd->bqhd", exp_weights, value_chunk)

            correction = torch.exp(prev_max_score - max_score)
            numerator = numerator * correction + exp_values
            denominator = denominator * correction + exp_weights.sum(
                dim=-1, keepdim=True
            )

            # nao esquece dessa merda
            prev_max_score = max_score

        outputs.append(numerator / denominator)

    outputs = torch.stack(outputs, dim=0)
    outputs = outputs.transpose(0, 1).reshape(batch, q_len, num_heads, dim_per_head)
    return outputs


def cross_batch_attention(
    query,
    key,
    value,
    cross_batch_range,
    max_query_att_lenght=1024,
    max_key_att_length=1024,
):
    batch_size, seq_len, num_heads, _ = query.shape

    num_attention = 1 + cross_batch_range

    cross_batch_rel_ids = torch.arange(0, -num_attention, -1).reshape(1, -1)
    batch_ids = torch.arange(0, batch_size).reshape(-1, 1)
    cross_batch_selector = cross_batch_rel_ids + batch_ids

    # other contexts
    cb_outer_keys = key[cross_batch_selector[:, 1:]]
    # ja abre as keys em comprimento, juntando os cross_batches
    cb_outer_keys = cb_outer_keys.reshape(
        (
            cb_outer_keys.shape[0],
            cb_outer_keys.shape[1] * cb_outer_keys.shape[2],
            cb_outer_keys.shape[3],
            cb_outer_keys.shape[4],
        )
    )
    # local + outer context
    cb_keys = torch.cat([cb_outer_keys, key], dim=1)

    cb_outer_values = value[cross_batch_selector[:, 1:]]
    cb_outer_values = cb_outer_values.reshape(
        (
            cb_outer_values.shape[0],
            cb_outer_values.shape[1] * cb_outer_values.shape[2],
            cb_outer_values.shape[3],
            cb_outer_values.shape[4],
        )
    )
    cb_values = torch.cat([cb_outer_values, value], dim=1)

    # precisa de uma mascara para atencao
    mask = _make_causal_mask(
        input_ids_shape=query.shape[:2],
        dtype=query.dtype,
        device=query.device,
        past_key_values_length=cb_outer_keys.shape[1],
    )

    mask = mask.repeat(1, num_heads, 1, 1)

    attn = block_wise_parallel_attention(
        query=query,
        key=cb_keys,
        value=cb_values,
        attention_mask=mask,
        query_chunk_size=max_query_att_lenght,
        key_chunk_size=max_key_att_length,
    )

    return attn
