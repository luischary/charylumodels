import torch
import torch.nn as nn
from flash_attn import flash_attn_func
import faiss
import numpy as np

from transformer.focused_decoder.model_memory import ModelMemory
from transformer.focused_decoder.attention import cross_batch_attention


class MemoryLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cross_batch_range: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.cross_batch_range = cross_batch_range
        self.head_dim = embed_dim // num_heads

        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)

        self.out_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout_projection = nn.Dropout(p=dropout)

    def reshape_from_attention(self, x):
        B, L, H, HD = x.shape
        # virou [batch, len, heads, head_dim]
        x_view = x.contiguous().view(B, L, self.embed_dim)
        # virou [batch, len, embed_dim]
        return x_view

    def forward(
        self,
        x: torch.Tensor,
        memory=None,
        dump_all: bool = False,
        dump_last: bool = True,
    ):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # reshaping for attention
        B, L, D = x.shape
        # to shape [batch, len, heads, head dim]
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)

        # time for attention
        if self.training:
            # uses cross batch on training
            x_att = cross_batch_attention(
                query=q,
                key=k,
                value=v,
                cross_batch_range=self.cross_batch_range,
                max_query_att_lenght=1024,
                max_key_att_length=1024
            )
        else:
            if memory is not None:
                # uses memory
                x_att, memory = self.forward_with_memory(
                    q, k, v, memory, dump_all, dump_last
                )
            else:
                x_att = flash_attn_func(
                    q.half(),
                    k.half(),
                    v.half(),
                    dropout_p=0.0,
                    causal=True,
                )

        # para inferencia
        if not self.training:
            # se o modelo esta em float32 precisa voltar para float32
            for parametro in self.out_projection.parameters():
                x_att_ajustado = x_att.type(parametro.dtype)
                break
        else:
            x_att_ajustado = x_att

        # faz a concatenacao, volta para o shape [batch, len, embed_dim]
        x_att_reshaped = self.reshape_from_attention(x_att_ajustado)

        # projecao final
        x_att_projected = self.out_projection(self.dropout_projection(x_att_reshaped))

        return x_att_projected, memory

    def forward_with_memory(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        memory: ModelMemory,
        dump_all: bool = False,
        dump_last: bool = True,
    ):
        """
        q, k, v - shape = [batch, heads, len, head_dim]
        memory = ModelMemory
        """
        # primeira coisa eh pegar da memoria os k e v extras
        # a dimensao de head atrapalha um pouco, pra me livrar disso preciso
        # fazer um flat para ficar [-1, head_dim]
        # primeiro guarda o shape original
        # nao adianta fazer nada se nao tiver nada na memoria para pegar
        B, L, H, HD = q.shape
        if memory.key_index.ntotal > 0:
            q_search = (
                q[:, :, :, :].detach().cpu().numpy()
            )
            m_k, m_v = memory.query_top_k(q_search)
            # shape da volta [len * heads * 128, head_dim]
            # transforma em tensor
            m_k = torch.tensor(m_k, dtype=k.dtype).to(q.device)
            m_v = torch.tensor(m_v, dtype=v.dtype).to(q.device)
            # coloca eles no shape que precisamos
            m_k = m_k.transpose(2, 3).reshape(B, -1, H, HD)
            m_v = m_v.transpose(2, 3).reshape(B, -1, H, HD)

            # agora podemos concatenar eles com os valores passados para calcular a atencao de uma vez
            k_full = torch.concat([m_k, k], dim=1)
            v_full = torch.concat([m_v, v], dim=1)
        else:
            v_full = v
            k_full = k

        # calcula as paradas enfim
        x_att = flash_attn_func(
            q.half(),
            k_full.half(),
            v_full.half(),
            dropout_p=0.0,
            causal=True,
        )
        # agora precisa atualizar a memoria
        if dump_all:
            v_search = (
                v.reshape((-1, HD)).detach().cpu().numpy()
            )
            k_search = (
                k.reshape((-1, HD)).detach().cpu().numpy()
            )
            memory.add_key_value(keys=k_search, values=v_search)
        elif dump_last:
            v_search = (
                v[:, 0, :, :]
                .reshape((-1, HD))
                .detach()
                .cpu()
                .numpy()
            )
            k_search = (
                k[:, 0, :, :]
                .reshape((-1, HD))
                .detach()
                .cpu()
                .numpy()
            )
            memory.add_key_value(keys=k_search, values=v_search)

        return x_att, memory
    
    def forward_with_memory_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        memory: ModelMemory,
        dump_all: bool = False,
        dump_last: bool = True,
    ):
        """
        q, k, v - shape = [batch, heads, len, head_dim]
        memory = ModelMemory
        """
        # primeira coisa eh pegar da memoria os k e v extras
        # a dimensao de head atrapalha um pouco, pra me livrar disso preciso
        # fazer um flat para ficar [-1, head_dim]
        # primeiro guarda o shape original
        # nao adianta fazer nada se nao tiver nada na memoria para pegar
        B, L, H, HD = q.shape

        # vai fazer por q e por head
        q_lens = []
        for q_idx in range(L):
            q_heads = []
            for h_idx in range(H):
                q_h = q[:, q_idx, h_idx, :] # [B, HD]
                q_h = q_h.unsqueeze(1).unsqueeze(2) # [B, len, head, hd]
                k_h = k[:, :, h_idx, :] # [B, L, HD]
                v_h = v[:, :, h_idx, :] # [B, L, HD]

                if memory.key_index.ntotal > 0:
                    q_search = q_h.detach().cpu().numpy().reshape(1, 1, 1, -1)
                
                    m_k, m_v = memory.query_top_k(q_search)
                    # shape da volta [top_k, head_dim]
                    # transforma em tensor
                    m_k = torch.tensor(m_k, dtype=k.dtype).to(q.device)
                    m_v = torch.tensor(m_v, dtype=v.dtype).to(q.device)
                    # coloca eles no shape que precisamos
                    m_k = m_k.reshape(B, -1, HD)
                    m_v = m_v.reshape(B, -1, HD)

                    # agora podemos concatenar eles com os valores passados para calcular a atencao de uma vez
                    k_full = torch.concat([m_k, k_h], dim=1).unsqueeze(dim=2) # [batch, len, head, dim]
                    v_full = torch.concat([m_v, v_h], dim=1).unsqueeze(dim=2)

                else:
                    k_full = k_h.unsqueeze(dim=2)
                    v_full = v_h.unsqueeze(dim=2)

                x_att = flash_attn_func(
                    q_h.half(),
                    k_full.half(),
                    v_full.half(),
                    dropout_p=0.0,
                    causal=True,
                )
                q_heads.append(x_att)
            q_heads = torch.concat(q_heads, dim=2)
            q_lens.append(q_heads)

        q_lens = torch.concat(q_lens, dim=1)

        # agora precisa atualizar a memoria
        if dump_all:
            v_search = (
                v.reshape((-1, HD)).detach().cpu().numpy()
            )
            k_search = (
                k.reshape((-1, HD)).detach().cpu().numpy()
            )
            memory.add_key_value(keys=k_search, values=v_search)
        elif dump_last:
            v_search = (
                v[:, 0, :, :]
                .reshape((-1, HD))
                .detach()
                .cpu()
                .numpy()
            )
            k_search = (
                k[:, 0, :, :]
                .reshape((-1, HD))
                .detach()
                .cpu()
                .numpy()
            )
            memory.add_key_value(keys=k_search, values=v_search)

        return q_lens, memory

    def cross_batch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_segmentos: int = 8,
        use_flash: bool = True,
        causal: bool = True,
        apply_dropout: bool = True,
        dropout: float = 0.1,
    ):
        # DEPRECATED

        """
        Funcao feita para o treinamento do 'focused transformer'.
        q, k, v - tensores para calculo da atencao no formato [batch, heads, len, head_embedd]
        num_segmentos - divisao dos documentos individuais para o cross-batch
        use_flash - flag para utilizacao ou nao do flash attention no calculo da atencao. q, k e v precisam ser float16
        causal - flag causal para utilizacao do flash_attention
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        q_segmented = q.reshape(batch_size, num_heads, num_segmentos, -1, head_dim)
        k_segmented = k.reshape(batch_size, num_heads, num_segmentos, -1, head_dim)
        v_segmented = q.reshape(batch_size, num_heads, num_segmentos, -1, head_dim)
        # print(q_segmented.shape)

        # indices do batch anterior
        batch_idx_prev = torch.arange(-1, batch_size - 1)
        # limite de segmentos para os contextos extras
        # divide o total de segmentos por 2 (metade do mesmo documento e metade adversarial)
        max_num_extra_segments = num_segmentos // 2
        # print(batch_idx_prev)
        x_att = None
        for segment_idx in range(num_segmentos):
            q_ = q_segmented[:, :, segment_idx, :, :]
            k_local = k_segmented[:, :, segment_idx, :, :]
            v_local = v_segmented[:, :, segment_idx, :, :]
            # print(q_.shape)
            # print(k_local.shape)
            # print(v_local.shape)

            # pegar mais caras so faz sentido se a quantidade de segmentos for maior que 1
            if num_segmentos > 1:
                # para conseguir limitar os tamanhos
                segment_start = segment_idx // max_num_extra_segments
                segment_end = segment_idx + 1
                # externo
                # "adversarial"
                k_ext_ad = k_segmented[
                    batch_idx_prev - 1, :, segment_start:segment_end, :, :
                ].reshape((batch_size, num_heads, -1, head_dim))
                v_ext_ad = v_segmented[
                    batch_idx_prev - 1, :, segment_start:segment_end, :, :
                ].reshape((batch_size, num_heads, -1, head_dim))
                # print(k_ext_ad.shape)
                # print(v_ext_ad.shape)

                # "do mesmo documento"
                k_ext_doc = k_segmented[:, :, segment_start:segment_end, :, :].reshape(
                    (batch_size, num_heads, -1, head_dim)
                )
                v_ext_doc = v_segmented[:, :, segment_start:segment_end, :, :].reshape(
                    (batch_size, num_heads, -1, head_dim)
                )
                # print(k_ext_doc.shape)
                # print(v_ext_doc.shape)

                # concatena as paradas respeitando a causalidade
                k_full = torch.concat([k_ext_ad, k_ext_doc, k_local], dim=-2)
                v_full = torch.concat([v_ext_ad, v_ext_doc, v_local], dim=-2)
            else:
                k_full = k_local
                v_full = v_local

            # attention
            if use_flash:
                final_att = flash_attn_func(
                    q_.transpose(1, 2),
                    k_full.transpose(1, 2),
                    v_full.transpose(1, 2),
                    dropout_p=dropout if apply_dropout else 0.0,
                    causal=causal,
                )
                final_att = final_att.transpose(2, 1)
            else:
                # TODO incluir mascara causal e colocar como parametro da funcao
                full_att = torch.matmul(q_, k_full.transpose(-2, -1))
                # print(att_doc.shape)

                final_att = torch.matmul(
                    torch.nn.functional.softmax(full_att, dim=-1), v_full
                )
                # print(final_att.shape)

            if x_att is None:
                x_att = final_att
            else:
                x_att = torch.cat([x_att, final_att], dim=-2)

        return x_att