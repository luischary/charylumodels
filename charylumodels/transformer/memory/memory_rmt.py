import torch.nn as nn
import torch
from torch import cat
from torch.nn.functional import relu as r


def cossine_similarity(q, k):
    q_norm = q / torch.linalg.norm(q, ord=2, dim=1)
    k_norm = k / torch.linalg.norm(k, ord=2, dim=1)
    return q_norm @ k_norm.T


def dpfp(x, nu=1):
    x = cat([r(x), r(-x)], dim=-1)
    x_rolled = cat([x.roll(shifts=j, dims=-1) for j in range(1, nu + 1)], dim=-1)
    x_repeat = cat([x] * nu, dim=-1)
    return x_repeat * x_rolled


class CompressiveMemory(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, nu: int = 1):
        super().__init__()

        self.embed_dim_k = embed_dim * (nu * 2)
        self.embed_dim_v = embed_dim
        self.num_heads = num_heads
        self.nu = nu

        memory = torch.zeros(
            (1, num_heads, self.embed_dim_k, self.embed_dim_v), dtype=torch.float32
        )
        z_norm = torch.zeros((1, num_heads, 1, self.embed_dim_k), dtype=torch.float32)

        self.register_buffer("memory", memory)
        self.register_buffer("z_norm", z_norm)

        self.activation = dpfp

    def get_retrieval_denominator(self, q_act):
        denominator = q_act @ self.z_norm.transpose(-2, -1) + 1e-8
        return denominator

    def sum_normalization(self, x):
        return x / x.sum(dim=-1, keepdim=True)

    def calculate_gamma(self, k_act: torch.Tensor) -> torch.Tensor:
        product = k_act @ self.z_norm.transpose(-2, -1)
        product = k_act * self.z_norm

        norm_k = torch.linalg.norm(k_act, ord=2, dim=-1, keepdim=True)

        gamma = 1 - (product / norm_k**2)
        return gamma

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        q_act = dpfp(q, nu=self.nu)
        q_act = self.sum_normalization(q_act)

        denominator = self.get_retrieval_denominator(q_act)
        numerator = q_act @ self.memory

        a_mem = numerator / denominator
        return a_mem

    def update_memory(
        self, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor = None
    ):
        """
        Beta eh um parametroque da a importancia da atualizacao (para sobrescrita de values)
        """
        k_act = dpfp(k, nu=self.nu)
        k_act = self.sum_normalization(k_act)

        v_anterior = self.retrieve(k)
        v_update = v - v_anterior
        if beta is not None:
            v_update = v_update * beta

        to_add = k_act.transpose(-2, -1) @ v_update
        self.memory = self.memory + to_add

        somatoria = k_act
        gamma = self.calculate_gamma(k_act)
        somatoria = k_act * gamma
        somatoria = somatoria.sum(dim=-2, keepdim=True)

        self.z_norm = self.z_norm + somatoria

    def reset(self):
        device = self.memory.device
        self.memory = torch.zeros(
            (1, num_heads, self.embed_dim_k, self.embed_dim_v),
            dtype=torch.float32,
            device=device,
        )
        self.z_norm = torch.zeros(
            (1, num_heads, 1, self.embed_dim_k), dtype=torch.float32, device=device
        )


if __name__ == "__main__":
    num_heads = 3
    batch_size = 6
    embed_dim = 256
    seq_len = 512

    mem = CompressiveMemory(embed_dim=embed_dim, num_heads=num_heads, nu=3)

    print("memory", mem.memory.shape)
    print("z_norm", mem.z_norm.shape)

    q1 = torch.randn((batch_size, num_heads, seq_len, embed_dim))
    k1 = torch.randn((batch_size, num_heads, seq_len, embed_dim))
    v1 = torch.randn((batch_size, num_heads, seq_len, embed_dim))

    retrieved = mem.retrieve(q1)
    print("retrieved", retrieved.shape)

    mem.update_memory(k1, v1, beta=torch.ones_like(v1) * 1.0)

    for _ in range(32):
        k_ = torch.randn((batch_size, num_heads, seq_len, embed_dim))
        v_ = torch.randn((batch_size, num_heads, seq_len, embed_dim))
        mem.update_memory(k_, v_, beta=torch.ones_like(v_) * 0.5)

        retrieved = mem.retrieve(k1)
        print(cossine_similarity(retrieved[0, 0, 0:1], v1[0, 0, 0:1]))
