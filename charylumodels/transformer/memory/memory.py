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
    def __init__(self, embed_dim_k: int, embed_dim_v: int, num_heads: int):
        super().__init__()

        self.embed_dim_k = embed_dim_k
        self.embed_dim_v = embed_dim_v
        self.num_heads = num_heads

        memory = torch.zeros((embed_dim_k, embed_dim_v))
        z_norm = torch.zeros((1, self.embed_dim_k))

        self.register_buffer("memory", memory)
        self.register_buffer("z_norm", z_norm)

        self.activation = dpfp

    def get_retrieval_denominator(self, q_act):
        denominator = q_act @ self.z_norm.transpose(-2, -1) + 1e-8
        return denominator

    def sum_normalization(self, x):
        return x / x.sum(dim=-1, keepdim=True)

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        q_act = self.activation(q)
        q_act = self.sum_normalization(q_act)

        denominator = self.get_retrieval_denominator(q_act)
        numerator = q_act @ self.memory

        a_mem = numerator / denominator
        return a_mem

    def update_memory(self, k: torch.Tensor, v: torch.Tensor):
        k_act = self.activation(k)
        k_act = self.sum_normalization(k_act)

        to_add = k_act.transpose(-2, -1) @ (v - self.retrieve(k))

        self.memory = self.memory + to_add

        somatoria = k_act

        self.z_norm = self.z_norm + somatoria


if __name__ == "__main__":
    num_heads = 1
    embed_dim = 1024

    mem = CompressiveMemory(
        embed_dim_k=embed_dim * 2, embed_dim_v=embed_dim * 1, num_heads=num_heads
    )

    print("memory", mem.memory.shape)
    print("z_norm", mem.z_norm.shape)

    q1 = torch.randn((1, embed_dim))
    k1 = torch.randn((1, embed_dim))
    v1 = torch.randn((1, embed_dim))

    retrieved = mem.retrieve(q1)
    print("retrieved", retrieved.shape)

    mem.update_memory(k1, v1)

    for _ in range(1000):
        k_ = torch.randn((1, embed_dim))
        v_ = torch.randn((1, embed_dim))
        mem.update_memory(k_, v_)

        retrieved = mem.retrieve(k1)
        print(cossine_similarity(retrieved, v1))
