import torch


def count_parameters(model):
    n_params = sum(p.numel() for p in model.parameters())
    print("number of parameters: %.2fM" % (n_params / 1e6,))


def create_causal_mask(query_len, key_len):
    if key_len > query_len:
        triangular_shape = (query_len, query_len)
        excedente = key_len - query_len
        mask = torch.concat(
            [
                torch.ones(query_len, excedente),
                torch.tril(torch.ones(triangular_shape)),
            ],
            dim=1,
        )
    elif query_len > key_len:
        raise NotImplementedError("O tamanho da query esta maior que da key")
    else:
        triangular_shape = (query_len, key_len)
        mask = torch.tril(torch.ones(triangular_shape))

    return mask.reshape(1, 1, query_len, key_len)


class TransformerCache:
    def __init__(self, cache_max_len: int = 8192):
        self.max_len = cache_max_len
        self.k = None
        self.v = None

    def update_v_cache(self, x):
        """
        x - [batch, len, heads, head_dim]
        """
        if self.v is None:
            self.v = x
        else:
            self.v = torch.cat([self.v, x], dim=2)

        if self.v.shape[2] > self.max_len:
            self.v = self.v[:, :, -self.max_len :, :]

    def update_k_cache(self, x):
        """
        x - [batch, len, heads, head_dim]
        """
        if self.k is None:
            self.k = x
        else:
            self.k = torch.cat([self.k, x], dim=2)

        if self.k.shape[2] > self.max_len:
            self.k = self.k[:, :, -self.max_len :, :]

    def add_v_cache(self, x, update_cache: bool = True):
        """
        x - [batch, heads, len, head_dim]
        """

        if self.v is not None:
            ans = torch.cat([self.v, x], dim=2)
        else:
            ans = x

        if update_cache:
            self.update_v_cache(x)
        return ans

    def add_k_cache(self, x, update_cache: bool = True):
        """
        x - [batch, heads, len, head_dim]
        """
        if self.k is not None:
            ans = torch.cat([self.k, x], dim=2)
        else:
            ans = x

        if update_cache:
            self.update_k_cache(x)
        return ans
