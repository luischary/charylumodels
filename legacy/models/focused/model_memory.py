import faiss
import numpy as np

def normalize_vector(vec):
    return vec / np.linalg.norm(vec, ord=2, axis=-1).reshape((-1, 1))

class ModelMemory:
    def __init__(self, embed_dim: int, top_k: int, max_size: int = 10_000):
        self.max_size = max_size
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.key_index = faiss.IndexFlatIP(self.embed_dim)
        # self.res = faiss.StandardGpuResources()  # use a single GPU
        # key_index = faiss.IndexFlatL2(self.embed_dim)
        # self.key_index = faiss.index_cpu_to_gpu(self.res, 0, key_index)
        
        self.keys_list = []
        self.values_list = []

    def add_key_value(self, keys: np.array, values: np.array):
        self.key_index.add(keys)

        values = list(values)
        self.values_list += values

        keys = list(keys)
        self.keys_list += keys

        # se passou do tamanho vai apagando os mais antigos
        if self.key_index.ntotal > self.max_size:
            to_remove = self.key_index.ntotal - self.max_size
            indices_to_remove = np.arange(to_remove)
            self.key_index.remove_ids(indices_to_remove)

            self.values_list = self.values_list[to_remove:]
            self.keys_list = self.keys_list[to_remove:]

    def reset(self):
        self.key_index.reset()
        self.values_list = []
        self.keys_list = []

    def query_top_k(self, queries: np.array):
        B, L, H, HD = queries.shape
        _, top_indices = self.key_index.search(queries.reshape((-1, self.embed_dim)), self.top_k)
        values_array = np.array(self.values_list)
        keys_array = np.array(self.keys_list)

        top_indices_flat = top_indices.reshape((-1,))
        keys_ans = keys_array[top_indices_flat].reshape(
            (B, L, H, self.top_k, HD)
        )
        values_ans = values_array[top_indices_flat].reshape(
            (B, L, H, self.top_k, HD)
        )

        return keys_ans, values_ans
