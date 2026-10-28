import torch
import torch.nn as nn
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(in_features, out_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.w, std=std, a = -3 * std, b = 3 * std)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.w, "... d_in, d_in d_out -> ... d_out") 

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.normal(mean=0, std=1, size=(num_embeddings, embedding_dim), device=device, dtype=dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = ((x ** 2).sum(dim=-1) / self.gain.shape[0] + self.eps) ** 0.5
        return (x / rms.unsqueeze(-1) * self.gain).to(in_dtype)

class SwiGLUFF(nn.Module):
    ...