import torch
import math
from einops import einsum


def softmax(x: torch.Tensor, dim: int):
    max_v, _ = torch.max(x, dim=dim, keepdim=True)
    x = torch.exp(x - max_v)
    return x / torch.sum(x, dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, d_k: int, mask: torch.Tensor | None = None):
    q_k = einsum(q, k, "... n dk, ... m dk -> ... n m") / math.sqrt(d_k)
    if mask is not None:
        q_k.masked_fill_(~mask, value=-torch.inf)
    attention =  softmax(q_k, dim=-1)
    return einsum(attention, v, "... n m, ... m d_v -> ... n d_v")


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    max_v, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - max_v
    selected_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    denominator = torch.sum(torch.exp(logits), dim=-1)
    return -torch.mean(selected_logits - torch.log(denominator))