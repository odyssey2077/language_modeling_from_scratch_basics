import math
import torch
import torch.nn as nn
import numpy as np
from collections.abc import Iterable
from cs336_basics.nn.functions import softmax

def cosine_lr_schedule(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return alpha_max * t / t_w
    elif t < t_c:
        return alpha_min + 0.5 * (1 + math.cos(((t - t_w) / (t_c - t_w)) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min

def gradient_clipping(parameters: Iterable[nn.Parameter], max_norm: float):
    """
    Correctly clips the gradients of a model's parameters.

    Args:
        parameters: An iterable of parameters to clip the gradients of.
        max_norm: The maximum L2 norm of the gradients.
    """
    # Filter for parameters that have gradients
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    # Calculate the total L2 norm of all gradients
    # We use a generator expression for memory efficiency
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2
    )

    # Calculate the clipping ratio
    clip_coef = max_norm / (total_norm + 1e-6)

    # Clip the gradients if the total norm exceeds max_norm
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)

# assume prompt is a one dimension tensor with shape [seq_len]
def decode(model: torch.nn.Module, prompt: torch.Tensor, special_token_id: int = -1, max_num_tokens: int | None = None, temperature: float | None = None, p: float | None = None) -> torch.Tensor:    
    count = 0
    while count < max_num_tokens:
        # shape [seq_len, vocab_size]
        y = model(prompt.unsqueeze(0))
        logits = y[0][-1]
        if temperature is not None:
            logits = logits / temperature
        probability = softmax(logits, dim=-1)
        if p is not None:
            # 1. Sort probabilities and keep their original indices
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            
            # 2. Find the smallest set of tokens whose cumulative probability is >= p
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            
            # Shift the mask to the right to ensure we keep the first token that exceeds p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False # Never remove the most likely token

            # 3. Zero out the probabilities of tokens to remove
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probabilities[indices_to_remove] = 0
            
            # 4. Re-normalize the remaining probabilities
            probabilities = probabilities / torch.sum(probabilities)
            
        sampled_idx = np.random.choice(np.arange(len(probability)), p=probability)
        prompt = torch.cat((prompt, torch.tensor([sampled_idx]).to(prompt.device)), dim=0)
        count +=1
        if sampled_idx == special_token_id:
            break
    
    return prompt





