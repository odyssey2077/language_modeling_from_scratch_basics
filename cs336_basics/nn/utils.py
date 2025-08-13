import math
import torch
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

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, epsilon=1e-6):
    grad_data = []
    for p in parameters:
        if p.grad is not None:
            grad_data.append(p.grad.data.view(-1))

    if len(grad_data) == 0:
        return

    all_grads = torch.cat(grad_data)
    l2_norm = torch.norm(all_grads).item()

    if l2_norm > max_l2_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data = p.grad.data * (max_l2_norm / (l2_norm + epsilon))

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





