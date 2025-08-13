from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss            
    

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, weight_decay=0.01, epsilon=1e-8):
        defaults = {"lr": lr, "b1": b1, "b2": b2, "weight_decay": weight_decay, "epsilon": epsilon}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            b1 = group["b1"]
            b2 = group["b2"]
            weight_decay = group["weight_decay"]
            epsilon = group["epsilon"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data # Get the gradient of loss with respect to p.
                state = self.state[p] # Get state associated with p.
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad**2
                alpha_t = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.data -= alpha_t * m / (math.sqrt(v) + epsilon)
                p.data -= alpha_t * weight_decay * p.data
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss         