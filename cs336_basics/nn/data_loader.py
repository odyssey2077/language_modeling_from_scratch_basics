import torch
import numpy as np


def load_data(x, batch_size, context_length, device):
    data_len = x.size
    num_possible_starts = data_len - context_length
    start_indices = np.random.randint(low=0, high=num_possible_starts, size=batch_size)
    x_indices = start_indices[:, np.newaxis] + np.arange(context_length)
    y_indices = x_indices + 1
    return torch.tensor(x[x_indices]).to(torch.long).to(device), torch.tensor(x[y_indices]).to(torch.long).to(device)