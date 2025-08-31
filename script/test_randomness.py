import numpy as np
import torch
import timeit

num_possible_starts = 541229347
batch_size = 64
n_runs = 2

# --- Group 1: Unique Samples (Without Replacement) ---
def slow_unique_numpy():
    # The slow way to get unique samples
    return np.random.choice(a=num_possible_starts, size=batch_size, replace=False)

rng = np.random.default_rng()
def fast_unique_numpy():
    # The fast way to get unique samples
    return rng.permutation(num_possible_starts)[:batch_size]

def fast_unique_pytorch():
    # The fast way to get unique samples
    return torch.randperm(num_possible_starts)[:batch_size]

# --- Group 2: Samples With Replacement ---
def fast_replacement_numpy():
    # Standard way to get samples with replacement
    return np.random.randint(0, num_possible_starts, size=batch_size)

def fast_replacement_pytorch():
    # Standard way to get samples with replacement
    return torch.randint(0, num_possible_starts, size=(batch_size,))


# --- Timing the results ---
t_slow_unique_np = timeit.timeit(slow_unique_numpy, number=n_runs)
t_fast_unique_np = timeit.timeit(fast_unique_numpy, number=n_runs)
t_fast_unique_torch = timeit.timeit(fast_unique_pytorch, number=n_runs)
t_fast_replace_np = timeit.timeit(fast_replacement_numpy, number=n_runs)
t_fast_replace_torch = timeit.timeit(fast_replacement_pytorch, number=n_runs)

print("--- Unique Samples (Without Replacement) ---")
print(f"Slow NumPy (np.random.choice): {t_slow_unique_np:.4f} seconds")
print(f"Fast NumPy (rng.permutation):   {t_fast_unique_np:.4f} seconds")
print(f"Fast PyTorch (torch.randperm):  {t_fast_unique_torch:.4f} seconds")
print("\n--- Samples With Replacement ---")
print(f"Fast NumPy (np.random.randint): {t_fast_replace_np:.4f} seconds")
print(f"Fast PyTorch (torch.randint):   {t_fast_replace_torch:.4f} seconds")

# Example Output:
# --- Unique Samples (Without Replacement) ---
# Slow NumPy (np.random.choice): 2.5401 seconds
# Fast NumPy (rng.permutation):   0.0161 seconds
# Fast PyTorch (torch.randperm):  0.0120 seconds
# 
# --- Samples With Replacement ---
# Fast NumPy (np.random.randint): 0.0025 seconds
# Fast PyTorch (torch.randint):   0.0031 seconds