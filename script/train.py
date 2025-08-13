#!/usr/bin/env python3
"""
Main training script for language model training.

This script provides a complete training pipeline with:
- Configurable model and optimizer hyperparameters
- Memory-efficient data loading with np.memmap
- Checkpoint saving and resuming
- Training and validation logging
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import cs336_basics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.nn.layers import TransformerLM
from cs336_basics.nn.data_loader import load_data
from cs336_basics.nn.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.nn.utils import gradient_clipping, cosine_lr_schedule
from cs336_basics.nn.functions import cross_entropy_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model')
    
    # Data arguments
    parser.add_argument('--train-data', type=str, required=True,
                        help='Path to training data (numpy memmap file)')
    parser.add_argument('--val-data', type=str, required=True,
                        help='Path to validation data (numpy memmap file)')
    parser.add_argument('--data-dtype', type=str, default='uint16',
                        help='Data type of the numpy arrays (default: uint16)')
    
    # Model arguments
    parser.add_argument('--vocab-size', type=int, required=True,
                        help='Vocabulary size')
    parser.add_argument('--context-length', type=int, default=256,
                        help='Maximum context length (default: 256)')
    parser.add_argument('--d-model', type=int, default=512,
                        help='Model dimension (default: 512)')
    parser.add_argument('--num-layers', type=int, default=6,
                        help='Number of transformer layers (default: 6)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--d-ff', type=int, default=2048,
                        help='Feed-forward dimension (default: 2048)')
    parser.add_argument('--rope-theta', type=float, default=10000.0,
                        help='RoPE theta parameter (default: 10000.0)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num-iterations', type=int, default=10000,
                        help='Number of training iterations (default: 10000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Maximum learning rate (default: 3e-4)')
    parser.add_argument('--min-learning-rate', type=float, default=3e-5,
                        help='Minimum learning rate (default: 3e-5)')
    parser.add_argument('--warmup-iters', type=int, default=100,
                        help='Number of warmup iterations (default: 100)')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                        help='Weight decay (default: 0.1)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping threshold (default: 1.0)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Save checkpoint every N iterations (default: 1000)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log training metrics every N iterations (default: 10)')
    parser.add_argument('--val-interval', type=int, default=100,
                        help='Evaluate on validation set every N iterations (default: 100)')
    parser.add_argument('--val-batches', type=int, default=10,
                        help='Number of validation batches to evaluate (default: 10)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (default: cuda if available)')
    
    return parser.parse_args()


def load_data_memmap(file_path, dtype='uint16'):
    """Load data using memory-mapped numpy array for efficiency."""
    # First, get the shape by loading the full array temporarily
    temp_array = np.load(file_path)
    shape = temp_array.shape
    del temp_array
    
    # Create a memory-mapped array
    memmap_path = file_path.replace('.npy', '.memmap')
    if not os.path.exists(memmap_path):
        # Create memmap file from npy
        arr = np.load(file_path)
        memmap_arr = np.memmap(memmap_path, dtype=dtype, mode='w+', shape=arr.shape)
        memmap_arr[:] = arr[:]
        memmap_arr.flush()
        del arr
    
    # Return memory-mapped array
    return np.memmap(memmap_path, dtype=dtype, mode='r', shape=shape)


def evaluate(model, val_data, batch_size, context_length, device, num_batches=10):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = load_data(val_data, batch_size, context_length, device)
            
            # Forward pass
            logits = model(x)
            
            # Calculate loss
            # Reshape for cross entropy: (batch * seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)
            
            loss = cross_entropy_loss(logits_flat, targets_flat)
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def main():
    args = parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data as memory-mapped arrays
    print("Loading training data...")
    train_data = load_data_memmap(args.train_data, args.data_dtype)
    print(f"Training data shape: {train_data.shape}")
    
    print("Loading validation data...")
    val_data = load_data_memmap(args.val_data, args.data_dtype)
    print(f"Validation data shape: {val_data.shape}")
    
    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device,
        dtype=torch.float32,
        theta=args.rope_theta
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_iteration = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from iteration {start_iteration}")
    
    # Training loop
    print("Starting training...")
    model.train()
    
    # Initialize metrics
    train_losses = []
    train_start_time = time.time()
    iteration_times = []
    
    for iteration in range(start_iteration, args.num_iterations):
        iter_start_time = time.time()
        
        # Get learning rate for this iteration
        lr = cosine_lr_schedule(
            iteration,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.num_iterations
        )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Load batch
        x, y = load_data(train_data, args.batch_size, args.context_length, device)
        
        # Forward pass
        logits = model(x)
        
        # Calculate loss
        # Reshape for cross entropy: (batch * seq_len, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.view(-1)
        
        loss = cross_entropy_loss(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Track metrics
        train_losses.append(loss.item())
        iteration_times.append(time.time() - iter_start_time)
        
        # Logging
        if (iteration + 1) % args.log_interval == 0:
            avg_loss = np.mean(train_losses[-args.log_interval:])
            avg_time = np.mean(iteration_times[-args.log_interval:])
            tokens_per_sec = (args.batch_size * args.context_length) / avg_time
            
            elapsed = time.time() - train_start_time
            print(f"Iter {iteration + 1}/{args.num_iterations} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tokens/sec: {tokens_per_sec:.0f} | "
                  f"Time/iter: {avg_time:.3f}s | "
                  f"Elapsed: {elapsed/60:.1f}min")
        
        # Validation
        if (iteration + 1) % args.val_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size, args.context_length, 
                              device, args.val_batches)
            print(f"Validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (iteration + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"checkpoint_iter_{iteration + 1}.pt"
            )
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_checkpoint.pt")
    save_checkpoint(model, optimizer, args.num_iterations, final_checkpoint_path)
    print(f"Training complete! Final checkpoint saved to {final_checkpoint_path}")
    
    # Final validation
    final_val_loss = evaluate(model, val_data, args.batch_size, args.context_length, 
                            device, args.val_batches * 10)  # More batches for final eval
    print(f"Final validation loss: {final_val_loss:.4f}")


if __name__ == "__main__":
    main()