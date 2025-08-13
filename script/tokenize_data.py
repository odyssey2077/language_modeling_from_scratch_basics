#!/usr/bin/env python3
"""
Script to tokenize specific text files using their corresponding BPE models
and save the tokenized outputs as numpy arrays.
"""

import os
import sys
import numpy as np
import time
from pathlib import Path

# Add parent directory to path to import cs336_basics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.tokenizer import Tokenizer


def tokenize_file(tokenizer, input_path, output_path):
    """Tokenize a single file and save as numpy array using memory-efficient streaming"""
    print(f"  Tokenizing {input_path.name}...")
    
    start_time = time.time()
    
    try:
        # Process file in chunks using encode_iterable
        token_ids = []
        
        with open(input_path, 'r', encoding='utf-8') as f:    
            # Use encode_iterable to process chunks
            for token_id in tokenizer.encode_iterable(f):
                token_ids.append(token_id)
        
        # Convert to numpy array of uint16
        tokens_array = np.array(token_ids, dtype=np.uint16)
        
        # Save as numpy array
        np.save(output_path, tokens_array)
        
        elapsed_time = time.time() - start_time
        tokens_per_second = len(token_ids) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"    Saved {len(token_ids)} tokens to {output_path}")
        print(f"    Time taken: {elapsed_time:.2f} seconds ({tokens_per_second:.0f} tokens/second)")
        
        # Verify by decoding first 100 tokens
        if len(token_ids) > 0:
            sample_tokens = token_ids[:100] if len(token_ids) > 100 else token_ids
            decoded_text = tokenizer.decode(sample_tokens)
            # Note: Can't easily verify against original due to chunking
            print(f"    First {len(sample_tokens)} tokens decode to {len(decoded_text)} characters")
        
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"    Error processing {input_path.name} after {elapsed_time:.2f} seconds: {e}")
        return False


def main():
    # Define paths
    data_dir = Path("data")
    bpe_output_dir = Path("bpe_output")
    output_dir = Path("tokenizer_output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Define file-model mappings
    mappings = [
        {
            "model": bpe_output_dir / "owt.pkl",
            "files": [
                data_dir / "owt_train.txt",
                data_dir / "owt_valid.txt"
            ]
        },
        # {
        #     "model": bpe_output_dir / "tiny_story_train.pkl",
        #     "files": [
        #         data_dir / "TinyStoriesV2-GPT4-train.txt",
        #         data_dir / "TinyStoriesV2-GPT4-valid.txt"
        #     ]
        # }
    ]
    
    # Process each mapping
    for mapping in mappings:
        model_path = mapping["model"]
        
        print(f"\nProcessing with model: {model_path.name}")
        
        # Check if model exists
        if not model_path.exists():
            print(f"  ERROR: Model file {model_path} not found")
            continue
        
        # Load tokenizer
        try:
            tokenizer = Tokenizer.from_files(model_path)
            print(f"  Successfully loaded tokenizer from {model_path}")
        except Exception as e:
            print(f"  ERROR: Failed to load tokenizer from {model_path}: {e}")
            continue
        
        # Create subdirectory for this model's outputs
        model_output_dir = output_dir / model_path.stem
        model_output_dir.mkdir(exist_ok=True)
        
        # Process each file for this model
        for input_file in mapping["files"]:
            if not input_file.exists():
                print(f"  WARNING: Input file {input_file} not found")
                continue
            
            # Generate output filename
            output_path = model_output_dir / f"{input_file.stem}.npy"
            
            # Tokenize and save
            tokenize_file(tokenizer, input_file, output_path)
    
    print(f"\nTokenization complete. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()