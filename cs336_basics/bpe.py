from cs336_basics.pretokenization import find_chunk_boundaries
import regex as re
from collections import defaultdict
from typing import Optional
from multiprocessing import Pool, cpu_count
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Token:
    def __init__(self, data: bytes, ord: int, count: int, prev=None, next=None):
        self.data = data
        self.ord = ord
        self.count = count
        self.prev = prev
        self.next = next

    # return the pairs invalidated and the new pairs, and the merged pair
    # assume self.next always exists
    def merge(self, pairs_frequency: dict[bytes, int], pairs_occurences: dict[bytes, set]) -> tuple[bytes, bytes]:
        cur_pair = (self.data, self.next.data)
        pairs_occurences[cur_pair].remove(self)
        if self.prev:
            pair_invalidated = (self.prev.data, self.data)
            pairs_occurences[pair_invalidated].remove(self.prev)
            pairs_frequency[pair_invalidated] -= self.count
        if self.next.next:
            pair_invalidated = (self.next.data, self.next.next.data)
            pairs_occurences[pair_invalidated].remove(self.next)
            pairs_frequency[pair_invalidated] -= self.count       

        pair = (self.data, self.next.data)
        # merge the pair
        self.data = self.data + self.next.data
        self.next = self.next.next
        if self.next:
            self.next.prev = self

        if self.prev:
            pair_added = (self.prev.data, self.data)
            pairs_occurences[pair_added].add(self.prev)
            pairs_frequency[pair_added] += self.count            
        if self.next:
            pair_added = (self.data, self.next.data)
            pairs_occurences[pair_added].add(self)
            pairs_frequency[pair_added] += self.count             
        
        return pair


# initialize token double linked list for efficient count update, count the initial pair frequency as well
def tokenize(word: bytes, word_count: int, pairs_occurences: dict[bytes, dict], pairs_frequency: dict[bytes, int]) -> Token:
    byte_list = [word[i:i+1] for i in range(len(word))]
    if len(byte_list) == 0:
        raise ValueError("cannot tokenize empty word")
    cur = Token(byte_list[0], 0, word_count, None, None)
    for i in range(1, len(byte_list)):
        node = Token(byte_list[i], cur.ord+1, word_count, cur, None)
        cur.next = node
        cur = cur.next
        pairs_occurences[(byte_list[i-1], byte_list[i])].add(cur.prev)
        pairs_frequency[(byte_list[i-1], byte_list[i])] += word_count


def get_pair_with_max_count(pairs_frequency: dict[bytes, int]) -> bytes:
    candidate_set = set()
    max_count = -1
    for b, count in pairs_frequency.items():
        if count > max_count or max_count == -1:
            max_count = count
            candidate_set = {b}
        elif count == max_count:
            candidate_set.add(b)

    return max(candidate_set)


def process_chunk(args):
    """Process a single chunk of the file"""
    input_path, start, end, special_tokens = args
    word_frequency = {}
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        
        # Remove special tokens
        chunks = re.split("|".join([re.escape(special_token) for special_token in special_tokens]), chunk)
        for chunk in chunks:
            matches = re.finditer(PAT, chunk)
            for match in matches:
                word = match.group(0).encode("utf-8")
                word_frequency[word] = word_frequency.get(word, 0) + 1
    
    return word_frequency


def build_word_frequency_table(input_path: str, special_tokens: list[str], num_chunks, num_processes) -> dict[bytes, int]:
    """Read file and build word frequency table using multiprocessing"""    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks)
    
    # Prepare arguments for each process
    process_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        process_args.append((input_path, start, end, special_tokens))
    
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, process_args)
    
    # Merge results from all processes
    word_frequency_table = {}
    for result in results:
        for word, count in result.items():
            word_frequency_table[word] = word_frequency_table.get(word, 0) + count
    
    return word_frequency_table
        

def bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_chunks=8, num_processes=8) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    word_frequency_table = build_word_frequency_table(input_path, special_tokens, num_chunks, num_processes)
    
    vocab = {}
    merges = []
    for i in range(256):
        vocab[i] = i.to_bytes(1, 'big')
    for special_token in special_tokens:
        idx = len(vocab)
        vocab[idx] = special_token.encode("utf-8")
    pairs_frequency = defaultdict(lambda: 0)
    pairs_occurences = defaultdict(lambda: set({}))
    for word in word_frequency_table.keys():
        tokenize(word, word_frequency_table[word], pairs_occurences, pairs_frequency)
    existing_vocab_size = len(vocab.keys())
    for _ in range(vocab_size - existing_vocab_size):
        pair_with_max_count = get_pair_with_max_count(pairs_frequency)
        idx = len(vocab)
        vocab[idx] = pair_with_max_count[0] + pair_with_max_count[1]
        occurences = sorted(pairs_occurences[pair_with_max_count], key=lambda x: (x.data, x.ord))
        for occurence in occurences:
            # could be deleted due to previous merges
            if occurence in pairs_occurences[pair_with_max_count]:      
                occurence.merge(pairs_frequency, pairs_occurences)
        merges.append(pair_with_max_count)
        del pairs_occurences[pair_with_max_count]
        del pairs_frequency[pair_with_max_count]

    return vocab, merges


def main():
    import argparse
    import json
    import pickle
    
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
    parser.add_argument('input_path', type=str, help='Path to input text file')
    parser.add_argument('vocab_size', type=int, help='Target vocabulary size')
    parser.add_argument('--special-tokens', nargs='*', default=["<|endoftext|>"], help='Special tokens to add to vocabulary')
    parser.add_argument('--output', type=str, default='bpe_output/bpe_model.pkl', help='Output path for serialized model')
    parser.add_argument('--format', choices=['pickle', 'json'], default='pickle', help='Output format')
    
    args = parser.parse_args()
    
    # Train BPE
    import time
    print(f"Training BPE on {args.input_path} with vocab size {args.vocab_size}...")
        
    start_time = time.time()
    vocab, merges = bpe(args.input_path, args.vocab_size, args.special_tokens, 100, min(cpu_count(), 8))
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Serialize results
    if args.format == 'pickle':
        with open(args.output, 'wb') as f:
            pickle.dump({'vocab': vocab, 'merges': merges, 'special_tokens': args.special_tokens}, f)
        print(f"Saved BPE model to {args.output} (pickle format)")
    else:  # json
        # Convert bytes to base64 strings for JSON serialization
        json_vocab = {k: v.decode('utf-8', errors='surrogateescape') for k, v in vocab.items()}
        json_merges = [(a.decode('utf-8', errors='surrogateescape'), 
                        b.decode('utf-8', errors='surrogateescape')) for a, b in merges]
        
        with open(args.output, 'w') as f:
            json.dump({
                'vocab': json_vocab,
                'merges': json_merges,
                'special_tokens': args.special_tokens
            }, f, indent=2)
        print(f"Saved BPE model to {args.output} (JSON format)")
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    if args.special_tokens:
        print(f"Special tokens: {args.special_tokens}")


if __name__ == "__main__":
    main()
