
import regex as re
from collections import defaultdict
from typing import Iterable, Iterator
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Token:
    def __init__(self, data: bytes, next=None):
        self.data = data
        self.next = next
    
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.invert_vocab = {v:k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        if self.special_tokens:
            self.special_tokens.sort(key=lambda x: len(x), reverse=True)

    @classmethod
    def from_files(cls, bpe_file_path):
        """
        Load a tokenizer from a BPE model file.
        Supports both pickle and JSON formats.
        """
        import pickle

        with open(bpe_file_path, 'rb') as f:
            data = pickle.load(f)
        
        return cls(data['vocab'], data['merges'], data['special_tokens'])
    
    def encode_token(self, token: bytes):
        byte_list = [token[i:i+1] for i in range(len(token))]
        head = Token(byte_list[0], None)
        cur = head
        for i in range(1, len(byte_list)):
            node = Token(byte_list[i], None)
            cur.next = node
            cur = cur.next
        for merge in self.merges:
            cur = head
            while cur and cur.next:
                if (cur.data, cur.next.data) == merge:
                    cur.data = cur.data + cur.next.data
                    cur.next = cur.next.next
                cur = cur.next
        int_list = []
        cur = head
        while cur:
            int_list.append(self.invert_vocab[cur.data])
            cur = cur.next
        return int_list

    def encode(self, text: str, token_ids_mapping: dict = {}) -> list[int]:
        # convert all tokens to int list
        token_ids_mapping = token_ids_mapping
        # pre-tokenize and handle special_token
        if text == '':
            return []
        
        if self.special_tokens:
            pattern = "|".join([re.escape(special_token) for special_token in self.special_tokens])
            parts = re.split(f'({pattern})', text)
        else:
            parts = [text]
        idx_bytes_mapping = defaultdict(lambda: [])
        new_tokens = set()
        for i in range(len(parts)):
            # skip special tokens
            if self.special_tokens and parts[i] in self.special_tokens:
                continue
            else:
                matches = re.finditer(PAT, parts[i])
                for match in matches:
                    word = match.group(0).encode("utf-8")
                    idx_bytes_mapping[i].append(word)
                    if not word in token_ids_mapping:
                        new_tokens.add(word)
        
        for token in new_tokens:
            token_ids_mapping[token] = self.encode_token(token)

        encode_result = []
        for i in range(len(parts)):
            # due to regex splitting, empty string appears before / after special tokens
            if parts[i] == '':
                continue
            if i not in idx_bytes_mapping:
                encode_result.append(self.invert_vocab[parts[i].encode("utf-8")])
            else:
                for token in idx_bytes_mapping[i]:
                    encode_result.extend(token_ids_mapping[token])
                
        return encode_result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient tokenization of an iterable of strings.
        Handles chunk boundaries correctly to ensure tokens don't get split.
        """
        token_ids_mapping = {}
        buffer = ""
        # 2GB in characters (assuming 1 byte per character for ASCII, up to 4 bytes for Unicode)
        # Using 2GB as the threshold
        MAX_BUFFER_SIZE = 1024 * 1024 * 100
        total_processing_time = 0.0
        processing_count = 0
        
        for chunk in iterable:
            # Add the new chunk to our buffer
            buffer += chunk
            
            # Only process if buffer is larger than 2GB
            if len(buffer) > MAX_BUFFER_SIZE:
                start_time = time.time()
                print("started processing")
                
                # Find the last safe split point using regex
                # We want to find the last complete token boundary
                matches = list(re.finditer(PAT, buffer))
                
                if matches:
                    # Find the end position of the last match
                    last_match_end = matches[-1].end()
                    
                    # Check if we've consumed the entire buffer
                    if last_match_end == len(buffer):
                        # All tokens are complete, process entire buffer
                        yield from self.encode(buffer, token_ids_mapping)
                        buffer = ""
                    else:
                        # Process up to the last complete token
                        to_process = buffer[:last_match_end]
                        yield from self.encode(to_process, token_ids_mapping)
                        # Keep the remainder for the next iteration
                        buffer = buffer[last_match_end:]
                
                elapsed_time = time.time() - start_time
                total_processing_time += elapsed_time
                processing_count += 1
                print(f"  Buffer processing #{processing_count}: {elapsed_time:.2f}s (total: {total_processing_time:.2f}s)")
        
        # Process any remaining text in the buffer
        if buffer:
            yield from self.encode(buffer, token_ids_mapping)
        
        if processing_count > 0:
            avg_time = total_processing_time / processing_count
            print(f"  Average buffer processing time: {avg_time:.2f}s over {processing_count} chunks")

    def decode(self, ids: list[int]) -> str:
        if len(ids) == 0:
            return ''
        raw_bytes =  b''.join([self.vocab[id] for id in ids])
        return raw_bytes.decode('utf-8', errors='replace')
        