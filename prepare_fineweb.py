import os
import argparse
import sys
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=100_000_000, help="Target token count")
    parser.add_argument("--out", type=str, default="fineweb_100m.tokens", help="Output file")
    args = parser.parse_args()

    print(f"Downloading FineWeb-Edu sample to generate {args.tokens/1e6:.1f}M tokens...")
    
    # Load the "sample-10BT" subset of FineWeb-Edu (high quality educational content)
    # Streaming mode prevents downloading the whole terabyte dataset.
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # Use GPT-2 tokenizer (standard for research comparisons)
    enc = tiktoken.get_encoding("gpt2")
    
    token_count = 0
    
    # We write as space-separated integers to match your v21 format
    # Using a buffer to minimize IO operations
    with open(args.out, "w", encoding="utf-8") as f:
        with tqdm(total=args.tokens, unit="tok") as pbar:
            for row in dataset:
                text = row['text']
                # encode_ordinary ignores special tokens, which is safer for raw text
                tokens = enc.encode_ordinary(text)
                tokens.append(enc.eot_token) # Add End-of-Text token delimiter
                
                # Write to file
                # We convert to string "t1 t2 t3..."
                f.write(" ".join(map(str, tokens)) + " ")
                
                len_toks = len(tokens)
                token_count += len_toks
                pbar.update(len_toks)
                
                if token_count >= args.tokens:
                    break
                
    print(f"\nSuccess! Saved {token_count:,} tokens to {args.out}")
    print(f"Vocab Size for this tokenizer is: {enc.n_vocab} (50257)")
    print("\nNOTE: When running v21, ensure you set --vocab-size 50304 (nearest multiple of 64) or use the auto-detect.")
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()