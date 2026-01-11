import os
import regex as re
import multiprocessing
import time
import time, psutil, os
from threading import Thread
import json

process = psutil.Process(os.getpid())
peak_mem = 0
running = True

def monitor():
    global peak_mem
    while running:
        mem = process.memory_info().rss
        peak_mem = max(peak_mem, mem)
        time.sleep(0.05)   # sample every 50ms



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    import regex as re
    from multiprocessing import Pool

    # Pre-tokenization
    # Count the frequency of str in text

    pre_start = time.time()
    with open(input_path, 'r', encoding="utf-8") as f:
        content = f.read()

        special_token_pat = "|".join(re.escape(tok) for tok in special_tokens)
        lines = re.split(special_token_pat, content)

    # Run multiprocessing
    with Pool(processes=10) as pool:
        pre_tok_list = pool.map(pre_tokenization, lines)

    pre_tok: dict[str, int] = {}


    for elem in pre_tok_list:
        for k, v in elem.items():
            pre_tok[k] = pre_tok.get(k, 0) + v


    tok: dict[tuple[bytes], int] = {}
    for k, v in pre_tok.items():
        k = k.encode('utf-8')
        t_b = tuple(bytes([b]) for b in k)
        tok[t_b] = v

    pre_end = time.time()

    pre_time =  print(f"Pre-tokenization takes {(pre_end - pre_start)}")

    merge_start = time.time()

    # Merge processing
    vocab: dict[int, bytes] = {}
    len_spec_tokens = len(special_tokens)

    # !!! Add special token bytes into vocab (special token should be removed from text)
    for i in range(len_spec_tokens):
        vocab[i] = special_tokens[i].encode('utf-8')

    # Add 0-255 bytes char
    for i in range(0, 256):
        vocab[i+len_spec_tokens] = bytes([i])

    # Check how many more vocab needed to be merged 
    lst = []
    while len(lst) < (vocab_size - 256 - len_spec_tokens): # Already have 256 bytes and special_tokens
        post_tok, mst_freq_bytes = merge(tok)
        tok = post_tok
        lst.append(mst_freq_bytes)

    merge_end = time.time()
    print(f"Merge takes {merge_end - merge_start}")

    # Use index, start from existing vocab and continue adding more...
    index = len(vocab)
    for i, merged_t in enumerate(lst):
        vocab[index + i] = merged_t[0] + merged_t[1]

    vocab_out = {i: v.decode("latin1") for i, v in vocab.items()}
    with open("vocab.json", "w") as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=2)

    with open("merges.txt", "w") as f:
        for a, b in lst:
            f.write(f"{a.decode('utf-8')} {b.decode('utf-8')}\n")

    mst_longest_key = max(
        vocab.items(),
        key = lambda item: len(item[1])
    )

    print(mst_longest_key)

    return vocab, lst

def merge(pre_merge: dict[tuple[bytes], int]
          )-> tuple[dict[tuple[bytes], int], tuple[bytes, bytes]]:
    """
        pre_merge: dict[(bytes, ...): freq]
        return:
            post_merge: dict[(merged bytes, ...): freq]
            mst_freq_btyes: tuple(bytes1, bytes2)    
    """

    count: dict[tuple[bytes], int] = {}
    post_merge: dict[tuple[bytes], int] = {}

    # Loop through all the bytes and find the most frequent pair in this iteration.
    for t, freq in pre_merge.items():
        for i in range(len(t)-1):
            pair = (t[i], t[i+1])
            count[pair] = count.get(pair, 0) + freq

    # Find the most frequent pair
    mst_freq_pair = max(
        count.items(),
        key=lambda item: (item[1], item[0])   # frequency first, then lexicographic
    )
    mst_freq_bytes = mst_freq_pair[0]
    tokA, tokB = mst_freq_bytes
    
    # To merge the mst_freq_bytes in key
    for k, v in pre_merge.items():
        new_tup = []
        
        i = 0
        while i < len(k):
            # In pre_merge, we look at the next byte, it might be merged or not, so we use *** +1 *** rather than actual len of merge bytes
            if i + 1 < len(k) and k[i] == tokA and k[i+1] == tokB: 
                new_tup.append(mst_freq_bytes[0] + mst_freq_bytes[1]) 
                i += 2 # Because after each merging, the elem merged together, we only need 2
            else:
                new_tup.append(k[i])      
                i += 1    

        new_tup = tuple(new_tup)
        post_merge[new_tup] = post_merge.get(new_tup, 0) + v # It might have existing key there

    return post_merge, mst_freq_bytes

def pre_tokenization(line: str) -> dict[str, int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tok: dict[str, int] = {} # This is a generics for dict

    if not line:
        return {} # Prevent returning NoneType
    
    matches = re.finditer(PAT, line)

    for match in matches:
        s = match.group()
        if s not in pre_tok:
            pre_tok[s] = 1
        else:
            pre_tok[s] += 1
        
    return pre_tok



    

if __name__ == "__main__":
    # start monitoring thread
    t = Thread(target=monitor)
    t.start()
    start = time.time()
    run_train_bpe("/Users/xinghaozhou/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 10000, ['<|endoftext|>'])
    end = time.time()

    running = False
    t.join()
    print("Total time cost:", end-start)
    print(f"Peak memory: {peak_mem / (1024**3):.3f} GB")
