import regex as re
import json
from typing import Iterator, Iterable
from cs336_basics.b2u import gpt2_bytes_to_unicode

class Tokenizer:
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens=None):
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        # Make a reverse look up vocab (bytes -> id) 
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

        # Make a rank merges for faster look up
        self.rank_merges = {(a, b): i for i, (a, b) in enumerate(self.merges)}

    @classmethod
    def from_files(cls, 
                   vocab_filepath:str,
                    merges_filepath:str, 
                    special_tokens=None):
        
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        b2u = gpt2_bytes_to_unicode()
        rev_b2u = {v: k for k, v in b2u.items()}

        new_vocab: dict[tuple[bytes], int] = {}
        for k, v in vocab.items():
            if v in special_tokens:
                v = v.encode('utf-8')
            else:
                v = bytes(rev_b2u[c] for c in v)

            new_vocab[k] = v
        vocab = new_vocab



        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_txt = f.read().splitlines()
        
        merges = []
        for i, line in enumerate(merges_txt):
            a_str, b_str = line.rstrip("\n").split(" ", 1)

            a = bytes(rev_b2u[bs] for bs in a_str)
            b = bytes(rev_b2u[bs] for bs in b_str)

            merges.append((a, b)) 

        tokenizer = Tokenizer(vocab, merges, special_tokens)

        return tokenizer
    

    def encode(self, text: str) -> list[int]:
        # Split the text by each special token
        special_token_pat = re.compile(
            "(" + "|".join(
                re.escape(t)
                for t in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"
        )

        # First, need to check if the self.special_tokens is empty.
        # If no special tokens, do not use special_token_pat split, otherwise, it splits on character
        if not self.special_tokens:
            lines = [text]
        else:
            matches = re.split(special_token_pat, text)
            lines = matches # just for convenience 

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        encoding_lst = []
        # Second, loop through each passage
        for line in lines:
            # If there is special tokens
            if line in self.special_tokens:
                line = line.encode('utf-8')
                encoding_lst.append(self.rev_vocab[line])
                continue

            # Make the passage as a list of <pre-token>
            matches = re.finditer(PAT, line)

            # For each pre-tok in the list, encode this 
            for match in matches:
                s = match.group()

                # For each elem, make it tuple-like bytes! 
                tok = s.encode('utf-8')
                tok = tuple(bytes([b]) for b in tok)

                while True:

                    if len(tok) < 2:
                        break

                    # All possible pairs
                    pairs = [(tok[i], tok[i+1]) for i in range(len(tok)-1)]

                    # Find the lowest rank

                    min_rank = float("inf")
                    min_pair = None
                    for pair in pairs:
                        cur_rank = self.rank_merges.get(pair, float("inf"))

                        if cur_rank < min_rank:
                            min_rank = cur_rank
                            min_pair = pair

                    if min_rank == float("inf") or min_pair is None:
                        break

                    # No more merges
                    if min_pair not in self.rank_merges:
                        break

                    i = 0
                    new_tok = []

                    # Scan and merge.
                    while i < len(tok):
                        if i < len(tok) - 1 and (tok[i], tok[i+1]) == min_pair:
                            new_tok.append(tok[i] + tok[i+1])
                            i += 2
                        else:
                            new_tok.append(tok[i])
                            i += 1

                    tok = tuple(new_tok)

                # After these, all the elem in tok will can be looked up through dict
                for t in tok:
                    encoding_lst.append(self.rev_vocab[t])

        return encoding_lst

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    
    def decode(self, ids: list[int]) -> str:

        pre_dec: bytes = b""
        for id in ids:
            if id not in self.vocab:
                pre_dec += b"\xef\xbf\xbd"
            else:
                pre_dec+=self.vocab[id]

        output = pre_dec.decode("utf-8", errors="replace")

        return output

if __name__ == "__main__":
    tok = Tokenizer.from_files("/Users/xinghaozhou/Desktop/cs336/assignment1-basics/cs336_basics/TinyStoriesV2-GPT4-train-vocab.json", "/Users/xinghaozhou/Desktop/cs336/assignment1-basics/cs336_basics/TinyStoriesV2-GPT4-train-merges.txt", ["<|endoftext|>"])

    import numpy as np
    from tqdm.auto import tqdm
    txt_path = "/Users/xinghaozhou/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    bin_path = "TinyStoriesV2-GPT4-valid.bin"

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()   

    count = 0

    with open(bin_path, "wb") as out_f:
        for token_id in tqdm(tok.encode_iterable([text])):
            count += 1
            out_f.write(np.int32(token_id).tobytes())


    
    #tok.encode('')


