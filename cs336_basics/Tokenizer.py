import ast
import regex as re
import json
from typing import Iterator, Iterable

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

    @classmethod
    def from_files(cls, 
                   vocab_filepath:str,
                    merges_filepath:str, 
                    special_tokens=None):
        
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        new_vocab: dict[tuple[bytes], int] = {}
        for k, v in vocab.items():
            v = v.encode('utf-8')
            new_vocab[k] = v 
        vocab = new_vocab


        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_txt = f.read().splitlines()
        
        merges = []
        for line in merges_txt:
            out = ast.literal_eval(line)
            merges.append(out)


        tokenizer = Tokenizer(vocab, merges, special_tokens)

        return tokenizer
    

    def encode(self, text: str) -> list[int]:
        # First split the text by each special token
        special_token_pat = re.compile(
            "(" + "|".join(
                re.escape(t)
                for t in sorted(self.special_tokens, key=len, reverse=True)
            ) + ")"
        )

        matches = re.split(special_token_pat, text)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        encoding_lst = []
        # Second, loop through each passage
        for line in matches:
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

                for merge in self.merges:
                    new_tup = []

                    i = 0
                    while i < len(tok):
                        # In pre_merge, we look at the next byte, it might be merged or not, so we use *** +1 *** rather than actual len of merge bytes
                        if i + 1 < len(tok) and tok[i] == merge[0] and tok[i+1] == merge[1]: 
                            new_tup.append(tok[i] + tok[i+1]) 
                            i += 2 # Because after each merging, the elem merged together, we only need 2
                        else:
                            new_tup.append(tok[i])      
                            i += 1    

                    new_tup = tuple(new_tup)
                    tok = new_tup


                # After these, all the elem in tok will can be looked up through dict
                for t in tok:
                    encoding_lst.append(self.rev_vocab[t])

                
        return encoding_lst

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        special_token_pat = "|".join(re.escape(tok) for tok in self.special_tokens)
        

        for it in iterable:
            lines = re.split(f"({special_token_pat})", it)

            for line in lines:
                # If there is special tokens, just skip
                if line in self.special_tokens:
                    line = line.encode('utf-8')
                    yield self.rev_vocab[line] # using yield as "computer and deliver"
                    continue

                matches = re.finditer(PAT, line)

                for match in matches:
                    s = match.group()

                    tok = s.encode('utf-8')
                    tok = tuple(bytes([b]) for b in tok)

                    for merge in self.merges:
                        new_tup = []

                        i = 0
                        while i < len(tok):
                            # In pre_merge, we look at the next byte, it might be merged or not, so we use *** +1 *** rather than actual len of merge bytes
                            if i + 1 < len(tok) and tok[i] == merge[0] and tok[i+1] == merge[1]: 
                                new_tup.append(tok[i] + tok[i+1]) 
                                i += 2 # Because after each merging, the elem merged together, we only need 2
                            else:
                                new_tup.append(tok[i])      
                                i += 1    

                        new_tup = tuple(new_tup)
                        tok = new_tup

                # After these, all the elem in tok will can be looked up through dict
                for t in tok:
                    yield self.rev_vocab[t] # using yield as "computer and deliver"


    
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
    tok = Tokenizer.from_files("/Users/xinghaozhou/Desktop/cs336/assignment1-basics/cs336_basics/vocab.json", "/Users/xinghaozhou/Desktop/cs336/assignment1-basics/cs336_basics/merges.txt", ["<|endoftext|>"])

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
            if count % 10 == 0:
                print(count)



    
    #tok.encode('')


