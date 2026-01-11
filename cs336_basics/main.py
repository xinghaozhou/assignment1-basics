import os
import regex as re
import multiprocessing

def train_bpe(input_path: str, 
              vocab_size: int,
              special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tok: dict[tuple[bytes], int] = {} # This is a generics for dict

    # This part is pre-tokenization
    with open(input_path, 'r', encoding="utf-8") as f:
        content = f.read()

        special_token_pat = "|".join(special_tokens)
        lines = re.split(special_token_pat, content, maxsplit=0, flags=0)

    for line in lines:
        words = line.split(" ")
        for word in words:
            chars = []
            for char in word:
                char = char.encode('utf-8')
                chars.append(char)
            
        tup = tuple(chars)
            
        if tup not in pre_tok:
            pre_tok[tup] = 1
        else:
            pre_tok[tup] +=1


    # What's in the list
    lst = []
    lst.extend(special_tokens)
    lst.extend


    post_merge, lst = merge(pre_tok)


    return 

def merge(pre_merge: dict[tuple[bytes], int])-> tuple[dict[tuple[bytes], int], list[tuple[bytes, bytes]]]:

    count: dict[tuple[bytes], int] = {}
    post_merge: dict[tuple[bytes], int] = {}

    # Loop through all the bytes and find the most frequent pair in this iteration.
    for tup_bytes in pre_merge:
        pair = []
        for i in range(len(tup_bytes)-1):
            pair = tup_bytes[i] + tup_bytes[i+1]
            if pair not in count:
                count[pair] = pre_merge[tup_bytes]
            else:
                count[pair] += pre_merge[tup_bytes]

    # Find the most frequent pair
    mst_freq_pair = max(count.items(), key=lambda item: item[1])
    mst_freq_bytes = mst_freq_pair[0]
    mst_freq_bytes_separate = tuple([mst_freq_bytes[i:i+1] for i in range(len(mst_freq_bytes))])
    
    # To merge the mst_freq_bytes in key
    for k, v in pre_merge.items():
        new_tup = []
        for i in range(len(k) - len(mst_freq_bytes_separate) + 1):
            # If bytes match up to mst_freq_bytes, merge
            if k[i: i+len(mst_freq_bytes_separate)] == mst_freq_bytes_separate:
                new_tup.append(mst_freq_bytes)  
                i = i+len(mst_freq_bytes_separate)
            else:
                new_tup.append(k[i])          

        new_tup = tuple(new_tup)
        
        post_merge[new_tup] = v

   
    return post_merge, [mst_freq_bytes]
        




        
            
            



    return 


            

    

        





    




if __name__ == "__main__":
    train_bpe("/Users/xinghaozhou/Desktop/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 1, ['<|endoftext|>'])
