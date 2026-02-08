import torch
from cs336_basics.Tokenizer import Tokenizer
from cs336_basics.checkpoint import load_checkpoint
from cs336_basics.inference import inference
from cs336_basics.transformer import transformer_lm
from cs336_basics.adamW import AdamW

import argparse

parser = argparse.ArgumentParser()

# Device
parser.add_argument("--device", help="Which device", default='mps')

parser.add_argument("--context_length", help="Context length", type=int)

# Transformers parts
parser.add_argument("--vocab_size", help="Vocab size", type=int)
parser.add_argument("--d_model", help="dimension of embeddings", type=int)
parser.add_argument("--num_layers", help="num of transformer blocks", type=int)
parser.add_argument("--num_heads", help="num of heads in multi-head attention", type=int)
parser.add_argument("--d_ff", help="dimension of feed-forward network", type=int)
parser.add_argument("--rope_theta", help="theta in RoPE", type=float)

# Optimizer parts
parser.add_argument("--betas", help="betas of AdamW", type=float, default=(0.9, 0.95), nargs=2)
parser.add_argument("--eps", help="eps of AdamW", type=float)
parser.add_argument("--weight_decay", help="weight decay of AdamW", type=float)

# path
parser.add_argument("--vocab_path", help="Vocab.json", type=str)
parser.add_argument("--merge_path", help="merge.txt", type=str)
parser.add_argument("--prompt_path", help="prompt.txt", type=str)
parser.add_argument("--ckpt_path", help="ckpt.pt", type=str)

# config
parser.add_argument("--temperature", help="temperature when generate", type=float)
parser.add_argument("--top_p", help="top-p", type=float)
parser.add_argument("--eos", help="EOS, default=0", type=int, default=0)

args = parser.parse_args()


if __name__ == "__main__":
    tok = Tokenizer.from_files(args.vocab_path, args.merge_path, ["<|endoftext|>"])

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        text = f.read()

    input_ids = torch.tensor(tok.encode(text))

    model = transformer_lm(vocab_size=args.vocab_size, d_model=args.d_model, context_length=args.context_length, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, rope_theta=args.rope_theta, device=args.device)
    optim = AdamW(model.parameters(), betas=tuple(args.betas), eps=args.eps, weight_decay=args.weight_decay)
      
    load_checkpoint(args.ckpt_path, model, optim)


    output_ids = inference(model=model, prompt=input_ids, max_new_tokens=256, temperature=args.temperature, top_p=args.temperature, eos_token_id=args.eos)


    # Default B = 1, use output_ids[0]; otherwise, using loop
    output = tok.decode(output_ids[0])

    print(output)



    
