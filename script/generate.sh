uv run generate.py \
  --device cuda \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --d_ff 1344 \
  --rope_theta 10000 \
  --num_layers 4 \
  --num_heads 16 \
  --betas 0.9 0.999 \
  --weight_decay 0.01 \
  --eps 1e-8 \
  --vocab_path /data/yaxuanli/ltu/cache/assignment1-basics/cs336_basics/TinyStoriesV2-GPT4-train-vocab.json\
  --merge_path /data/yaxuanli/ltu/cache/assignment1-basics/cs336_basics/TinyStoriesV2-GPT4-train-merges.txt\
  --prompt_path /data/yaxuanli/ltu/cache/assignment1-basics/script/prompt.txt\
  --ckpt_path /data/yaxuanli/ltu/cache/assignment1-basics/script/64_final.pt\
  --temperature 0.9 \
  --top_p 0.9 \
  


