import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
import torch

from cs336_basics.linear import Linear
from cs336_basics.rms import RMSnorm
from cs336_basics.FFN import SwiGLU
from cs336_basics.multiheadattn_test import CausalMultiHeadSelfAttention
from cs336_basics.embedding import Embedding
from cs336_basics.softmax import Softmax

    
class transformer_block(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 device: str | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()

        kwargs = {'device': device, 'dtype': dtype}

        self.theta = theta
        self.max_seq_len = max_seq_len

        self.use_rope = False # use rope when theta provided

        if theta:
            self.use_rope = True

        self.attn = CausalMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len, use_rope=self.use_rope, )
        self.ln1 = RMSnorm(d_model=d_model, **kwargs)
        self.ln2 = RMSnorm(d_model=d_model, **kwargs)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, **kwargs)        


    def forward(self, 
                in_features: Float[Tensor, " batch sequence_length d_model"],
                token_positions: Float[Tensor, "... seq_length"] | None = None):

        if self.theta is not None and self.max_seq_len is not None:
            token_positions = torch.arange(0, in_features.size(1)) # Make token_position 

        # First Block
        pre_in_features_1 = in_features.clone()
        in_features = self.attn(self.ln1(in_features), token_positions=token_positions)
        in_features += pre_in_features_1


        # Second Block
        pre_in_features_2 = in_features.clone()
        in_features = self.ffn(self.ln2(in_features))
        in_features += pre_in_features_2

        return in_features
        # Position-wise feedforward

class transformer_lm(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 context_length: int, 
                 num_layers: int, 
                 num_heads: int, 
                 d_ff: int,
                 rope_theta: float,
                 device: str | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()

        kwargs = {'device': device, 'dtype': dtype}

        self.token_embedding = Embedding(num_embeddings=vocab_size, embeddings_dim=d_model, **kwargs) # Did one hot for me 

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_length, **kwargs))

        self.ln_final = RMSnorm(d_model=d_model, **kwargs)
        self.lm_head = Linear(d_model, vocab_size, **kwargs)
        self.softmax = Softmax(dim=-1)

    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]):
        x = self.token_embedding(in_indices) # [B, T] -> (one hot -> [B, T, vocab_size] @ [vocab_size, d_model]) = [B, T, d_model]

        for layer in self.layers:
            x = layer(x) # [B, T, d_model] -> Transformer layer = [B, T. d_model]

        x = self.ln_final(x) # [B, T, d_model] -> [B, T, d_model]
  
        x = self.lm_head(x) # [B, T, d_model] -> [B, T, vocab_size]

        # Noeed need for softmax here
        #x = self.softmax(x) # [B, T, vocab_size] -> [B, T, vocab_size] (prob) 

        return x