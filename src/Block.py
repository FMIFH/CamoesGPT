import torch.nn as nn

from FeedForward import FeedForward
from MultiHeadedAttenttion import MultiHeadedAttention


class Block(nn.Module):
    def __init__(self,context_size, n_embed, num_heads, dropout) -> None:
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadedAttention(context_size, n_embed, head_size, num_heads, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.layer_norm_attention = nn.LayerNorm(n_embed)
        self.layer_norm_ffw = nn.LayerNorm(n_embed)
        
    
    def forward(self, x):
        x = x + self.sa(self.layer_norm_attention(x))
        x = x + self.ffwd(self.layer_norm_ffw(x))
        return x
        

