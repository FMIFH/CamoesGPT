import torch
import torch.nn as nn
from torch.nn import functional as F

from Attention import AttentionHead

class MultiHeadedAttention(nn.Module):
    def __init__(self,context_size, n_embed, head_size, num_heads, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(context_size, n_embed, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(x))
    