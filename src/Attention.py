import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):
    def __init__(self,context_size, n_embed, head_size, dropout) -> None:
        super().__init__()
        self.key    = nn.Linear(n_embed, head_size, bias=False)
        self.query  = nn.Linear(n_embed, head_size, bias=False)
        self.value  = nn.Linear(n_embed, head_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('trill', torch.tril(torch.ones(context_size,context_size)))
        
    def forward(self,x):
        B,T,C = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        attention = Q @ K.transpose(-2,-1) * C **-0.5
        attention = attention.masked_fill(self.trill[:T,:T] == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        return attention @ V