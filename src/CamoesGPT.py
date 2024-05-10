import torch
import torch.nn as nn
from torch.nn import functional as F

from Block import Block

class CamoesGPT(nn.Module):
    
    def __init__(self,vocab_size, context_size = 256, n_embed=384, num_heads=6, num_layers=6, dropout=0.2, device='cpu') -> None:
        super().__init__()
        
        self.context_size = context_size
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        
        #self.sa_head = AttentionHead(n_embed,n_embed)
        #self.sa_heads = MultiHeadedAttention(n_embed, n_embed//4 , 4)
        #self.ffwd = FeedForward(n_embed)
        
        self.blocks = nn.Sequential(
            *[Block(context_size, n_embed, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
        
    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_emb = self.token_embedding_table(idx) #(Batch Size, Context, n_embed)
        position_emb = self.position_embedding_table(torch.arange(T, device=self.device)) #(Context, n_embed)
        
        x = token_emb + position_emb #(Batch Size, Context, n_embed)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(Batch Size, Context, Vocab Size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets= targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.context_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
