import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW

from tqdm import tqdm

torch.manual_seed(42)

batch_size = 64
context_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#n_embed = 32

max_new_tokens = 500


with open('Lusiadas/lusiadas.txt', 'r', encoding='ISO-8859-1') as f:
    text = f.read()
    
    
chars = sorted(list(set(text)))
vocab_size = len(chars)
str_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_str = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [str_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_str[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
split_index = int(0.9*len(data))
train_data = data[:split_index]
val_data = data[split_index:]


def get_batch(split, context_size, batch_size):
    data = train_data if split=='train' else val_data
    index = torch.randint(len(data)- context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in index])
    y = torch.stack([data[i+1:i+context_size+1] for i in index])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss(context_size, batch_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split, context_size, batch_size)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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
        

class MultiHeadedAttention(nn.Module):
    def __init__(self,context_size, n_embed, head_size, num_heads, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(context_size, n_embed, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(x))
    

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), 
            nn.ReLU(),
            nn.Linear(4 * n_embed,n_embed),
            nn.Dropout(dropout)
            )
        
    def forward(self,x):
        return self.net(x)

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
        


class BigramLanguageModel(nn.Module):
    
    def __init__(self,context_size = 256, n_embed=384, num_heads=6, num_layers=6, dropout=0.2) -> None:
        super().__init__()
        self.context_size = context_size
        
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
        position_emb = self.position_embedding_table(torch.arange(T, device=device)) #(Context, n_embed)
        
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

        
model = BigramLanguageModel(context_size = context_size, n_embed=384, num_heads=6, num_layers=6, dropout=0.2)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)


for iter in tqdm(range(max_iters)):
    
    if iter % eval_interval == 0:
        losses = estimate_loss(context_size, batch_size)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train', context_size, batch_size)
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    
idx = torch.zeros((1,1), dtype=torch.long, device=device)
response = model.generate(idx, max_new_tokens)
decoded_response = decode(response[0].tolist())
print(decoded_response)