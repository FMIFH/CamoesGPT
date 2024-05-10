import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from CamoesGPT import CamoesGPT


from tqdm import tqdm

torch.manual_seed(42)


batch_size = 64
context_size = 256

max_iters = 5000
eval_interval = 500
learning_rate = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embed = 384
num_heads = 6
num_layers= 6
dropout=0.2

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

        
model = CamoesGPT(vocab_size, context_size, n_embed, num_heads, num_layers, dropout)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

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