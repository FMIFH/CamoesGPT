import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from CamoesGPT import CamoesGPT
import os
from matplotlib import pyplot as plt

print(os.getcwd())

from tqdm import tqdm

torch.manual_seed(42)


batch_size = 64
context_size = 256

max_iters = 15000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 100

n_embed = 128
num_heads = 8
num_layers= 16
dropout=0.3

max_new_tokens = 1000


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

        
model = CamoesGPT(vocab_size, context_size, n_embed, num_heads, num_layers, dropout, device=device)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

epoch_losses = {
    'epoch' : range(0,max_iters+1,eval_iters),
    'training' : [],
    'valid' : []

}
for iter in tqdm(range(max_iters+1)):
    
    if iter % eval_interval == 0:
        losses = estimate_loss(context_size, batch_size)
        print(f"\nstep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        torch.save(model, f"{os.getcwd()}/models/bigger_model2_{iter}_{int(losses['val']*1000)}.pt")
        epoch_losses['training'].append(losses['train'])
        epoch_losses['valid'].append(losses['val'])

    
    xb, yb = get_batch('train', context_size, batch_size)
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

plt.plot(epoch_losses['epoch'], epoch_losses['training'], label='train_loss')
plt.plot(epoch_losses['epoch'], epoch_losses['valid'], label='val_loss')
plt.legend()
plt.savefig(f"{os.getcwd()}/plots/bigger_model2_training.png") 
#plt.show()
    
idx = torch.zeros((1,1), dtype=torch.long, device=device)
response = model.generate(idx, max_new_tokens)
decoded_response = decode(response[0].tolist())
print(decoded_response)