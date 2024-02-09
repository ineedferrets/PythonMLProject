import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32     # the maximum context length for predictions
block_size = 8      # how many independent sequences processed in parallel
max_iters = 8000
eval_interval = 400
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32         # number of embedding dimensions
# ----------------------

torch.manual_seed(2543)

# import the input text for training
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# generate a list of all the unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# map each character to an integer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string and then output a list of integers
encode = lambda s: [stoi[c] for c in s]
# decoder: take a list of integers and then output a string
decode = lambda l: ''.join([itos[i] for i in l])

# now we encode the entire text dataset and stor into into a torch Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# now, we split up the data into train and validation sets, where the
# first 90% is for training and the rest is for validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# now we want to batch some of the training data to help with loading
# and to parallelise the training
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token reads off the logits for the next token
        # from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (Batch, Time) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channel)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # reshape matrices to conform with PyTorch cross entropy
            # input expectations
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every few iterations, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()
    
# generate from the model 
context = torch.zeros((1,1), dtype=torch.long, device=device)   
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))