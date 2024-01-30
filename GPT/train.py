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
import torch
data = torch.tensor(encode(text), dtype=torch.long)

# now, we split up the data into train and validation sets, where the
# first 90% is for training and the rest is for validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# now we want to batch some of the training data to help with loading
# and to parallelise the training
torch.manual_seed(5443)
batch_size = 4 # number of sequences processed in parallel
block_size = 8 # max context length for predictions

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # each token reads off the logits for the next token
        # from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets):
        # idx and targets are both (Batch, Time) tensor of integers
        logits = self.token_embedding_table(idx) # (Batch, Time, Channel)
        return logits

m = BigramLanguageModel(vocab_size)
out = m(xb,yb)
print(out.shape)