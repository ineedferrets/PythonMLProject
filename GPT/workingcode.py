import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C = 4, 8, 32 # batch, time, channels
x = torch.randn(B,T,C)

# we are now going to find the mean value for the previous tokens
# and the token at time t in each batch
# this is a weak form of aggregation of the information of the
# history behind a token
xbow = torch.zeros((B,T,C))
for b in range (B):
    for t in range (T):
        xprev = x[b,:t+1] # (t, C)
        xbow[b,t] = torch.mean(xprev, 0)

# instead of the inefficient double nested for loop, we can use
# matrix manipulation of a diagonal matrix of ones to summate
# over time and then divide to get the average
wei = torch.tril(torch.ones(T,T))
wei = wei / wei.sum(1, keepdim=True)
# seeing that the second dimension matches the final dimension
# of the first variable, Python (or PyTorch maybe?) assumes the
# multiplication is happening to each batch
xbow2 = wei @ x # (B,T,T) @ (B,T,C) ===> (B, T, C)

# another way of doing this is via softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1) # normalises 0's and -inf's over each row
out = wei @ x

# let's see a single Head perform self attention
head_size = 16

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)      # (B, T, 16)
q = query(x)    # (B, T, 16)
wei = q @ k.transpose(-2, -1)   # (B, T, 16) @ (B, 16, T) ====> (B, T, T)

wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1) # normalises 0's and -inf's over each row

v = value(x)
out = wei @ v
print(out.shape)
print(wei)