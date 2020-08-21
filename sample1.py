import torch
from torch.distributions import Categorical

a = torch.tensor([[0.1,0.2,0.3],[0.1,0.2,0.3]])
print(a.max(dim=1))


m = Categorical(a)
action = m.sample()
print(action)