import torch

batch_size = 2

preds = torch.tensor([[7,2,3,4],
					  [1,2,3,4]], dtype=torch.float32)

labels = torch.tensor([[1,0,0,0],
					  [0,1,0,0]], dtype=torch.long)
				  

l = -torch.mean(torch.sum(labels * torch.log(preds), dim=1))
print(labels * torch.log(preds))
l = -torch.mean(torch.sum(labels.view(batch_size, -1) * torch.log(preds.view(batch_size, -1)), dim=1))

print(l)
