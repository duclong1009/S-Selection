import json

with open(f"cifar100/data_idx/dirichlet_0.1.json", "r") as f:
    data_idx = json.load(f)
    
for i in data_idx.keys():
    print(f"Client {i} has {len(data_idx[i])}")