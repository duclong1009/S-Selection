from torchvision import datasets
from torchvision import datasets, transforms
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def save_dataset_idx(list_idx_sample, path, filename="dataset_idx.json"):
    if not Path(path).exists():
        os.system(f"mkdir -p {path}")
        
    with open(path + filename, "w+") as outfile:
        json.dump(list_idx_sample, outfile)
        

def sta(client_dict, train_dataset, num_client=10, num_class=10):
    rs = []
    for client in range(num_client):
        tmp = []
        for i in range(num_class):
            tmp.append(sum(train_dataset[j][1] == i for j in client_dict[client]))
        rs.append(tmp)
    df = pd.DataFrame(rs,columns=[f"Label_{i}" for i in range(num_class)])
    return df


def iid(dataset, total_client):
    client_id = [i for i in range(total_client)]
    client_dict = {}
    
    total_data = len(dataset)
    # client_ndata = int(total_data/total_client)
    client_ndata = int(total_data/total_client)
    
    sample_idx = [i for i in range(total_data)]
    
    for id in client_id:
        client_dict[id] = np.random.choice(sample_idx, client_ndata, False).tolist()
        sample_idx = list(set(sample_idx) - set(client_dict[id]))
    
    return client_dict


if __name__ == '__main__':
    train_dataset = datasets.CIFAR10( "./data/cifar/", train=True, download=True, transform=None)
    print("total sample of dataset", len(train_dataset))
    total_clients = 50
    client_dict = iid(train_dataset, total_client=total_clients)
    
    print("Gen done!")
    save_dataset_idx(client_dict, f"cifar10/", f"cifar10_iid_{total_clients}client_1000data.json")
    df = sta(client_dict, train_dataset, num_client=total_clients, num_class=10)
    print("Total sample in modified dataset", df.values.sum())
    df.to_csv(f"cifar10/cifar10_iid_{total_clients}client_1000data.csv", index=False, header=False)