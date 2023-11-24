import random
import time
import collections
import math
import numpy as np
import pdb

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def by_labels_non_iid_split(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)
    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    # get subset
    n_samples = int(len(dataset) * frac)
    # selected_indices = np.random.randint(0, len(dataset), n_samples)
    selected_indices = np.random.choice([i for i in range(len(dataset))], n_samples, replace=False)
    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}
    for idx in selected_indices:
        label = dataset[idx][1]
        group_id = label2cluster[label]
        
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_indices = [[] for _ in range(n_clients)]  
    clients_indices_dict = {}  
    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster
    
    for cluster_id in range(n_clusters):
            weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
            clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    
    clients_counts = np.cumsum(clients_counts, axis=1)

    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            clients_indices[client_id] += [int(i) for i in indices]       
    for client_id in range(n_clients):
            clients_indices_dict[client_id] = clients_indices[client_id]
    return clients_indices, clients_indices_dict
import tqdm.tqdm as tqdm
import pandas as pd 
def sta(client_dict, train_dataset, num_client=20, num_label=10):
    rs = []
    for client in tqdm(range(num_client)):
        tmp = []
        for i in range(num_label):
            tmp.append(sum(train_dataset[j][1] == i for j in client_dict[client]))
        rs.append(tmp)
    df = pd.DataFrame(rs,columns=[f"Label_{i}" for i in range(num_label)])
    return df


import torchvision
dataset = torchvision.datasets.CIFAR100('./cifar100', train=True, download=True)
n_classes = 100
n_clients = 100
n_clusters = 100
alpha = 0.1
frac = 1
data_idx, clients_indices_dict = by_labels_non_iid_split(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234)

df = sta(clients_indices_dict, dataset, n_clients, n_classes)
df.to_csv(f"cifar100/data_idx/dirichlet_{alpha}_stat.csv")
saved_folder = ""
import json
list_n_samples = []
for i in range(n_clients):
    list_n_samples.append(len(clients_indices_dict[i]))
print(list_n_samples)
with open(f"cifar100/data_idx/dirichlet_{alpha}.json", "w") as f:
    json.dump(clients_indices_dict, f)
breakpoint()