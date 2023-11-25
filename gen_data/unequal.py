import os
import json
import numpy as np


def noniid_unequal(labels, num_users, idxs=None):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # breakpoint()
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards

    num_shards, num_imgs = 352, 30 
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    # idxs = np.arange(num_shards*num_imgs)
    if idxs is None:
        idxs = np.arange(len(labels))
    # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 4

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    for i in range(num_users):
        dict_users[i] = [int(j) for j in dict_users[i]]

    return dict_users

def save_dataset_idx(list_idx_sample,path="dataset_idx.json"):
    with open(path, "w+") as outfile:
        json.dump(list_idx_sample, outfile)

import pandas as pd
def sta(client_dict,labels):
    rs = []
    for client in range(100):
        tmp = []
        for i in range(150):
            tmp.append(sum(labels[j] == i for j in client_dict[client]))
        rs.append(tmp)
    df = pd.DataFrame(rs,columns=[f"Label_{i}" for i in range(150)])
    return df     

from torchvision import datasets, transforms  
import json
if __name__ == '__main__':
    # apply_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))])
    path = "pill_dataset/medium_pilldataset"
    with open(f"pill_dataset/medium_pilldataset/image_categories_dict.json", "r") as f:
        image_categories = json.load(f)
    idxs = []
    for key in image_categories.keys():
        if isinstance(image_categories[key], dict):
            # breakpoint()
            for k in image_categories[key].keys():
                idxs += image_categories[key][k]

    print(f"Len dataset {len(idxs)}")
    num_clients = 200
    client_dict = noniid_unequal(image_categories['labels'],num_clients,idxs)
    index = 3
    import os 
    saved_dict_path = f"{path}/{num_clients}client/unequal"
    if not os.path.exists(f"{saved_dict_path}"):
        os.makedirs(f"{saved_dict_path}")
    save_dataset_idx(client_dict, f"{saved_dict_path}/data_scenario_{index}.json")
    df = sta(client_dict,image_categories['labels'])
    df.to_csv(f"{saved_dict_path}/data_scenario_{index}.csv")