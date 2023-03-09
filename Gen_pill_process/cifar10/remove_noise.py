import numpy as np
import json
import numpy as np
import os

# remove_rate = [0.1,0.3,0.5,0.7]
remove_rate = [1,0.5,0.1,0]
saved_folder_path = f"cifar10_combine_noise/config6"

config_new ={}
config_new["removed_rate"] = remove_rate
with open(f"cifar10_combine_noise/blurry_cifar10_iid_100client_1000data_combine1/config.json","r") as f:
    config = json.load(f)

with open(f"cifar10_combine_noise/blurry_cifar10_iid_100client_1000data_combine1/data_idx.json","r") as f:
    data_idx = json.load(f)

import shutil

blurry_id_degree = config["blurry_id_degree"]
new_dict = {}
for client in data_idx.keys():
    config_new["removed_id"] = {}
    removed_samples_client = []
    data_idx_client = data_idx[client]
    for i, noise_degree in enumerate(blurry_id_degree[client].keys()):
        noise_idx = [idx for idx in data_idx[client] if idx in blurry_id_degree[client][noise_degree]]
        n_removed_sample = int(len(noise_idx) * remove_rate[i])
        removed_samples = np.random.choice(noise_idx,n_removed_sample,replace=False)
        removed_samples_client += list(removed_samples)
    
    config_new["removed_id"][client] = [int(t) for t in removed_samples_client]
    new_dict[client] = [idx for idx in data_idx_client if idx not in removed_samples_client]


if not os.path.exists(saved_folder_path):
    os.makedirs(saved_folder_path)

with open(f"{saved_folder_path}/data_idx.json","w") as f:
    json.dump(new_dict,f)

with open(f"{saved_folder_path}/config.json","w") as f:
    json.dump(config_new,f)


shutil.copy("cifar10_combine_noise/blurry_cifar10_iid_100client_1000data_combine1/X_train.npy",f"{saved_folder_path}/X_train.npy")
shutil.copy("cifar10_combine_noise/blurry_cifar10_iid_100client_1000data_combine1/Y_train.npy",f"{saved_folder_path}/Y_train.npy")
