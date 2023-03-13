
from PIL import Image, ImageFilter
import numpy as np
import json
import copy
import os
#### Config
data_path = "Dataset_scenarios/cifar10/100client/CIFAR-noniid-fedavg_unequal_1.json"

def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)
folder_name = data_path.split("/")[-1].split(".")[0]
# 
with open("cifar10/origin/X_train.npy","rb") as f:
    x_raw_data = np.load(f)
copy_x = copy.deepcopy(x_raw_data)
# 
with open("cifar10/origin/Y_train.npy","rb") as f:
    y_raw_data = np.load(f)


with open(data_path, "r") as f:
    data_idx = json.load(f)

import math
noise_degree_rate = [0.25, 0.25, 0.25, 0.25]
noise_degree_list = [1,2,3,4]
config_dict = {}
n_clients = len(data_idx)
blurry_rate = [0.5] * n_clients
config_dict["blurry_rate"] = [float(i) for i in blurry_rate]
config_dict["blurry_id_degree"] = {}

for client in range(n_clients):
    idx_ = data_idx[str(client)]
    n_samples = len(idx_)
    n_blur_imgs = int(blurry_rate[client] * n_samples)
    n_noise_rate = [math.floor(noise_degree_rate[0] * n_blur_imgs), len(noise_degree_rate)]
    selected_samples = np.random.choice(idx_,n_noise_rate,replace=False)
    config_dict["blurry_id_degree"][client] = {}
    for i,noise_d in enumerate(noise_degree_list):
        selected_idx = selected_samples[:,i]
        config_dict["blurry_id_degree"][client][i] = [int(i) for i in selected_idx]
        for idx in  selected_idx:
            image = Image.fromarray(np.array(copy_x[idx]))
            filtered = image.filter(ImageFilter.GaussianBlur(radius=i))
            copy_x[idx] = np.array(filtered)

saved_path = f"cifar10_unequal/blurry_{folder_name}_combine1"
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
# 
with open(f"{saved_path}/X_train.npy","wb") as f:
    np.save(f,copy_x)

with open(f"{saved_path}/Y_train.npy","wb") as f:
    np.save(f,y_raw_data)

with open(f"{saved_path}/config.json","w") as f:
    json.dump(config_dict,f)

with open(f"{saved_path}/data_idx.json","w") as f:
    json.dump(data_idx,f)