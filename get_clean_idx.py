from PIL import Image, ImageFilter
import numpy as np
import json
import copy
import os
#### Config
data_path = "Dataset_scenarios/cifar10/cifar10_iid_100client_1000data.json"
blurry_radius = 4
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

with open(data_path, "r") as f:
    data_idx = json.load(f)

with open(f"cifar10/blurry_0.3_100_1/config.json", "r") as f:
    noisy_idx = json.load(f)["blurry_id"]

clean_idx = {}
for k in data_idx.keys():
    tem = [i for i in data_idx[k] if i not in noisy_idx[k]]
    clean_idx[k] = tem 

with open("data_idx.json","w") as f:
    json.dump(clean_idx,f)