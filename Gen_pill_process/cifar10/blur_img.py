
from PIL import Image, ImageFilter
import numpy as np
import json
import copy
import os
#### Config
data_path = "Dataset_scenarios/cifar10/CIFAR10_pareto_10clients_10class.json"
blurry_radius = 7
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
# breakpoint()
with open("cifar10/origin/X_train.npy","rb") as f:
    x_raw_data = np.load(f)
copy_x = copy.deepcopy(x_raw_data)
# breakpoint()
with open("cifar10/origin/Y_train.npy","rb") as f:
    y_raw_data = np.load(f)


with open(data_path, "r") as f:
    data_idx = json.load(f)
blurry_rate = [0.3,0.3,0.3,0.25,0.25,0.25,0.2,0.2,0.2,0.1,0.1,0.1,0,0,0,0,0,0,0,0]
config_dict = {}
config_dict["blurry_rate"] = [float(i) for i in blurry_rate]
config_dict["blurry_id"] = {}
n_clients = len(data_idx)
for client in range(n_clients):
    idx_ = data_idx[str(client)]
    n_samples = len(idx_)
    n_blur_imgs = int(blurry_rate[client] * n_samples)
    selected_samples = np.random.choice(idx_,n_blur_imgs,replace=False)
    config_dict["blurry_id"][client] = [int(i) for i in selected_samples]

    for idx in  selected_samples:
        image = Image.fromarray(np.array(copy_x[idx]))
        filtered = image.filter(ImageFilter.GaussianBlur(radius=blurry_radius))
        copy_x[idx] = np.array(filtered)


saved_path = f"cifar10/blurry_{folder_name}"
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
# breakpoint()
with open(f"{saved_path}/X_train.npy","wb") as f:
    np.save(f,copy_x)

with open(f"{saved_path}/Y_train.npy","wb") as f:
    np.save(f,y_raw_data)

with open(f"{saved_path}/config.json","w") as f:
    json.dump(config_dict,f)

with open(f"{saved_path}/data_idx.json","w") as f:
    json.dump(data_idx,f)