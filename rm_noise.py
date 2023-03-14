import json
path_ = "cifar10/gauss_cifar10_iid_100client_1000data"
with open(f"{path_}/data_idx.json","r") as f:
    data_idx = json.load(f)

with open(f"{path_}/config.json","r") as f:
    data_config = json.load(f)

n_clients = len(data_idx)
clean_dict = {}
for client in data_idx.keys():
    clean_list = [i for i in data_idx[client] if i not in data_config["blurry_id"][client]]
    clean_dict[client] = clean_list
    # 

with open(f"{path_}/clean_idx.json","w") as f:
    json.dump(clean_dict,f)
