import numpy as np
import torch
import importlib
# from 
epochs = 3
saved_path =  f"centrailize/Without_rm_eval_loginterval_3"
with open(f"{saved_path}/conf_arr_{epochs}.npy","rb") as f:
    arr = np.load(f)

with open(f"{saved_path}/result_{epochs}.json","r") as f:
    import json
    result =json.load(f)

list_score = result["list_score"]
list_idx = result["list_idx"]
# 

with open(f"{saved_path}/cluster_id_{epochs}.json","r") as f:
        import json
        cluster_idx = json.load(f)


with open("cifar10/gauss_cifar10_iid_100client_1000data_2/data_idx.json","r") as f:
        import json
        data_idx = json.load(f)

with open("cifar10/gauss_cifar10_iid_100client_1000data_2/config.json","r") as f:
        import json
        data_config = json.load(f)
        list_noise_idx = data_config["blurry_id"]
total_list_noise = []
for i in list_noise_idx.keys():
      total_list_noise += list_noise_idx[i]
# 
single_set = {"0":[]}
for key in data_idx.keys():
    single_set["0"] += data_idx[key]
idx_noise = [i for i,k in enumerate(single_set["0"]) if k in total_list_noise]
# 
list_n_sams = []
list_n_nois = []
for cluster_id in cluster_idx.keys():
      n_sam = len(cluster_idx[cluster_id])
      n_noi = len([i for i in idx_noise if i in cluster_idx[cluster_id]])

      list_n_sams.append(n_sam)
      list_n_nois.append(n_noi)
import matplotlib.pyplot as plt
plt.bar(range(10),list_n_sams)
plt.bar(range(10),list_n_nois)
ten = torch.tensor(arr)
sf = torch.nn.Softmax(-1)(ten)
conf_val,max_index = torch.max(sf,-1)
com_arr = np.array([list_score,conf_val.tolist()])
# 
com_arr = com_arr.T
# 
#plotting the results:
import matplotlib.pyplot as plt 

# plt.scatter(com_arr[idx_noise, 0] , com_arr[idx_noise, 1])
# # plt.legend()
# plt.show()
plt.savefig(f"{saved_path}/bar_noise_cluster_{epochs}.png")
plt.cla()

