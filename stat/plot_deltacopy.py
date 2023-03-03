import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
import os


dict_path = f"Saved_Results/gauss_cifar10_iid_100client_1000data_2"
saved_path = f"gauss_cifar10_iid_100client_1000data_2"
session_name = f"fedavg_log_ratio_1.0_C_0.3_no_score"
path_ = f"{dict_path}/{session_name}.json"

def check_dir(dict_path):
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

with open(path_, "r") as f:
    result_dict = json.load(f) 
with open("cifar10/gauss_cifar10_iid_100client_1000data_2/data_idx.json","r") as f:
    data_idx = json.load(f)

with open("cifar10/gauss_cifar10_iid_100client_1000data_2/config.json","r") as f:
    data_config = json.load(f)
print(result_dict.keys())
import matplotlib.pyplot as plt
client_id = 0
# blue : sach va k bi xoa 
# green : noise va bi xoa
# red: sach va bi xoa
# black: noise va k bi xoa
list_score = []
list_idx = data_idx[str(client_id)]
list_noise = data_config["blurry_id"][str(client_id)]
list_round = []
n_rows = 6
n_cols=4
list_noise_odx = []
for round in range(1,40):
    if client_id in result_dict[f"Round {round}"]["selectd_client"]:
        # row = i // n_cols
        # col = i % n_cols
        list_ = result_dict[f"Round {round}"]["list_conf"][f"client_{client_id}"]
        list_score.append(list_)
        list_round.append(round)
# 
a = np.array(list_score)

delta = a[1:,:] - a[:-1,:]
delta= np.mean(np.exp(100 * delta),0)

list_d_ = []
list_c_ = []
for i,id in enumerate(list_idx):
    if id in list_noise:
        if a[-1,i] < 0.4:
            list_d_.append(i)
    else:
        if a[-1,i] < 0.4:
            list_c_.append(i)
list_ = list_d_ + list_c_
# 
print(f"Clean img {len(list_c_)}/{len(list_noise)}  Dirty img {len(list_d_)}/{a.shape[1] -len(list_noise)}")
sorted_idx = np.argsort(a[-1,:])
list_idx_less = np.where(a[-1,:] < 0.4)
sorted_idx_by_delta = np.argsort(delta[list_])
list_d = []
list_c = []
# 
for i,id in enumerate(list(sorted_idx_by_delta[:50])):
    # if list_[id] in list_idx_less[0]:
        if list_[id] in list_noise:
                list_d.append(i)
        else:
        
                list_c.append(i)
print(f"Sorted Clean img {len(list_c)}/{len(list_c_)}  Dirty img {len(list_d)}/{a.shape[1] -len(list_d_)}")

# me = np.mean(a,0)
# delta = a[1:,:] - a[:-1,:]
# 
# np.mean(np.exp(10 * delta[:,4]))
# # plt.bar(range(a.shape[1]),me)
# for i,id in enumerate(list_idx):
#     if id in list_noise:
#         list_noise_odx.append(i)
# # 
# list_clean = [i for i in range(a.shape[1]) if i not in list_noise_odx]

