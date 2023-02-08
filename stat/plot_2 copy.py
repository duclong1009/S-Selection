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
client_id = 2
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
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=(20,4), sharex=True,sharey=True)
for round in range(1,50):
    if client_id in result_dict[f"Round {round}"]["selectd_client"]:
        list_ = result_dict[f"Round {round}"]["list_conf"][f"client_{client_id}"]
        list_score.append(list_)
        list_round.append(round)
a = np.array(list_score)
delta = a[1:,:] - a[:-1,:]
for i,id in enumerate(list_idx):
    exten = ""
    color = "blue"
    if id in list_noise:
        color = "black"
        exten = "noise"
    plt.plot(range(delta.shape[0]), delta[:,i],color=color)
    # plt.bar(list_round, delta[:,i],color=color)
    check_dir(f"stat/Fig/{saved_path}/{session_name}/convergence/client_{client_id}")
    plt.savefig(f"stat/Fig/{saved_path}/{session_name}/convergence/client_{client_id}/ID {id}_{exten}")
    plt.ylim((-1,1))
    plt.clf()
# breakpoint()
        # ax[row,col].bar(range_idx_0,list_0,color="blue")
        # ax[row,col].bar(range_idx_1,list_1,color="green")
        # ax[row,col].bar(range_idx_2,list_2,color="red")
        # ax[row,col].bar(range_idx_3,list_3,color="black")
        # ax[row,col].set(xlabel=f"{client}")
    # check_dir(f"stat/Fig/{saved_path}/{session_name}/score")
    # plt.savefig(f"stat/Fig/{saved_path}/{session_name}/score/Round {round}")
    # plt.clf()