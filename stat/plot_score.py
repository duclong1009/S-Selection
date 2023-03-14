import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
import os


dict_path = f"Saved_Results/gauss_cifar10_iid_100client_1000data"
saved_path = f"gauss_cifar10_iid_100client_1000data"
session_name = f"fedalgo6base_ratio_0.9_C_0.3_"
path_ = f"{dict_path}/{session_name}.json"

def check_dir(dict_path):
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

with open(path_, "r") as f:
    result_dict = json.load(f) 
with open("cifar10/gauss_cifar10_iid_100client_1000data/data_idx.json","r") as f:
    data_idx = json.load(f)

with open("cifar10/gauss_cifar10_iid_100client_1000data/config.json","r") as f:
    data_config = json.load(f)

import matplotlib.pyplot as plt
n_rows = 5
n_cols = 6

# blue : sach va k bi xoa 
# green : noise va bi xoa
# red: sach va bi xoa
# black: noise va k bi xoa
round = 25
for round in range(1,150):
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=(20,4), sharex=True,sharey=True)
    for i,client in enumerate(result_dict[f"Round {round}"]["selectd_client"]):
        list_idx = data_idx[str(client)]
        list_noise = data_config["blurry_id"][str(client)]
        row = i // n_cols
        col = i % n_cols
        list_ = result_dict[f"Round {round}"]["score_list"][f"client_{client}"]

        list_0 = []
        range_idx_0 = []
        list_1 = []
        range_idx_1 = []
        list_2 = []
        range_idx_2 = []
        list_3 = []
        range_idx_3 = []
        thresh_score = result_dict[f"Round {round}"]["threshold_score"][f"client_{client}"]
        for st, sample_score in enumerate(list_):
            idx = list_idx[st]
            if idx not in list_noise:
                if list_[st] < thresh_score:
                    list_2.append(sample_score)
                    range_idx_2.append(st)
                else:
                    list_0.append(sample_score)
                    range_idx_0.append(st)
            else:
                if list_[st] < thresh_score:
                    list_1.append(sample_score)
                    range_idx_1.append(st)
                else:
                    list_3.append(sample_score)
                    range_idx_3.append(st)
        ax[row,col].bar(range_idx_0,list_0,color="blue")
        ax[row,col].bar(range_idx_1,list_1,color="green")
        ax[row,col].bar(range_idx_2,list_2,color="red")
        ax[row,col].bar(range_idx_3,list_3,color="black")
        ax[row,col].set(xlabel=f"{client}")
    check_dir(f"stat/Fig/{saved_path}/{session_name}/score")
    plt.savefig(f"stat/Fig/{saved_path}/{session_name}/score/Round {round}")
    plt.clf()