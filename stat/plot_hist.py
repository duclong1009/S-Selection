import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
import os
dict_path = f"Saved_Results/gauss_CIFAR10_pareto_10clients_10class"
saved_path = f"gauss_CIFAR10_pareto_10clients_10class"
session_name = f"fedalgo7_ratio_1.0_C_0.3_config2_inverse"
path_ = f"{dict_path}/{session_name}.json"

def check_dir(dict_path):
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

with open(path_, "r") as f:
    result_dict = json.load(f) 

import matplotlib.pyplot as plt
n_rows = 2
n_cols = 3

round = 25
for round in range(1,150):
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=(20,4), sharex=True,sharey=True)
    for i,client in enumerate(result_dict[f"Round {round}"]["selectd_client"]):
        row = i // n_cols
        col = i % n_cols
        list_ = result_dict[f"Round {round}"]["score_list"][f"client_{client}"]
        ax[0,col].hist(list_,bins=1000)
        # ax[row,col].set_xticklabels(my_dict.keys())
        ax[0,col].set(xlabel=f"Client {client}")

    for i,client in enumerate(result_dict[f"Round {round+1}"]["selectd_client"]):
        row = i // n_cols
        col = i % n_cols
        list_ = result_dict[f"Round {round+1}"]["score_list"][f"client_{client}"]
        ax[1,col].hist(list_,bins=1000)
        # ax[row,col].set_xticklabels(my_dict.keys())
        ax[1,col].set(xlabel=f"Client {client}")

    check_dir(f"stat/Fig/{saved_path}/{session_name}/histogram")
    plt.savefig(f"stat/Fig/{saved_path}/{session_name}/histogram/Round {round}")
    plt.clf()