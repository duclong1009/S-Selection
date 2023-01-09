import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
import os
dict_path = f"Saved_Results/NIID_middlebias_10clients"
saved_path = f"NIID_middlebias_10clients"
session_name = f"fedalgo7_ratio_1.0_C_0.3_config2"
path_ = f"{dict_path}/{session_name}.json"

def check_dir(dict_path):
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

with open(path_, "r") as f:
    result_dict = json.load(f) 
import statistics as stat
import matplotlib.pyplot as plt
n_rows = 2
n_cols = 5

start_r = 81

for start_r in range(1,181,10):
    end_round = start_r + n_cols * n_rows - 1
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=(20,8), sharey=True)
    for i,round in enumerate(range(start_r,end_round+1)):
        row = i // n_cols
        col = i % n_cols
        my_dict = result_dict[f"Round {round}"]["score_list"]
        list_std = []
        for val in my_dict.values():
            list_std.append(stat.stdev(val))
        ax[row,col].bar(my_dict.keys(),list_std)
        # ax[row,col].set_xticklabels()
        ax[row,col].set(xlabel=f"Round {round}")
    check_dir(f"stat/Fig/{saved_path}/{session_name}/std")
    plt.savefig(f"stat/Fig/{saved_path}/{session_name}/std/Round {start_r}_{end_round}")
    plt.clf()