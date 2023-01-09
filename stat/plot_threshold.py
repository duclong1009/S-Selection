import numpy as np
import pandas as pd 
import json
import matplotlib.pyplot as plt
import os
dict_path = f"/home/ace15032jo/Projects/repo2/SampleSelection_easyFL/Saved_Results/NIID_middlebias_10clients"
saved_path = f"NIID_middlebias_10clients"
session_name = f"fedalgo7_ratio_1.0_C_0.3_config2"
path_ = f"{dict_path}/{session_name}.json"

def check_dir(dict_path):
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

with open(path_, "r") as f:
    result_dict = json.load(f) 

import matplotlib.pyplot as plt

list_ = []
end_round = 57
for round in range(1,end_round):
    list_.append(result_dict[f"Round {round}"]["selected_ratio"])
plt.plot(list_)
check_dir(f"Fig/{saved_path}/{session_name}/removed_rate")
plt.savefig(f"Fig/{saved_path}/{session_name}/removed_rate/Round {round}")
plt.clf()

list_ = []
rounds = range(1,end_round)
for round in rounds:
    list_.append(list(result_dict[f"Round {round}"]["threshold_score"].values())[0])
plt.plot(rounds,list_)
check_dir(f"Fig/{saved_path}/{session_name}/threshold")
plt.savefig(f"Fig/{saved_path}/{session_name}/threshold/Round {round}")
plt.clf()
