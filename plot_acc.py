# import matplotlib.pyplot as plt
# import json
# import numpy as np
# import os
# start_round = 80
# n_rounds = 123
# path = "Saved_Results/NIID_10client_bias_noisy_rerun"
# session = [i for i in os.listdir(path) if "wandb" in i]
# session = ["fedavg_ratio_1.0_C_0.3__wandb.json","fedalgo6_dev_ratio_0.95_C_0.3__wandb.json","fedalgo1_ratio_0.8_C_0.3__wandb.json"]
# label_list = ["Fedavg", "Fedalgo6","Fedalgo1"]
# for i,s in enumerate(session):
#     with open(f"{path}/{s}") as f:
#         log_dict = json.load(f)
#         # n_rounds = len(log_dict)
#         list_acc = []
#         for round in range(1,n_rounds+1):
#             list_acc.append(log_dict[f"Round {round}"]["Accuracy/Testing Accuracy"])
#         # 
#         plt.plot(range(start_round,n_rounds), np.array(list_acc)[start_round:n_rounds], label=f"{label_list[i]}")
#         # 
#         print(s)
# plt.legend()
# plt.xlabel("Communication Round")
# plt.ylabel("Accuracy")
# plt.savefig(f"{path}/Accuracy")


import matplotlib.pyplot as plt
import json
import numpy as np
import os
start_round = 80
# n_rounds = 123
path = "Saved_Results/NIID_10client_Gr_rerun"
session = [i for i in os.listdir(path) if "wandb" in i]
# session = ["fedprox_ratio_1.0_C_0.5__wandb.json","fedalgo6_dev_fedprox_ratio_0.8_C_0.5__wandb.json","fedalgo1_fedprox_ratio_0.8_C_0.5__wandb.json"]
label_list = ["FedProx", "Fedalgo6","Fedalgo1"]
for i,s in enumerate(session):
    with open(f"{path}/{s}") as f:
        log_dict = json.load(f)
        n_rounds = len(log_dict)
        list_acc = []
        for round in range(1,n_rounds+1):
            list_acc.append(log_dict[f"Round {round}"]["Accuracy/Testing Accuracy"])
        # 
        plt.plot(range(1,n_rounds+1), np.array(list_acc), label=f"{label_list[i]}")
        # 
        print(s)
plt.legend()
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.savefig(f"{path}/Accuracy")