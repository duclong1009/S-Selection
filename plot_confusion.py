from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
# path = "Saved_Results/gauss_cifar10_iid_100client_1000data_2/fedalgo6base_ratio_0.8_C_0.3__wandb.json"
path = "Saved_Results/gauss_cifar10_iid_100client_1000data_2/fedavg_log_ratio_1.0_C_0.3_no_score_wandb.json"
with open(path,"r") as f:
    result = json.load(f)
# 
round = 250
conf = result[f"Round {round}"]["Confusion_matrix"]
display_labels = np.array(range(10))
disp = ConfusionMatrixDisplay(confusion_matrix=np.array(conf),display_labels=display_labels)
disp.plot()
plt.plot()
plt.savefig("Conf_fedavg.png")