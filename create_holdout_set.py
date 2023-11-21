import numpy as np
import json

data_idx_path = "pill_dataset/medium_pilldataset/100client/dirichlet/mixed_noise_data_idx_alpha_0.1_cluster_20.json"
with open(data_idx_path) as f:
    data_idx = json.load(f)
holdout_idx = {}
size_of_holdout = 0.3
n_clients = len(data_idx)
for client_id in range(n_clients):
    # client_id = str(client_id)
    client_train_data = data_idx[str(client_id)]
    n_samples = len(client_train_data)
    seleted_holdout = np.random.choice(client_train_data, int(size_of_holdout * n_samples), replace=False)
    holdout_idx[client_id] = [int(i) for i in seleted_holdout]
    if len(seleted_holdout) == 0 :
        print(client_id)
with open(f"pill_dataset/medium_pilldataset/100client/dirichlet/mixed_noise_data_idx_alpha_0.1_cluster_20_holdout.json", "w") as f:
    json.dump(holdout_idx, f)
breakpoint()