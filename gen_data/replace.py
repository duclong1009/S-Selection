import json
import numpy as np
import random
np.random.seed(43)
random.seed(43)
with open("pill_dataset/medium_pilldataset/categories.json", "r") as f:
    categories = json.load(f)

with open("pill_dataset/medium_pilldataset/100client/dirichlet/clean_data_idx_alpha_0.1_cluster_30.json", "r") as f:
    data_idx = json.load(f)

noise_type = "bright"

original_list = []
replace_list = []

for label in categories.keys():
    list_noise_id = categories[label][noise_type]
    list_clean = categories[label]['clean']
    n_noise_imgs = len(list_noise_id)
    random.shuffle(list_clean)
    selected_clean = np.random.choice(list_clean, n_noise_imgs, replace=False)
    # breakpoint()
    assert len(original_list) == len(set(original_list))
    assert len(replace_list) == len(set(replace_list))
    original_list += list(selected_clean)
    replace_list += list(list_noise_id)

hash_dict = {}
for i,j in zip(original_list, replace_list):
    hash_dict[i] = j

import copy
for client_id in data_idx.keys():
    client_id_list = copy.deepcopy(data_idx[client_id])
    for img in original_list:
        if img in client_id_list:
            client_id_list.remove(img)
            client_id_list.append(hash_dict[img])
    data_idx[client_id] = client_id_list
# for i, clean_img_id in enumerate(original_list):
#     for client_id in data_idx.keys():
#         if clean_img_id in data_idx[client_id]:
#             data_idx[client_id].remove(clean_img_id)
#             data_idx[client_id].append(replace_list[i])
            # break

with open(f"pill_dataset/medium_pilldataset/100client/dirichlet/{noise_type}_data_idx_alpha_0.1_cluster_30.json", "w") as f:
    json.dump(data_idx, f)
















































# import copy
# data_idx = copy.deepcopy(data_idx)
# list_clean = []
# for key in categories.keys():
#     label = int(key)
#     list_clean_idx = categories[key]['clean']
#     list_clean += list_clean_idx

# noise_type = 'cover'
# list_noise = []
# for key in categories.keys():
#     label = int(key)
#     list_noise_idx = categories[key][noise_type]
#     list_noise += list_noise_idx


# random.shuffle(list_clean)
# n_noise_imgs = len(list_noise)
# list_selected_rl = np.random.choice(list_clean, n_noise_imgs, replace=False)

# hash_dict = {}

# for i,j in zip(list_selected_rl, list_noise):
#     hash_dict[i] = j

# for clean_img_id in list_selected_rl:
#     for client_id in data_idx.keys():
#         if clean_img_id in data_idx[client_id]:
#             data_idx[client_id].remove(clean_img_id)
#             data_idx[client_id].append(hash_dict[clean_img_id])

# with open(f"pill_dataset/medium_pilldataset/100client/dirichlet/{noise_type}_data_idx_alpha_0.1_cluster_20.json", "w") as f:
#     json.dump(data_idx, f)
# breakpoint()