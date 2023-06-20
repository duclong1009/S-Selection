import json
import numpy as np

idx_file_path ="pill_dataset/medium_pilldataset/100client/clean_unequal_1"
with open(f"{idx_file_path}.json", "r") as f:
    idx_data = json.load(f)


dataset_path = "pill_dataset/medium_pilldataset"

with open(f"{dataset_path}/image_categories_dict.json", "r") as f:
    data_categories = json.load(f)

t = 2
replace_dict = {}
for k in data_categories.keys():
    if k != "labels":
        try:
            clean_img = data_categories[k]["1"]
            noise_img = data_categories[k][str(t)]
            replace_imgs = np.random.choice(clean_img, len(noise_img), replace=False)
            for i, img_idx in enumerate(replace_imgs):
                replace_dict[img_idx] = noise_img[i]
        except:
            pass
new_dict = {}
for client_id in idx_data.keys():
    list_idx_client = idx_data[client_id]
    for idx in list_idx_client:
         if idx in replace_dict.keys():
              list_idx_client.remove(idx)
              list_idx_client.append(replace_dict[idx])
    new_dict[client_id] = list_idx_client



with open(f"{idx_file_path}_remove_{t}.json", "w") as f:
     json.dump(new_dict,f) 
# with open()
        # except:
        #     pass
        #     print(k)
