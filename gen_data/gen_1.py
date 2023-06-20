import os 
import json

path = "pill_dataset/medium_pilldataset"
list_pills = os.listdir(f"{path}/train")

with open(f"{path}/train_dataset_samples.json", "r") as f:
    data_infor = json.load(f)['samples']

covert_type_dict = {"clean":1,
                    "bright":2, 
                    "cover":3,
                    "zoom":4}

saved_dict = {}
saved_dict['labels'] = []
for i,item in enumerate(data_infor):
    image_path, class_id = item
    type_image, pill_name = image_path.split("/")[-2], image_path.split("/")[-3]
    if pill_name not in saved_dict.keys():
        saved_dict[pill_name] = {}
    if covert_type_dict[type_image.lower()] not in saved_dict[pill_name].keys():
        saved_dict[pill_name][covert_type_dict[type_image.lower()]] = [i]
    else:
        saved_dict[pill_name][covert_type_dict[type_image.lower()]].append(i)
    saved_dict['labels'].append(class_id)
with open(f"{path}/image_categories_dict.json","w") as f:
    json.dump(saved_dict,f)
# breakpoint()g