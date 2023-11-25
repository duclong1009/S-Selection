import torchvision
from torchvision import datasets, transforms
import torch
import numpy as np
from PIL import Image, ImageEnhance
import torchvision
from torchvision import datasets, transforms
import torch
import numpy as np
from PIL import Image, ImageEnhance

train_dataset = torchvision.datasets.CIFAR100('./cifar100', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR100('./cifar100', train=False, download=True,  transform=transforms.Compose([transforms.ToTensor()]))
list_x = []
list_y = []
ratio = 0.3
n_samples = int(ratio * len(train_dataset))
to_tensor = transform=transforms.Compose([transforms.ToTensor()])

selected_idx = np.random.choice([i for i in range(len(train_dataset))], n_samples, replace=False)
factor = 5

for id in range(len(train_dataset)):
    if id not in selected_idx:
        list_x.append(to_tensor(train_dataset[id][0]).unsqueeze(0))
    else:
        o_image = train_dataset[id][0]
        img_enhancer = ImageEnhance.Brightness(train_dataset[id][0])
        enhanced_output = img_enhancer.enhance(factor)
        enhanced_output.save("output/bright-image.png")
        o_image.save("output/nor-image.png")

        list_x.append(to_tensor(enhanced_output).unsqueeze(0))
    list_y.append(train_dataset[id][1])
x_train = torch.cat(list_x, 0).numpy()
selected_idx = [int(i) for i in selected_idx]
with open("cifar100_bright/y.json", "w") as f:
    import json
    json.dump({"label":list_y},f)

with open("cifar100_bright/noise_y_3.json", "w") as f:
    import json
    json.dump({"noise_idx": selected_idx},f)
import numpy as np
with open("cifar100_bright/x_3.npy","wb") as f:
    np.save(f,x_train)

train_dataset = torchvision.datasets.CIFAR100('./cifar100', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR100('./cifar100', train=False, download=True,  transform=transforms.Compose([transforms.ToTensor()]))
list_x = []
list_y = []
ratio = 0.3
n_samples = int(ratio * len(train_dataset))
to_tensor = transform=transforms.Compose([transforms.ToTensor()])

selected_idx = np.random.choice([i for i in range(len(train_dataset))], n_samples, replace=False)
factor = 5

for id in range(len(train_dataset)):
    if id not in selected_idx:
        list_x.append(to_tensor(train_dataset[id][0]).unsqueeze(0))
    else:
        o_image = train_dataset[id][0]
        img_enhancer = ImageEnhance.Brightness(train_dataset[id][0])
        enhanced_output = img_enhancer.enhance(factor)
        enhanced_output.save("output/bright-image.png")
        o_image.save("output/nor-image.png")

        list_x.append(to_tensor(enhanced_output).unsqueeze(0))
    list_y.append(train_dataset[id][1])
x_train = torch.cat(list_x, 0).numpy()
selected_idx = [int(i) for i in selected_idx]
with open("cifar100_bright/y.json", "w") as f:
    import json
    json.dump({"label":list_y},f)

with open("cifar100_bright/noise_y_3.json", "w") as f:
    import json
    json.dump({"noise_idx": selected_idx},f)
import numpy as np
with open("cifar100_bright/x_3.npy","wb") as f:
    np.save(f,x_train)
breakpoint()