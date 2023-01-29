from torchvision import datasets, transforms
import numpy as np

from PIL import Image, ImageFilter
import numpy as np
import json

transforms_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.CIFAR10("./data/cifar10/", train=True, download=True, transform=transforms_cifar10)
test_dataset = datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=transforms_cifar10)
train_dataset.data
with open("cifar10/origin/X_train.npy","wb") as f:
    np.save(f,train_dataset.data)
with open("cifar10/origin/Y_train.npy","wb") as f:
    np.save(f,train_dataset.targets)
breakpoint()
image = Image.fromarray(np.array(train_dataset.data[0]))
filtered = image.filter(ImageFilter.GaussianBlur(radius=7))
breakpoint()