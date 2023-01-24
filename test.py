from torchvision import datasets, transforms
import numpy as np

from PIL import Image, ImageFilter
import numpy as np
import json

transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST("./data/mnist/", train=True, download=True, transform=transforms_mnist)
test_dataset = datasets.MNIST("./data/mnist/", train=False, download=True, transform=transforms_mnist)
train_dataset.data
with open("mnist/origin/X_train.npy","wb") as f:
    np.save(f,train_dataset.data)
with open("mnist/origin/Y_train.npy","wb") as f:
    np.save(f,train_dataset.targets)
breakpoint()
image = Image.fromarray(np.array(train_dataset.data[0]))
filtered = image.filter(ImageFilter.GaussianBlur(radius=7))
breakpoint()