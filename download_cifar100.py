import torchvision

train_data = torchvision.datasets.CIFAR100('./cifar100', train=True, download=True)
test_data = torchvision.datasets.CIFAR100('./cifar100', train=False, download=True)