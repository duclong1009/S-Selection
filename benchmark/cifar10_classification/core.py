from torchvision import datasets, transforms
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
from benchmark.toolkits import IDXTaskPipe as TaskPipe
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import CustomDataset, load_dataset_idx
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image

class Specific_dataset(VisionDataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if train:
            self.data = np.load(f"{root}/X_train.npy")
            self.targets = np.load(f"{root}/Y_train.npy")
        else:
            self.data = np.load("cifar10/original/X_test.npy")
            self.targets = np.load("cifar10/original/Y_test.npy")

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.data)

import json

class TaskReader(object):
    def __init__(self,reader_config):
        print(">>>>>> Using cifar10 dataset")
        self.reader_config = reader_config


    def load_data(self,):
        transforms_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = Specific_dataset(root= self.reader_config["data_path"],train=True,transform=transforms_cifar10)
        # train_dataset = datasets.cifar10("./data/cifar10/", train=True, download=True, transform=transforms_cifar10)
        test_dataset = datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=transforms_cifar10)
        return train_dataset, test_dataset
    
    def setup_clients(self,):
        train_dataset, test_dataset = self.load_data()
        data_idx = load_dataset_idx(f"{self.reader_config['data_path']}/data_idx.json")
        
        n_clients = len(data_idx)
        train_data = [CustomDataset(train_dataset, data_idx[idx]) for idx in range(n_clients)]
        return train_data, test_dataset, n_clients


