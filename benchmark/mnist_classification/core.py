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
            self.data = np.load("cifar/original/X_test.npy")
            self.targets = np.load("cifar/original/Y_test.npy")

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



class TaskReader(object):
    def __init__(self,reader_config):
        print(">>>>>> Using MNIST dataset")
        self.reader_config = reader_config


    def load_data(self,):
        transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # 
        # train_dataset = Specific_dataset(
        #     "Gen_data/train",
        #     transform=transforms.Compose([transforms.ToTensor()]),
        #     samples=read_samples,
        # )
        train_dataset = Specific_dataset(root= self.reader_config["data_path"],train=True,transform=transforms_mnist)
        # train_dataset = datasets.MNIST("./data/mnist/", train=True, download=True, transform=transforms_mnist)
        test_dataset = datasets.MNIST("./data/mnist/", train=False, download=True, transform=transforms_mnist)
        return train_dataset, test_dataset
    
    def setup_clients(self,):
        train_dataset, test_dataset = self.load_data()
        data_idx = load_dataset_idx(f"{self.reader_config['data_path']}/data_idx.json")
        n_clients = len(data_idx)
        train_data = [CustomDataset(train_dataset, data_idx[idx]) for idx in range(n_clients)]
        return train_data, test_dataset, n_clients

# class TaskPipe(IDXTaskPipe):
#     def __init__(self):
#         super(TaskPipe, self).__init__()

# class TaskCalculator(ClassificationCalculator):
#     def __init__(self, device):
#         super(TaskCalculator, self).__init__(device)

