from torchvision import datasets, transforms
from benchmark.toolkits import DefaultTaskGen, load_dataset_idx, CustomDataset
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
from benchmark.toolkits import IDXTaskPipe as TaskPipe
from torch.utils.data import DataLoader
import torchvision



class TaskReader(object):
    def __init__(self,reader_config):
        self.reader_config = reader_config
        
    def load_data(self,):
        # with open(f"{self.reader_config['data_path']}/train_dataset_samples.json", "r") as fp:
        #     import json
        #     read_samples = json.load(fp)['samples']
        train_dataset = torchvision.datasets.CIFAR100('./cifar100', train=True, download=True,  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test_dataset = torchvision.datasets.CIFAR100('./cifar100', train=False, download=True,  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        return train_dataset, test_dataset
    
    def setup_clients(self,):
        train_dataset, test_dataset = self.load_data()
        data_idx = load_dataset_idx(self.reader_config["idx_path"])
        n_clients = len(data_idx)
        train_data = [CustomDataset(train_dataset, data_idx[idx]) for idx in range(n_clients)]
        return train_data, test_dataset, n_clients
