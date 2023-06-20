import torchvision.datasets
from torchvision import datasets, transforms
from benchmark.toolkits import CustomDataset, DefaultTaskGen, PillImageFolder, load_dataset_idx
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
from benchmark.toolkits import IDXTaskPipe as TaskPipe
from torchvision.datasets import ImageFolder

class TaskReader(object):
    def __init__(self,reader_config):
        self.reader_config = reader_config
        
    def load_data(self,):
        with open(f"{self.reader_config['data_path']}/train_dataset_samples.json", "r") as fp:
            import json
            read_samples = json.load(fp)['samples']
        train_dataset = PillImageFolder(
            f"{self.reader_config['data_path']}/train",
            transform=transforms.Compose([transforms.ToTensor()]),
            samples=read_samples,
        )
        
        test_dataset = ImageFolder(
            f"{self.reader_config['data_path']}/test",
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        return train_dataset, test_dataset
    
    def setup_clients(self,):
        train_dataset, test_dataset = self.load_data()
        data_idx = load_dataset_idx(self.reader_config["idx_path"])
        n_clients = len(data_idx)
        train_data = [CustomDataset(train_dataset, data_idx[idx]) for idx in range(n_clients)]
        return train_data, test_dataset, n_clients