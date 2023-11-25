from torchvision import datasets, transforms
from benchmark.toolkits import DefaultTaskGen, load_dataset_idx, CustomDataset
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
from benchmark.toolkits import IDXTaskPipe as TaskPipe
from torch.utils.data import DataLoader
import torchvision

# class TaskGen(DefaultTaskGen):
#     def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.2, seed=0):
#         super(TaskGen, self).__init__(benchmark='cifar100_classification',
#                                       dist_id=dist_id,
#                                       num_clients=num_clients,
#                                       skewness=skewness,
#                                       rawdata_path='./benchmark/RAW_DATA/CIFAR100',
#                                       local_hld_rate=local_hld_rate,
#                                       seed=seed
#                                       )
#         self.num_classes = 100
#         self.save_task = TaskPipe.save_task
#         self.visualize = self.visualize_by_class
#         self.source_dict = {
#             'class_path': 'torchvision.datasets',
#             'class_name': 'CIFAR100',
#             'train_args': {
#                 'root': '"'+self.rawdata_path+'"',
#                 'download': 'True',
#                 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])',
#                 'train': 'True'
#             },
#             'test_args': {
#                 'root': '"'+self.rawdata_path+'"',
#                 'download': 'True',
#                 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])',
#                 'train': 'False'
#             }
#         }

#     def load_data(self):
#         self.train_data = datasets.CIFAR100(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
#         self.test_data = datasets.CIFAR100(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))



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
