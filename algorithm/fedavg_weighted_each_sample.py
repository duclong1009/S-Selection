from .fedbase import BasicServer, BasicClient
import numpy as np
import utils.fmodule
from utils import fmodule
import copy
import os
import utils.fflow as flw
import utils.system_simulator as ss
import math
import collections
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None,device='cpu'):
        super(Server, self).__init__(option, model, clients, test_data,device)

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None,device='cpu'):
        super(Client, self).__init__(option, name, train_data, valid_data,device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.noisy_id = option["noisy_id"][str(name)]

    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        if self.data_loader == None:
            print(f"Client {self.id} init its dataloader")
            self.data_loader = DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.loader_num_workers,
                shuffle=True,
            )
        model.train()
        optimizer = self.calculator.get_optimizer(
            model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )
        list_training_loss = []
        for epoch in range(self.epochs):
            training_loss = 0
            for data, labels, idxs in self.data_loader:
                mapped_idxs = [self.train_data.idxs[ix] for ix in idxs]
                weighted = [1] * len(mapped_idxs)
                for ii,ix in enumerate(mapped_idxs):
                    if ix in self.noisy_id:
                        weighted[ii] =  self.option["weight_for_noise"]
                
                data, labels = data.float().to(self.device), labels.long().to(
                    self.device
                )
                optimizer.zero_grad()
                outputs = model(data)
                loss = self.criterion(outputs, labels)
                loss = torch.mean(loss * torch.tensor(weighted).to(self.device)) 
                # breakpoint()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            list_training_loss.append(training_loss / len(self.data_loader))
        if not "training_loss" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["training_loss"] = {}
        utils.fmodule.LOG_DICT["training_loss"][
            f"client_{self.id}"
        ] = list_training_loss
        utils.fmodule.LOG_WANDB["mean_training_loss"] = sum(list_training_loss) / len(
            list_training_loss
        )
        return 