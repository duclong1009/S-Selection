"""
This is a non-official implementation of 'Federated Learning Based on Dynamic Regularization'
(http://arxiv.org/abs/2111.04263). The official implementation is at 'https://github.com/alpemreacar/FedDyn'
"""
from .fedbase import BasicServer, BasicClient
import copy
from utils import fmodule
import torch
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

import time

from torch.utils.data import DataLoader
class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None,device='cpu'):
        super(Server, self).__init__(option, model, clients, test_data, device)
        self.init_algo_para({'alpha': 0.1})
        self.h = self.model.zeros_like()

    def aggregate(self, models, list_vols=None):
        self.h = self.h - self.alpha * (1.0 / self.num_clients * fmodule._model_sum(models) - self.model)
        new_model = fmodule._model_average(models) - 1.0 / self.alpha * self.h
        return new_model


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None, device='cpu'):
        super(Client, self).__init__(option, name, train_data, valid_data, device)
        self.gradL = None
        self.alpha = option['alpha']

    @ fmodule.with_multi_gpus
    def train(self, model):
        if self.data_loader == None:
            print(f"Client {self.id} init its dataloader")
            self.data_loader = DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                num_workers=self.loader_num_workers,
                shuffle=True,
            )

        if self.gradL == None:self.gradL = model.zeros_like()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
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
                data, labels = data.float().to(self.device), labels.long().to(
                    self.device
                )
                model.zero_grad()
                l2 = 0
                l3 = 0
                optimizer.zero_grad()
                outputs = model(data)
                loss = self.calculator.criterion(outputs, labels)
                for pgl, pm, ps in zip(self.gradL.parameters(), model.parameters(), src_model.parameters()):
                    l2 += torch.dot(pgl.view(-1), pm.view(-1))
                    l3 += torch.sum(torch.pow(pm - ps, 2))
                loss = loss - l2 + 0.5 * self.alpha * l3
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            # training_loss.append(training_loss / len(current_dataloader))
            list_training_loss.append(training_loss / len(self.data_loader))
        return

