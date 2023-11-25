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
import torch.nn as nn
import time

from torch.utils.data import DataLoader
class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None,device='cpu'):
        super(Server, self).__init__(option, model, clients, test_data, device)
        self.init_algo_para({'alpha': 0.1})
        self.h = self.model.zeros_like()
        self.sampler = utils.fmodule.Sampler

    def iterate(self):
        self.selected_clients = self.sample()
        utils.fmodule.LOG_DICT["selectd_client"] = self.selected_clients
        flw.logger.info(f"Selected clients : {self.selected_clients}")

        threshold_score = self.cal_threshold(self.selected_clients)
        self.threshold_score = threshold_score
        received_information = self.communicate(self.selected_clients)
        n_samples = received_information["n_samples"]
        utils.fmodule.LOG_DICT["selected_samples"] = n_samples
        print("Number samples of this round: ",n_samples)
        models = received_information["model"]
        list_vols = copy.deepcopy(self.local_data_vols)
        for i, cid in enumerate(self.selected_clients):
            list_vols[cid] = n_samples[i]
        # self.total_data_vol = sum(self.local_data_vols)
        print(
            f"Total samples which participatfe training : {sum(n_samples)}/{sum([self.local_data_vols[i] for i in self.selected_clients])} samples"
        )
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models,list_vols)

    def aggregate(self, models, list_vols=None):
        self.h = self.h - self.alpha * (1.0 / self.num_clients * fmodule._model_sum(models) - self.model)
        new_model = fmodule._model_average(models) - 1.0 / self.alpha * self.h
        return new_model

    def communicate_with(self, client_id):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client_id: the id of the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        # package the necessary information
        svr_pkg = self.pack_threshold(client_id)
        # listen for the client's response
        return self.clients[client_id].reply(svr_pkg)
    
    def pack_threshold(self, client_id):
        utils.fmodule.LOG_DICT["threshold_score"] = self.threshold_score 
        return {"threshold": self.threshold_score}

    def calculate_importance(self, selected_clients):
        cpkqs = [self.communicate_score_with(cid) for id, cid in enumerate(selected_clients)]
        self.received_score = self.unpack_score(cpkqs)

    def unpack_score(self, cpkqs):
        list_ = []
        for l in cpkqs:
            list_.append(l["score"])
        return list_

    def communicate_score_with(self, client_id):
        svr_pkg = self.pack_model(client_id)
        return self.clients[client_id].reply_score(svr_pkg)

    def cal_threshold(self, selected_clinets):
        self.calculate_importance(selected_clinets)
        list_n_, interval_histogram = self.aggregate_hist()
        threshold_value = utils.fmodule.Sampler.cal_threshold(
            (list_n_, interval_histogram)
        )
        return threshold_value
    
    def aggregate_hist(self):
        max_len = max([len(x) for x in self.received_score])
        new_hist = [np.concatenate((x, np.zeros((max_len - len(x))))) for x in self.received_score]
        return np.sum(new_hist,0), [i * self.option['interval_histogram'] for i in range(max_len + 1)]
from benchmark.toolkits import CustomDataset

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None, device='cpu'):
        super(Client, self).__init__(option, name, train_data, valid_data, device)
        self.gradL = None
        self.alpha = option['alpha']
        self.interval_histogram = option['interval_histogram']

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

    def calculate_importance(self, model):
        criteria = nn.CrossEntropyLoss()
        _, score_list_on_cl = utils.fmodule.Sampler.cal_score(
            self.train_data, model, criteria, self.device
        )
        self.score_cached = score_list_on_cl

    def cal_histogram(self,):
        import copy, math
        score_list = copy.deepcopy(self.score_cached)
        n_bins = math.ceil(max(score_list) / self.interval_histogram)
        return np.histogram(score_list, bins= [self.interval_histogram * i for i in range(n_bins +1)])[0]
    
    def reply_score(self, svr_pkg):
        model = self.unpack_model(svr_pkg)
        self.model = copy.deepcopy(model)
        self.calculate_importance(copy.deepcopy(model))
        value_hist = self.cal_histogram()
        if not "score_list" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["score_list"] = {}
        utils.fmodule.LOG_DICT["score_list"][f"client_{self.id}"] = list(self.score_cached)


        cpkg = self.pack_histogram(value_hist)
        return cpkg

    def pack_histogram(self, score_list):
        return {"score": score_list}

    def unpack_threshold(self, svr_pkg):
        return svr_pkg["threshold"]
    

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the updated
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        threshold = self.unpack_threshold(svr_pkg)
        selected_idx = utils.fmodule.Sampler.sample_using_cached(self.score_cached,threshold)
        if len(selected_idx) != 0 :
        # selected_idx = range(len(self.train_data))
            current_dataset = CustomDataset(self.train_data, selected_idx)
            self.data_loader = DataLoader(
                current_dataset,
                batch_size=self.batch_size,
                num_workers=self.loader_num_workers,
                shuffle=True,
            )
            self.threshold = threshold
            self.train(self.model)
        cpkg = self.pack(self.model, len(selected_idx))
        return cpkg

    def pack(self, model, n_trained_samples):
        return {"model": model, "n_samples": n_trained_samples}
