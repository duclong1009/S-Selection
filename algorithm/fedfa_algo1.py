from benchmark.toolkits import CustomDataset
from .fedbase import BasicServer, BasicClient
import utils.fflow as flw
import utils.fmodule
import copy
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time 

"""
This is a non-official implementation of 'Fairness and Accuracy in Federated Learning' (http://arxiv.org/abs/2012.10069)
"""
from utils import fmodule
from .fedbase import BasicServer, BasicClient
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None, device='cpu'):
        super(Server, self).__init__(option, model, clients, test_data, device)
        self.init_algo_para({'beta': 0.5, 'gamma': 0.9})
        # self.m = fmodule._modeldict_zeroslike(self.model.state_dict())
        self.m = copy.deepcopy(self.model) * 0.0
        self.alpha = 1.0 - self.beta
        self.eta = option['learning_rate']
        self.sampler = utils.fmodule.Sampler

    def iterate(self):
        # sample clients
        self.selected_clients = self.sample()
        utils.fmodule.LOG_DICT["selectd_client"] = self.selected_clients
        flw.logger.info(f"Selected clients : {self.selected_clients}")
        threshold_score = self.cal_threshold(self.selected_clients)
        self.threshold_score = threshold_score

        # training
        res = self.communicate(self.selected_clients)
        models, losses, ACC, F, n_samples = res['model'], res['loss'], res['acc'], res['freq'], res['n_samples']
        utils.fmodule.LOG_DICT["selected_samples"] = n_samples
        print("Number samples of this round: ",n_samples)

        # aggregate
        # calculate ACCi_inf, fi_inf
        sum_acc = np.sum(ACC)
        sum_f = np.sum(F)
        ACCinf = [-np.log2(1.0*acc/sum_acc+0.000001) for acc in ACC]
        Finf = [-np.log2(1-1.0*f/sum_f+0.00001) for f in F]
        sum_acc = np.sum(ACCinf)
        sum_f = np.sum(Finf)
        ACCinf = [acc/sum_acc for acc in ACCinf]
        Finf = [f/sum_f for f in Finf]
        # calculate weight = αACCi_inf+βfi_inf
        p = [self.alpha*accinf+self.beta*finf for accinf,finf in zip(ACCinf,Finf)]
        wnew = self.aggregate(models, p)
        dw = wnew -self.model
        # calculate m = γm+(1-γ)dw
        self.m = self.gamma * self.m + (1 - self.gamma) * dw
        self.model = wnew - self.m * self.eta
        return

    def communicate_score_with(self, client_id):
        svr_pkg = self.pack_model(client_id)
        return self.clients[client_id].reply_score(svr_pkg)
    
    def aggregate(self, models, p):
        return fmodule._model_average(models, p)


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
    
    def cal_threshold(self, selected_clinets):
        self.calculate_importance(selected_clinets)
        list_n_, interval_histogram = self.aggregate_hist()
        threshold_value = utils.fmodule.Sampler.cal_threshold(
            (list_n_, interval_histogram)
        )
        return threshold_value
    
    def calculate_importance(self, selected_clients):
        cpkqs = [self.communicate_score_with(cid) for id, cid in enumerate(selected_clients)]
        self.received_score = self.unpack_score(cpkqs)

    def aggregate_hist(self):
        max_len = max([len(x) for x in self.received_score])
        new_hist = [np.concatenate((x, np.zeros((max_len - len(x))))) for x in self.received_score]
        return np.sum(new_hist,0), [i * self.option['interval_histogram'] for i in range(max_len + 1)]
    
    def unpack_score(self, cpkqs):
        list_ = []
        for l in cpkqs:
            list_.append(l["score"])
        return list_

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None, device="cpu"):
        super(Client, self).__init__(option, name, train_data, valid_data, device)
        self.frequency = 0
        self.momentum = option['gamma']
        self.interval_histogram = option['interval_histogram']

    def reply(self, svr_pkg):
        threshold = self.unpack_threshold(svr_pkg)
        selected_idx = utils.fmodule.Sampler.sample_using_cached(self.score_cached,threshold)
        if len(selected_idx) != 0 :
            current_dataset = CustomDataset(self.train_data, selected_idx)
            self.data_loader = DataLoader(
                current_dataset,
                batch_size=self.batch_size,
                num_workers=self.loader_num_workers,
                shuffle=True,
            )
            self.threshold = threshold
        metrics = self.test(self.model,'train')
        acc, loss = metrics['accuracy'], metrics['loss']
        self.train(self.model)
        cpkg = self.pack(self.model, loss, acc, len(selected_idx))
        return cpkg
    
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
    
    def pack(self, model, loss, acc, n_samples):
        self.frequency += 1
        return {
            "model":model,
            "loss":loss,
            "acc":acc,
            "freq":self.frequency,
            "n_samples": n_samples
        }
