"""
This is a non-official implementation of Scaffold proposed in 'Stochastic
Controlled Averaging for Federated Learning' (ICML 2020).
"""
from benchmark.toolkits import CustomDataset
import utils.fflow as flw
import utils.fmodule
import copy
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time 
from .fedbase import BasicServer, BasicClient
import copy
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None,device="cpu"):
        super(Server, self).__init__(option, model, clients, test_data,device)
        self.init_algo_para({'eta':1.0})
        self.cg = self.model.zeros_like()
        self.sampler = utils.fmodule.Sampler

    def pack_model(self, client_id):
        return {
            "model": copy.deepcopy(self.model),
            "cg": self.cg,
        }

    def iterate(self):
        # sample clients
        self.selected_clients = self.sample()
        utils.fmodule.LOG_DICT["selectd_client"] = self.selected_clients
        flw.logger.info(f"Selected clients : {self.selected_clients}")

        threshold_score = self.cal_threshold(self.selected_clients)
        self.threshold_score = threshold_score
        # local training
        received_information = self.communicate(self.selected_clients)
        dys, dcs, n_samples = received_information['dy'], received_information['dc'], received_information['n_samples']
        # aggregate
        utils.fmodule.LOG_DICT["selected_samples"] = n_samples
        print("Number samples of this round: ",n_samples)
        self.model, self.cg = self.aggregate(dys, dcs)
        return

    def aggregate(self, dys, dcs):
        # x <-- x + eta_g * dx = x + eta_g * average(dys)
        # c <-- c + |S|/N * dc = c + |S|/N * average(dcs)
        dx = fmodule._model_average(dys)
        dc = fmodule._model_average(dcs)
        new_model = self.model + self.eta * dx
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c

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
    
class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None,device="cpu"):
        super(Client, self).__init__(option, name, train_data, valid_data,device)
        self.c = None
        self.interval_histogram = option['interval_histogram']

    @fmodule.with_multi_gpus
    def train(self, model, cg):
        """
        The codes of Algorithm 1 that updates the control variate
          12:  ci+ <-- ci - c + 1 / K / eta_l * (x - yi)
          13:  communicate (dy, dc) <-- (yi - x, ci+ - ci)
          14:  ci <-- ci+
        Our implementation for efficiency
          dy = yi - x
          dc <-- ci+ - ci = -1/K/eta_l * (yi - x) - c = -1 / K /eta_l *dy - c
          ci <-- ci+ = ci + dc
          communicate (dy, dc)
        """
        model.train()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        cg.freeze_grad()
        if self.c is None: self.c = model.zeros_like()
        self.c.freeze_grad()
        optimizer = self.calculator.get_optimizer(model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.train_one_step(model, batch_data)['loss']
            loss.backward()
            # y_i <-- y_i - eta_l ( g_i(y_i)-c_i+c )  =>  g_i(y_i)' <-- g_i(y_i)-c_i+c
            for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()):
                pm.grad = pm.grad - pc + pcg
            optimizer.step()
        dy = model - src_model
        dc = -1.0 / (self.num_steps * self.learning_rate) * dy - cg
        self.c = self.c + dc
        return dy, dc

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
            dy, dc= self.train(self.model, self.c_g)
        else:
            dy, dc = 0, 0 
        # dy, dc = self.train(model, c_g)
        cpkg = self.pack(dy, dc, len(selected_idx))
        return cpkg

    def pack(self, dy, dc, n_samples):
        return {
            "dy": dy,
            "dc": dc,
            "n_samples": n_samples
        }



    def unpack_model(self, received_pkg):
        # unpack the received package
        return received_pkg['model'], received_pkg['cg']

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
        model, cg = self.unpack_model(svr_pkg)
        self.c_g = cg
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
