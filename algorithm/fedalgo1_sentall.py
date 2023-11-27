from benchmark.toolkits import CustomDataset
from .fedbase import BasicServer, BasicClient
import importlib
import utils.fflow as flw
import utils.fmodule
import copy
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import time 

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None, device="cpu"):
        super(Server, self).__init__(option, model, clients, test_data, device)
        self.sampler = utils.fmodule.Sampler
        self.ratio = option['ratio']
        
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

    def communicate(self, selected_clients):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        client_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        for cid in communicate_clients:
            client_package_buffer[cid] = None
        for client_id in communicate_clients:
            response_from_client_id = self.communicate_with(client_id)
            packages_received_from_clients.append(response_from_client_id)

        for i, cid in enumerate(communicate_clients):
            client_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [
            client_package_buffer[cid]
            for cid in selected_clients
            if client_package_buffer[cid]
        ]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

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

    def get_threshold(self, list_score):
        n_samples = len(list_score)
        n_selected_samples = int(self.ratio * n_samples)
        idxs = np.argsort(list_score)
        threshold_idxs = idxs[-n_selected_samples]
        threshold_value = list_score[threshold_idxs]
        return threshold_value
    
    def cal_threshold(self, selected_clinets):
        self.calculate_importance(selected_clinets)
        list_score = np.concatenate(self.received_score)
        threshold_value = self.get_threshold(list_score)
        return threshold_value
 
class Client(BasicClient):
    def __init__(self, option, name="", train_data=None, valid_data=None, device="cpu"):
        super(Client, self).__init__(option, name, train_data, valid_data, device)

    def calculate_importance(self, model):
        criteria = nn.CrossEntropyLoss()
        _, score_list_on_cl = utils.fmodule.Sampler.cal_score(
            self.train_data, model, criteria, self.device
        )
        self.score_cached = score_list_on_cl

    def reply_score(self, svr_pkg):
        import time
        start_time = time.time()
        
        model = self.unpack_model(svr_pkg)
        self.model = copy.deepcopy(model)
        self.calculate_importance(copy.deepcopy(model))
        if not "score_list" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["score_list"] = {}
        # if self.score_cached 
        utils.fmodule.LOG_DICT["score_list"][f"client_{self.id}"] = list(self.score_cached)
        if not "assess_importance_time" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["assess_importance_time"] = {}
        
        utils.fmodule.LOG_DICT["assess_importance_time"][f"client_{self.id}"] = time.time() - start_time

        cpkg = self.pack_score_list(self.score_cached)
        return cpkg

    def pack_score_list(self, score_list):
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
        import time
        threshold = self.unpack_threshold(svr_pkg)
        start_time = time.time()
        selected_idx = utils.fmodule.Sampler.sample_using_cached(self.score_cached,threshold)
        selected_time = time.time() - start_time
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
            start_time = time.time()
            self.train(self.model)
            local_training_time = time.time() - start_time
            
            if not "local_training_time" in utils.fmodule.LOG_DICT.keys():
                utils.fmodule.LOG_DICT["local_training_time"] = {}
            utils.fmodule.LOG_DICT["local_training_time"][f"client_{self.id}"] = local_training_time
        
        if not "selected_time" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["selected_time"] = {}
        utils.fmodule.LOG_DICT["selected_time"][f"client_{self.id}"] = selected_time
        
        cpkg = self.pack(self.model, len(selected_idx))
        return cpkg

    def pack(self, model, n_trained_samples):
        return {"model": model, "n_samples": n_trained_samples}
