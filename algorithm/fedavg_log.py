from benchmark.toolkits import CustomDataset
from .fedbase import BasicServer, BasicClient
import importlib
import utils.fflow as flw
import utils.fmodule
import copy
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import math

class Server(BasicServer):
    """
    Nhu fedalgo1 nhung su dung heuristic tinh h_t
    """
    def __init__(self, option, model, clients, test_data=None, device="cpu"):
        super(Server, self).__init__(option, model, clients, test_data, device)
        self.sampler = utils.fmodule.Sampler
        self.threshold_score = 0
        #init goodness_cached
        self.goodness_cached = {}
        self.saved_histogram = {}
        for _ in range(len(clients)):
            self.goodness_cached[_] = 0
            # self.saved_histogram[_] = []
        self.score_range = 0
        self.max_score = 0

    
    def cal_threshold(self, aggregated_histogram):
        # 
        if aggregated_histogram == None:
            return 0
        threshold_value = utils.fmodule.Sampler.cal_threshold(
            aggregated_histogram
        )
        return threshold_value

    def iterate(self):
        self.selected_clients = self.sample()
        utils.fmodule.LOG_DICT["selectd_client"] = self.selected_clients
        flw.logger.info(f"Selected clients : {self.selected_clients}")
        
        aggregated_histogram = self.communicate_score(self.selected_clients)
        self.aggregated_histogram = aggregated_histogram
        self.threshold_score = self.cal_threshold(self.aggregated_histogram)
        received_information = self.communicate(self.selected_clients)
        n_samples = received_information["n_samples"]
        utils.fmodule.LOG_DICT["selected_samples"] = n_samples
        models = received_information["model"]
        list_vols = copy.deepcopy(self.local_data_vols)
        for i, cid in enumerate(self.selected_clients):
            list_vols[cid] = n_samples[i]
        # self.total_data_vol = sum(self.local_data_vols)
        print(
            f"Total samples which participate training : {sum(n_samples)}/{sum([self.local_data_vols[i] for i in self.selected_clients])} samples"
        )
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models,list_vols)

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
        local_threshold = self.threshold_score
        if "threshold_score" not in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["threshold_score"] = {}
        utils.fmodule.LOG_DICT["threshold_score"][
            f"client_{client_id}"
        ] = local_threshold

        return {
            "threshold": local_threshold,
            "score_range": self.score_range,
        }


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

    def pack_model_range(self, client_id):
        return {
            "score_range": self.score_range,
            "model": copy.deepcopy(self.model),
        }

    def aggregate_histogram(self, list_histograms):
        import math
        n_bins = math.ceil(self.max_score / self.score_range)
        final_histogram = np.zeros((n_bins))
        for list_n_, list_range in list_histograms:
            n_ = list_n_.shape[0]
            new_histogram = list(list_n_) + [0] * (n_bins - n_)
            final_histogram += new_histogram
        return final_histogram, range(n_bins + 1) * np.array(self.score_range)


    def communicate_score_with_client(self, selected_clients):
        cpkqs = [self.communicate_score_with(cid) for id, cid in enumerate(selected_clients)]
        self.received_score = self.unpack(cpkqs)
        return self.received_score

    def communicate_score_with(self, client_id):
        svr_pkg = self.pack_model_range(client_id)
        # 
        return self.clients[client_id].reply_score(svr_pkg)

    def update_goodness_cached(self, goodness, selected_clients): 
        for i,id in enumerate(selected_clients):
            self.goodness_cached[id] = goodness[i]

    def communicate_score(self, selected_clients):
        received_informations = self.communicate_score_with_client(selected_clients)
        # 
        list_histograms = received_informations["score"]
        list_max_scores = received_informations["max_score"]
        list_goodness = received_informations["goodness"]
        max_sc = max(list_max_scores)
        self.max_score = max(max_sc,self.max_score)
        #Save goodness score 
        self.update_goodness_cached(list_goodness,selected_clients)

        aggregated_histogram = None
        if self.score_range > 0:
            aggregated_histogram = self.aggregate_histogram(list_histograms)
            
        else:
            self.score_range = max(list_max_scores)/self.option["bins"]

        return aggregated_histogram

import statistics

class Client(BasicClient):
    def __init__(self, option, name="", train_data=None, valid_data=None, device="cpu"):
        super(Client, self).__init__(option, name, train_data, valid_data, device)

    def calculate_importance(self, model):
        criteria = nn.CrossEntropyLoss()
        # sorted_idx, sorted_score, np.array(list_conf), np.array(list_prediction)
        # list_score, list_idx,list_conf,list_prediction
        _, score_list_on_cl,list_conf,list_prediction = utils.fmodule.Sampler.get_information(
            self.train_data, model, criteria, self.device
        )
        if not "list_conf" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["list_conf"] = {}
        if not "list_prediction" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["list_prediction"] = {}
            
        utils.fmodule.LOG_DICT["list_conf"][f"client_{self.id}"] = list_conf
        utils.fmodule.LOG_DICT["list_prediction"][f"client_{self.id}"] = list_prediction
        #  
        self.score_cached = score_list_on_cl


    def cal_goodness(self, list_of_score):
        list_ = list(list_of_score)
        if self.option["type_of_goodness"] == "mean":
            return sum(list_)/ len(list_)
        elif self.option["type_of_goodness"] == "median":
            return statistics.median(list_)

    def unpack_model_range(self,svr_pkg):
        return svr_pkg["model"], svr_pkg["score_range"]

    def build_histogram(self, list_score, score_range):
        n_bins = math.ceil(max(list_score) / score_range)
        bins = range(n_bins + 1) * np.array(score_range)
        historgram_ = np.histogram(list_score, bins)
        return historgram_

    def reply_score(self, svr_pkg):
        model,score_range = self.unpack_model_range(svr_pkg)
        self.score_range = score_range
        self.model = model
        histogram=None
        self.calculate_importance(copy.deepcopy(model))
        if not "score_list" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["score_list"] = {}
        # utils.fmodule.LOG_DICT["score_list"][f"client_{self.id}"] = list(self.score_cached)
        utils.fmodule.LOG_DICT["score_list"][f"client_{self.id}"] = []
        if score_range !=0 :
            histogram = self.build_histogram(self.score_cached, score_range)
        cpkg = self.pack_score(histogram)

        return cpkg

    def pack_score(self, histogram):
        if self.score_range ==0 :
            return {
                "score": [],
                # "max_score": max(self.score_cached),
                # "goodness": self.cal_goodness(self.score_cached)
                "max_score":0,
                "goodness":0
            }
        else:
            return {
                "score": histogram,
                # "max_score": max(self.score_cached),
                # "goodness": self.cal_goodness(self.score_cached)
                "max_score":0,
                "goodness":0
            }

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
        # selected_idx = utils.fmodule.Sampler.sample_using_cached(self.score_cached,threshold)
        selected_idx = range(len(self.score_cached))
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
