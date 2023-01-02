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
import statistics

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None, device="cpu"):
        super(Server, self).__init__(option, model, clients, test_data, device)
        self.sampler = utils.fmodule.Sampler
        self.threshold_score = 0
        #init goodness_cached
        self.goodness_cached = {}
        for _ in range(len(clients)):
            self.goodness_cached[_] = 0
        self.have_cache = set()
        self.score_range = 0
        self.max_score = 0

    def f_round(
        self,
    ):
        if self.current_round == 1:
            return 0
        probability = (
            1.0
            / (self.current_round * self.option['o'] * math.sqrt(2 * math.pi))
            * (math.exp(-0.5 * ((math.log(self.current_round) - self.option['u']) / self.option['o']) ** 2))
        )
        return probability

    def aggregate_histogram(self, list_histograms):
        n_bins = math.ceil(self.max_score / self.score_range)
        final_histogram = np.zeros((n_bins))
        for list_n_, list_range in list_histograms:
            n_ = list_n_.shape[0]
            new_histogram = list(list_n_) + [0] * (n_bins - n_)
            final_histogram += new_histogram
        return final_histogram, range(n_bins) * np.array(self.score_range)

    def f_goodness(self,):
        list_goodness = [self.goodness_cached[i] for i in self.selected_clients]
        mean_goodness = sum(list_goodness)/len(list_goodness)
        if mean_goodness == 0 :
            return 0
        return 1.0/ mean_goodness * 2

    def modify_ratio(self,):
        f_round = self.f_round()
        f_goodness = self.f_goodness()
        f_total = 5.0 * f_round + 1.0 * f_goodness
        extra_ignored_rate = min(f_total, 0.2)
        self.option["ratio"] -= extra_ignored_rate
        print(f"Update ratio round {self.current_round}: {self.option['ratio'] + extra_ignored_rate} >>>>  {self.option['ratio']}")
    
    def iterate(self):
        saved_ratio = copy.deepcopy(self.option["ratio"])
        self.selected_clients = self.sample()
        utils.fmodule.LOG_DICT["selected_clients"] = self.selected_clients
        flw.logger.info(f"Selected clients : {self.selected_clients}")
        self.modify_ratio()
        received_information = self.communicate(self.selected_clients)
        n_samples = received_information["n_samples"]
        models = received_information["model"]
        list_score = received_information["score_list"]
        list_max_score = received_information["max_score"]
        # breakpoint()
        list_goodness = received_information["goodness_score"]
        self.update_goodness_cached(list_goodness,self.selected_clients)
        self.max_score = max(list_max_score)
        self.received_score = list_score

        if self.score_range != 0:
            if sum([len(i) for i in list_score]) != 0:
                aggreagted_histogram = self.aggregate_histogram(list_score)
                threshold = self.cal_threshold(aggreagted_histogram)
                self.threshold_score = threshold
        if self.score_range == 0:
            self.score_range = self.max_score / self.option["bins"]

        list_vols = copy.deepcopy(self.local_data_vols)
        for i, cid in enumerate(self.selected_clients):
            list_vols[cid] = n_samples[i]
        print(f"Total samples which participate training : {sum(n_samples)} samples")
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, list_vols)
        self.have_cache.update(self.selected_clients)
        self.option["ratio"] = saved_ratio
        
    def update_goodness_cached(self, goodness, selected_clients): 
        for i,id in enumerate(selected_clients):
            self.goodness_cached[id] = goodness[i]
            
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
        svr_pkg = self.pack_model_threshold(client_id)
        # listen for the client's response
        return self.clients[client_id].reply(svr_pkg)

    def pack_model_threshold(self, client_id):
        if "threshold_score" not in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["threshold_score"] = {}

        utils.fmodule.LOG_DICT["threshold_score"][
            f"client_{client_id}"
        ] = self.threshold_score
        return {
            "model": copy.deepcopy(self.model),
            "threshold": self.threshold_score,
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

    def unpack_score(self, cpkqs):
        list_ = []
        for l in cpkqs:
            list_ += list(l["score"])
        return list_

    def communicate_score_with(self, client_id):
        svr_pkg = self.pack(client_id)
        return self.clients[client_id].reply_score(svr_pkg)

    def cal_threshold(self, received_score):
        list_n_, interval_histogram = np.histogram(np.array(received_score), bins=1000)
        threshold_value = utils.fmodule.Sampler.cal_threshold(
            (list_n_, interval_histogram)
        )
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

    def cal_goodness(self, list_of_score):
        list_ = list(list_of_score)
        if self.option["type_of_goodness"] == "mean":
            return sum(list_)/ len(list_)
        elif self.option["type_of_goodness"] == "median":
            return statistics.median(list_)

    def reply_score(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.model = copy.deepcopy(model)
        self.calculate_importance(copy.deepcopy(model))
        cpkg = self.pack_score(self.score_cached)

        return cpkg

    def pack_score(self, score_list):
        return {"score": score_list}

    def unpack_model_threshold(self, svr_pkg):
        return svr_pkg["model"], svr_pkg["threshold"], svr_pkg["score_range"]

    def build_histogram(self, list_score, score_range):
        import math

        n_bins = math.ceil(max(list_score) / score_range)
        bins = range(n_bins + 1) * np.array(score_range)
        historgram_ = np.histogram(list_score, bins)
        return historgram_

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

        model, threshold, score_range = self.unpack_model_threshold(svr_pkg)
        self.score_range = score_range
        self.model = model
        self.calculate_importance(model)
        if not "score_list" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["score_list"] = {}
        utils.fmodule.LOG_DICT["score_list"][f"client_{self.id}"] = list(
            self.score_cached
        )
        if self.score_range != 0:
            historgram_ = self.build_histogram(self.score_cached, self.score_range)
            self.historgram_ = historgram_
        goodness_score = self.cal_goodness(self.score_cached)
        selected_idx = utils.fmodule.Sampler.sample_using_cached(
            self.score_cached, threshold
        )
        current_dataset = CustomDataset(self.train_data, selected_idx)
        self.data_loader = DataLoader(
            current_dataset,
            batch_size=self.batch_size,
            num_workers=self.loader_num_workers,
            shuffle=True,
        )
        self.threshold = threshold
        self.train(self.model)
        cpkg = self.pack(self.model, len(selected_idx),goodness_score)
        return cpkg

    def pack(self, model, n_trained_samples,goodness_score):
        if self.score_range == 0:
            return {
                "model": model,
                "n_samples": n_trained_samples,
                "score_list": [],
                "max_score": max(self.score_cached),
                "goodness_score": goodness_score,
            }
        else:
            return {
                "model": model,
                "n_samples": n_trained_samples,
                "score_list": self.historgram_,
                "max_score": max(self.score_cached),
                "goodness_score": goodness_score,
            }
