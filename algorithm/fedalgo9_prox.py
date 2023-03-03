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
import pandas as pd
import torch

class FuzzyLogicBase:
    def __init__(self, config) -> None:
        self.config = config
        self.round_config = self.config["round"]
        self.history_cached_config = self.config["history_sampled"]
        self.median_config = self.config["median"]
        self.upperbound_score = self.config["upper_threshold"]
        self.use_upperbound_score = False
        self.config["middle_table"] = pd.read_csv(
            f"{self.config['path']}/middle.csv"
        ).values

        self.config["high_table"] = pd.read_csv(
            f"{self.config['path']}/high.csv"
        ).values
        
    def run(self, input):
        fuzzied_set = self.fuzzifier(input)
        output_rule = self.rulebase(fuzzied_set)
        removed_percentage = self.defuzzifier(output_rule)
        return removed_percentage

    def fuzzifier(self, input):
        round = input["current_round"]
        median_list = input["median_list"]
        history_cached = input["history_cached"]
        mean_median = sum(median_list) / len(median_list)
        fuzzied_value = {}

        if round <= self.round_config["low"]:
            fuzzied_value["round"] = 0
        elif round <= self.round_config["middle"]:
            fuzzied_value["round"] = 1
        else:
            fuzzied_value["round"] = 2

        if mean_median <= self.median_config["low"]:
            fuzzied_value["median"] = 0
        elif mean_median <= self.median_config["middle"]:
            fuzzied_value["median"] = 1
        else:
            fuzzied_value["median"] = 2

        if history_cached <= self.history_cached_config["low"]:
            fuzzied_value["history_cached"] = 0
        elif history_cached <= self.history_cached_config["middle"]:
            fuzzied_value["history_cached"] = 1
        else:
            fuzzied_value["history_cached"] = 2

        return fuzzied_value

    def defuzzifier(self, output_rule):
        return output_rule

    def rulebase(self, fuzzied_set):
        returned_value = 0
        if fuzzied_set["round"] == 0:
            return 0
        elif fuzzied_set["round"] == 1:
            table_inffer = self.config["middle_table"]
        else:
            if fuzzied_set["median"] == 0:
                self.use_upperbound_score = True
            table_inffer = self.config["high_table"]
        return table_inffer[fuzzied_set["history_cached"], fuzzied_set["median"]]


class Server(BasicServer):
    """
    Cat dinh khi bat dau hoi tu
    """

    def __init__(self, option, model, clients, test_data=None, device="cpu"):
        super(Server, self).__init__(option, model, clients, test_data, device)
        self.sampler = utils.fmodule.Sampler
        self.threshold_score = 0
        option["fuzzy_config"] = option["fuzzy_config"][0]
        self.fuzzy_algo = FuzzyLogicBase(option["fuzzy_config"])
        # init goodness_cached
        self.goodness_cached = {}
        self.saved_histogram = {}
        for _ in range(len(clients)):
            self.goodness_cached[_] = 0
            # self.saved_histogram[_] = []
        self.score_range = 0
        self.max_score = 0
        self.history_selected_clients = []
        self.init_algo_para({'mu':0.1})
        
    def cal_threshold(self, aggregated_histogram):
        # 
        if aggregated_histogram == None:
            return 0
        threshold_value = utils.fmodule.Sampler.cal_threshold(aggregated_histogram)
        return threshold_value

    def iterate(self):
        self.selected_clients = self.sample()
        utils.fmodule.LOG_DICT["selectd_client"] = self.selected_clients
        flw.logger.info(f"Selected clients : {self.selected_clients}")

        aggregated_histogram = self.communicate_score(self.selected_clients)
        utils.fmodule.LOG_DICT["selected_ratio"] = utils.fmodule.Sampler.sampler_config[
            "ratio"
        ]
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
        self.model = self.aggregate(models, list_vols)
        self.history_selected_clients.append(self.selected_clients)

    def check_nearly_participated(self):
        if self.current_round == 1:
            return 0
        cared_round = self.option["fuzzy_config"]["history_sampled"]["n_rounds"]
        cared_round = min(cared_round, self.current_round - 1)
        # breakpoint
        # cared_list = self.history_selected_clients[:-cared_round]
        set_ = set()
        for i in range(1, cared_round + 1):
            set_.update(self.history_selected_clients[-i])
        list_ = list(set_)
        clients = len([i for i in self.selected_clients if i in list_])
        return clients * 1.0 / len(self.selected_clients)

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
            "use_upperbound_score": self.fuzzy_algo.use_upperbound_score,
            "upperbound_score": self.fuzzy_algo.upperbound_score,
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
        cpkqs = [
            self.communicate_score_with(cid) for id, cid in enumerate(selected_clients)
        ]
        self.received_score = self.unpack(cpkqs)
        return self.received_score

    def communicate_score_with(self, client_id):
        svr_pkg = self.pack_model_range(client_id)
        # 
        return self.clients[client_id].reply_score(svr_pkg)

    def update_goodness_cached(self, goodness, selected_clients):
        for i, id in enumerate(selected_clients):
            self.goodness_cached[id] = goodness[i]

    def communicate_score(self, selected_clients):
        received_informations = self.communicate_score_with_client(selected_clients)
        # 
        list_histograms = received_informations["score"]
        list_max_scores = received_informations["max_score"]
        list_goodness = received_informations["goodness"]
        max_sc = max(list_max_scores)
        self.max_score = max(max_sc, self.max_score)
        # Save goodness score
        self.update_goodness_cached(list_goodness, selected_clients)
        removed_percentage = self.fuzzy_algo.run(
            {
                "current_round": self.current_round,
                "history_cached": self.check_nearly_participated(),
                "median_list": list_goodness,
            },
        )
        utils.fmodule.Sampler.set_ratio(1 - removed_percentage)
        aggregated_histogram = None
        if self.score_range > 0:
            aggregated_histogram = self.aggregate_histogram(list_histograms)

        else:
            self.score_range = max(list_max_scores) / self.option["bins"]

        return aggregated_histogram


import statistics


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
            return sum(list_) / len(list_)
        elif self.option["type_of_goodness"] == "median":
            return statistics.median(list_)

    def unpack_model_range(self, svr_pkg):
        return svr_pkg["model"], svr_pkg["score_range"]

    def build_histogram(self, list_score, score_range):
        n_bins = math.ceil(max(list_score) / score_range)
        bins = range(n_bins + 1) * np.array(score_range)
        historgram_ = np.histogram(list_score, bins)
        return historgram_

    def reply_score(self, svr_pkg):
        model, score_range = self.unpack_model_range(svr_pkg)
        self.score_range = score_range
        self.model = model
        histogram = None
        self.calculate_importance(copy.deepcopy(model))
        if not "score_list" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["score_list"] = {}
        utils.fmodule.LOG_DICT["score_list"][f"client_{self.id}"] = list(
            self.score_cached
        )
        if score_range != 0:
            histogram = self.build_histogram(self.score_cached, score_range)
        cpkg = self.pack_score(histogram)

        return cpkg

    def pack_score(self, histogram):
        if self.score_range == 0:
            return {
                "score": [],
                "max_score": max(self.score_cached),
                "goodness": self.cal_goodness(self.score_cached),
            }
        else:
            return {
                "score": histogram,
                "max_score": max(self.score_cached),
                "goodness": self.cal_goodness(self.score_cached),
            }

    def unpack_threshold(self, svr_pkg):
        return svr_pkg["threshold"], svr_pkg["use_upperbound_score"], svr_pkg["upperbound_score"]
    
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
        threshold,use_upperbound_score, upperbound_score = self.unpack_threshold(svr_pkg)
        if use_upperbound_score:
            selected_idx = utils.fmodule.Sampler.sample_upperbound(
            self.score_cached, threshold,upperbound_score
        )
        else:
            selected_idx = utils.fmodule.Sampler.sample_using_cached(
                self.score_cached, threshold
            )
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
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        
        for epoch in range(self.epochs):
            training_loss = 0
            for data, labels,idxs in self.data_loader:
                data, labels = data.float().to(self.device), labels.long().to(
                    self.device
                )
                optimizer.zero_grad()
                outputs = model(data)
                loss = self.calculator.criterion(outputs,labels)
                loss_proximal = 0
                for pm, ps in zip(model.parameters(), src_model.parameters()):
                    loss_proximal += torch.sum(torch.pow(pm - ps, 2))
                loss = loss + 0.5 * self.mu * loss_proximal
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            # training_loss.append(training_loss / len(current_dataloader))
            list_training_loss.append(training_loss / len(self.data_loader))
        if not "training_loss" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["training_loss"] = {}
        utils.fmodule.LOG_DICT["training_loss"][f"client_{self.id}"] = list_training_loss
        utils.fmodule.LOG_WANDB["mean_training_loss"] = sum(list_training_loss)/len(list_training_loss)
        return