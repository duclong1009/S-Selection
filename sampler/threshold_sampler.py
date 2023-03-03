from .base_sampler import _BaseSampler
import numpy as np
import torch


class Sampler(_BaseSampler):
    def __init__(self, sampler_config):
        super().__init__(sampler_config)

    def get_information(self, dataset, model, criteria, device):
            if self.use_sampler:
                if self.score == "all_gnorm_threshold":
                    assert self.threshold is not None, "Error: threshold input is None"
                    list_score, list_idx,list_conf,list_prediction = self.get_infor(
                        dataset, model, criteria, device
                    )
                    sorted_idx = np.array(list_idx)
                    sorted_score = np.array(list_score)
                elif self.score == "loss":
                    list_score, list_idx = self.cal_loss(
                        dataset, model, criteria, device
                    )
                    sorted_idx = np.array(list_idx)
                    sorted_score = np.array(list_score)
                else:
                    raise "Not correct score type"
            return sorted_idx, sorted_score, list_conf, list_prediction

    def get_infor(self, dataset, model, criteria, device):
        # device = torch.device(device)
        # device = "cuda"
        optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
        list_score = [0] * len(dataset)
        list_idx = []
        list_conf = []
        model = model.to(device)
        list_prediction = []

        with torch.no_grad():
            for idx, data in enumerate(dataset):
                # 
                optimizer.zero_grad()
                x, y = data[0].unsqueeze(0).to(device), torch.tensor(
                    data[1]).unsqueeze(0).to(device)
                y_pred = model(x)
                list_conf.append(torch.softmax(y_pred,1).max().item())
                # loss = criteria(y_pred, y)
                # loss.backward()
                # score_ = self.cal_gnorm(model)
                # list_score.append(score_)
                list_idx.append(idx)
                list_prediction.append(torch.argmax(y_pred).item())
        return list_score, list_idx,list_conf,list_prediction
    
    def cal_cof(self, dataset, model, criteria, device):
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset,batch_size=64,shuffle=False)
        list_idx = range(len(dataset))
        list_conf = []
        model = model.to(device)

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                # 
                # 
                x, y = data[0].to(device), torch.tensor(
                    data[1]).to(device)
                # 
                y_pred = model(x)
                list_conf +=torch.softmax(y_pred,1).max(1)[0].tolist()
        return  list_idx,list_conf


    def cal_score(self, dataset, model, criteria, device):
        if self.use_sampler:
            if self.score == "all_gnorm_threshold":
                assert self.threshold is not None, "Error: threshold input is None"
                list_score, list_idx = self.cal_gnorm_model_weight(
                    dataset, model, criteria, device
                )
                sorted_idx = np.array(list_idx)
                sorted_score = np.array(list_score)
            elif self.score == "loss":
                list_score, list_idx = self.cal_loss(
                    dataset, model, criteria, device
                )
                sorted_idx = np.array(list_idx)
                sorted_score = np.array(list_score)
            else:
                raise "Not correct score type"
        return sorted_idx, sorted_score

    def sample(self, dataset, model, criteria, device):
        if self.use_sampler:
            if self.score == "all_gnorm_threshold":
                assert self.threshold is not None, "Error: threshold input is None"
                list_score, list_idx = self.cal_gnorm_model_weight(
                    dataset, model, criteria, device
                )
                selected_idx = np.array(list_idx)[
                    np.where(np.array(list_score) > self.threshold)
                ]
        else:
            selected_idx = np.array(range(len(dataset)))
        return selected_idx

    def sample_using_cached(self, cached_score, threshold):
        list_idx = range(len(cached_score))
        if threshold > 0:
            selected_idx = np.array(list_idx)[
                np.where(np.array(cached_score) > threshold)
            ]
        else:
            selected_idx = np.array(list_idx)
        return selected_idx

    def sample_upperbound(self, cached_score, threshold, upperbound_score):
        list_idx = range(len(cached_score))
        # 
        if threshold > 0:
            selected_idx = np.array(list_idx)[
                np.where((np.array(cached_score) >= threshold)
                & (np.array(cached_score) <= upperbound_score))
            ]
        else:
            selected_idx = np.array(list_idx)[np.where(np.array(cached_score) <= upperbound_score)]
        return selected_idx

    def sample_lower_using_cached(self, cached_score, threshold):
        list_idx = range(len(cached_score))
        selected_idx = np.array(list_idx)[np.where(np.array(cached_score) <= threshold)]
        return selected_idx

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_ratio(self, ratio):
        self.sampler_config["ratio"] = ratio

    def cal_upper_threshold(
        self,
        histogram,
    ):
        print("Using upperbound score")
        list_n_, value_list = histogram
        total_samples = sum(list_n_)
        thresh_list = value_list[:-1]

        thresh_to_keep = int(self.sampler_config["ratio"] * total_samples)
        check_ = 0
        # list_n_increase =
        for i, val in enumerate(list_n_):
            check_ += val
            if check_ >= thresh_to_keep:
                thresh_value = thresh_list[i]
                break
        return thresh_value

    def cal_threshold(
        self,
        histogram,
    ):
        list_n_, value_list = histogram
        total_samples = sum(list_n_)
        thresh_list = value_list[:-1]
        if self.sampler_config["ratio"] == 1:
            return 0
        thresh_to_keep = int(self.sampler_config["ratio"] * total_samples)
        check_ = 0
        # list_n_increase =
        for i, val in enumerate(list_n_[::-1]):
            check_ += val
            if check_ >= thresh_to_keep:
                thresh_value = thresh_list[::-1][i]
                break
        return thresh_value
