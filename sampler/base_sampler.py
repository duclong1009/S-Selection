from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import torch
import numpy as np


class _BaseSampler(object):
    def __init__(self,sampler_config):
        super(_BaseSampler, self).__init__()
        
        self.score = sampler_config["score"]
        self.threshold = sampler_config["threshold"]
        self.ratio = sampler_config["ratio"]
        self.use_sampler = sampler_config["use_sampler"]
        self.sampler_config = sampler_config
        
    def sample(self,):
        raise NotImplementedError
    
    def cal_gnorm(self,model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    def cal_gnorm_score_batch(self,batch,model,criteria,deivce):
        pass
    
    def cal_loss(self, dataset, model, criteria, device):
        # device = torch.device(device)
        # device = "cuda"
        optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
        list_score = []
        list_idx = []
        model = model.to(device)
        with torch.no_grad():
            for idx, data in enumerate(dataset):
                # breakpoint()
                optimizer.zero_grad()
                x, y = data[0].unsqueeze(0).to(device), torch.tensor(
                    data[1]).unsqueeze(0).to(device)
                y_pred = model(x)
                loss = criteria(y_pred, y)
                list_score.append(loss.cpu().item())
                list_idx.append(idx)
        return list_score, list_idx
    def cal_gnorm_model_weight(self, dataset, model, criteria, device):
        # device = torch.device(device)
        # device = "cuda"
        optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
        list_score = []
        list_idx = []
        model = model.to(device)
        
        for idx, data in enumerate(dataset):
            # breakpoint()
            optimizer.zero_grad()
            x, y = data[0].unsqueeze(0).to(device), torch.tensor(
                data[1]).unsqueeze(0).to(device)
            y_pred = model(x)
            loss = criteria(y_pred, y)
            loss.backward()
            score_ = self.cal_gnorm(model)
            list_score.append(score_)
            list_idx.append(idx)

        return list_score, list_idx
    
    def __cal_gnorm_last_layer(self, dataset, model, criteria, device):
        # device = torch.device(device)
        # device = "cuda"
        optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
        list_score = []
        list_idx = []
        model = model.to(device)
        # breakpoint()
        for idx, data in enumerate(dataset):
            optimizer.zero_grad()          
            x, y = data[0].unsqueeze(0).to(device), torch.tensor(data[1]).unsqueeze(0).to(device)
            y_pred = model(x)
            loss = criteria(y_pred, y)
            loss.backward()
            score_ = model.model.fc.weight.grad.detach().data.norm()
            list_score.append(score_.item())
            list_idx.append(idx)

        return list_score, list_idx
    
    def cal_threshold(self, histogram):
        pass


class Sampler(_BaseSampler):
    """ Calculate score each sample before training and select samples

    Args:
        _BaseSampler (_type_): _description_
    """

    def __init__(self,sampler_config):
        super(Sampler, self).__init__(sampler_config)
        # self.score = score
        # self.threshold = threshold
        # self.ratio = ratio
        # self.use_sampler = use_sampler
        


    def sample(self, dataset, model, criteria, device):
        if self.use_sampler:
            if self.score == "gnorm_ratio":
                assert self.ratio is not None, "Error: Ratio input is None"
                n_samples = int(len(dataset) * self.ratio)
                list_score, list_idx = self.__cal_gnorm_last_layer(
                    dataset, model, criteria, device)

                # sort list_idx based on its score
                sorted_list_idx = np.array(
                    list_idx)[np.argsort(np.array(list_score))[::-1]]
                selected_idx = sorted_list_idx[:n_samples]
                
            elif self.score == "gnorm_threshold":
                assert self.threshold is not None, "Error: threshold input is None"
                list_score, list_idx = self.__cal_gnorm_last_layer(
                    dataset, model, criteria, device)
                selected_idx = np.array(list_idx)[np.where(
                    np.array(list_score) > self.threshold)]
                
            elif self.score == "all_gnorm_ratio":
                assert self.ratio is not None, "Error: Ratio input is None"
                n_samples = int(len(dataset) * self.ratio)
                list_score, list_idx = self.cal_gnorm_model_weight(
                    dataset, model, criteria, device)

                # sort list_idx based on its score
                sorted_list_idx = np.array(
                    list_idx)[np.argsort(np.array(list_score))[::-1]]
                selected_idx = sorted_list_idx[:n_samples]
                
            elif self.score == "all_gnorm_threshold":
                assert self.threshold is not None, "Error: threshold input is None"
                list_score, list_idx = self.cal_gnorm_model_weight(
                    dataset, model, criteria, device)
                selected_idx = np.array(list_idx)[np.where(
                    np.array(list_score) > self.threshold)]
                
            elif self.score == "random":
                assert self.ratio is not None, "Error: Ratio input is None"
                n_samples = int(len(dataset) * self.ratio)
                selected_idx = np.random.choice(
                    range(len(dataset)), n_samples, replace=False)

            else:
                raise "Error: Not valid score mode"
        else:
            selected_idx = range(len(dataset))
        return selected_idx