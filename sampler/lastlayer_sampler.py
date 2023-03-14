from .threshold_sampler import Sampler as sl
import numpy as np
import torch
import math

class Sampler(sl):
    def __init__(self, sampler_config):
        super().__init__(sampler_config)

    def cal_score_lastlayer(self, dataset, model, criteria, device):
        model.eval()
        list_score = [[] for i in range(10)] 
        list_idx = [[] for i in range(10)] 
        optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
        model = model.to(device)
        list_loss = []
        list_output = []
        for idx, data in enumerate(dataset): 
            optimizer.zero_grad()
            x, y = data[0].unsqueeze(0).to(device), torch.tensor(
                data[1]).unsqueeze(0).to(device)
            y_pred = model(x)
            list_output.append(y_pred.cpu().detach().numpy())
            loss = criteria(y_pred, y)
            list_loss.append(loss.item())
            loss.backward()
            grad = model.fc.weight.grad.sum(-1)[y].item()

            list_idx[y.item()].append(idx)
            list_score[y.item()].append(grad)
        conf_array = np.concatenate(list_output,0)
        print(f"Data len {len(dataset)}")
        return list_score, list_idx, conf_array, list_loss
    
    def cal_score_lastlayer_abs(self, dataset, model, criteria, device):
        model.eval()
        list_score = [[] for i in range(10)] 
        list_idx = [[] for i in range(10)] 
        optimizer = torch.optim.SGD(model.parameters(),lr=1e-5)
        model = model.to(device)
        list_loss = []
        list_output = []
        for idx, data in enumerate(dataset): 
            optimizer.zero_grad()
            x, y = data[0].unsqueeze(0).to(device), torch.tensor(
                data[1]).unsqueeze(0).to(device)
            y_pred = model(x)
            list_output.append(y_pred.cpu().detach().numpy())
            loss = criteria(y_pred, y)
            list_loss.append(loss.item())
            loss.backward()
            grad = model.fc.weight.grad.sum(-1)[y].item()

            list_idx[y.item()].append(idx)
            list_score[y.item()].append(abs(grad))
        conf_array = np.concatenate(list_output,0)
        print(f"Data len {len(dataset)}")
        return list_score, list_idx, conf_array, list_loss


    def cal_threshold(
        self,
        histogram,
    ):
        list_threshold = []

        for label in range(10):
            list_score = histogram[label]
            n_samples = len(list_score)
            sorted_list = np.argsort(list_score)
            thresh_to_keep = int(self.sampler_config["ratio"] * n_samples)
            idx = sorted_list[-thresh_to_keep]
            list_threshold.append(list_score[idx])
        return list_threshold
    
    def sample_using_cached(self, cached_score, threshold, list_idx):
        selected_idx = []
        for label in range(10):
            l_idx = list_idx[label]
            score_list = cached_score[label]
            th = threshold[label]
            idx_ = np.array(l_idx)[
                np.where(np.array(score_list) > th)
            ]
            selected_idx += list(idx_)
        return selected_idx
