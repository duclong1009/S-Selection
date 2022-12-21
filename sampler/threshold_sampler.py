
from .base_sampler import _BaseSampler
import numpy as np
import torch


class Sampler(_BaseSampler):
    def __init__(self, sampler_config):
        super().__init__(sampler_config)
        
    def cal_score(self,dataset, model, criteria, device):
        if self.use_sampler:
            if self.score == "all_gnorm_threshold":
                assert self.threshold is not None, "Error: threshold input is None"
                list_score, list_idx = self.cal_gnorm_model_weight(
                    dataset, model, criteria, device)
                sorted_idx = np.array(list_idx)
                sorted_score =  np.array(list_score)
            else:
                raise "Not correct score type"
        return sorted_idx, sorted_score
        
    def sample(self,dataset, model, criteria, device):
        if self.use_sampler:
            if self.score == "all_gnorm_threshold":
                assert self.threshold is not None, "Error: threshold input is None"
                list_score, list_idx = self.cal_gnorm_model_weight(
                    dataset, model, criteria, device)
                selected_idx = np.array(list_idx)[np.where(
                    np.array(list_score) > self.threshold)]
        else:
            selected_idx = np.array(range(len(dataset)))
        return selected_idx
    
    def sample_using_cached(self,cached_score,threshold):
        list_idx = range(len(cached_score))
        if threshold > 0:
            selected_idx = np.array(list_idx)[np.where(
                        np.array(cached_score) > threshold)]
        else: 
            selected_idx = np.array(list_idx)
        return selected_idx
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def cal_threshold(self, histogram):
        list_n_ , value_list = histogram
        total_samples = sum(list_n_)
        thresh_list = value_list[:-1]
        thresh_ = int(self.sampler_config["ratio"] * total_samples)
        check_ = 0
        for i, val in enumerate(list_n_[::-1]):
            check_ += val
            if check_ >= thresh_:
                thresh_value = thresh_list[::-1][i]
                break
        return thresh_value
