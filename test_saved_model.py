import sys
import wandb
import torch
from torch.utils.data import DataLoader
from benchmark.cifar10_classification.core import *
from tqdm import tqdm
import importlib
import argparse
import json
import torch.nn as nn
from benchmark.cifar10_classification.model.resnet18 import Model
def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--data_path",type=str,default=None)
    parser.add_argument("--batch_size",type=int, default =32)
    parser.add_argument("--checkpoint_path",type=str,default="resnet18")
    parser.add_argument("--rm_id",type=int,default= 0)
    parser.add_argument("--model",default="resnet18")
    return parser.parse_args()

def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    # read configuration file
    seed_everything(42)
    args = read_option()

    transforms_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    

    # raw_train_dataset = Specific_dataset(root= args.data_path,train=True,transform=transforms_cifar10)
    # train_dataset = CustomDataset(raw_train_dataset,single_set["0"])

    test_dataset = datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=transforms_cifar10)

    # train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    device = torch.device("cuda")
    bmk_model_path = ".".join(["benchmark", "cifar10_classification", "model", args.model])
    model = getattr(importlib.import_module(bmk_model_path), "Model")().to(device)
    model_path = "Saved_Results/gauss_cifar10_iid_100client_1000data_2/fedalgo6base_ratio_0.8_C_0.3_test_saved_model.pt"
    # model_path = "Saved_Results/gauss_cifar10_iid_100client_1000data_2/fedavg_ratio_1.0_C_0.3_test_save_model.pt"
    model.load_state_dict(torch.load(model_path))

    softmax = nn.Softmax(-1)
    # 
    criterion = torch.nn.CrossEntropyLoss()
    list_gr = []
    list_pd = []
    testing_loss = 0
    list_testing_loss = []
    list_conf = []

    model.eval()
    with torch.no_grad():
        for data, labels in test_dataloader:
            data, labels = data.float().to(device), labels.long().to(
                device
            )
            outputs = model(data)
            list_gr += labels.tolist()
            list_conf.append(softmax(outputs).detach().cpu().numpy())
            list_pd += outputs.max(-1)[1].tolist()
            
            loss = criterion(outputs, labels)
            testing_loss += loss.item()
            
            # 
    conf_arr = np.concatenate(list_conf,0)
    testing_acc = 1.0 * sum(np.array(list_gr) == np.array(list_pd)) / len(list_gr)
    list_testing_loss.append(testing_loss / len(test_dataloader))
    print(f"Testing loss {testing_loss / len(test_dataloader)}  Testing acc {testing_acc}")
    list_conf_true = []
    # for i,true_label in enumerate(list_gr):
    #     if list_gr[i] == list_pd[i]:
    #         list_conf_true.append(conf_arr[i,true_label])
    # print(f"True predictions: {len(list_conf_true)}")
    # import matplotlib.pyplot as plt
    # plt.hist(list_conf_true,bins=100)
    # plt.ylim(0,4000)
    # plt.savefig("fedavg_hist.png")

    # for i,true_label in enumerate(list_gr):
    #     if list_gr[i] != list_pd[i]:
    #         list_conf_true.append(conf_arr[i,true_label])
    # print(f"True predictions: {len(list_conf_true)}")
    # import matplotlib.pyplot as plt
    # plt.hist(list_conf_true,bins=100)
    # plt.ylim(0,4000)
    # plt.savefig("fedalgo1_hist_wrong.png")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,5,sharex=True)

    for main_label in range(10):
        rows = main_label // 5
        cols = main_label % 5
        list_conf_true= []
        for i,true_label in enumerate(list_gr):
            if list_gr[i] == main_label:
                list_conf_true.append(conf_arr[i,true_label])
        print(f"Sample {len(list_conf_true)}")
        ax[rows,cols].set(xlabel=f"Label {main_label}")
        ax[rows,cols].hist(list_conf_true,bins=1000)
        print(f"True predictions: {len(list_conf_true)}")
    import matplotlib.pyplot as plt
    # plt.hist(list_conf_true,bins=100)
    # plt.ylim(0,4000)
    plt.savefig("fedalgo1_hist_across_label.png")
    # breakpoint()
