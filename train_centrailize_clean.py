import wandb
import torch
from torch.utils.data import DataLoader
from benchmark.cifar10_classification.core import *
from tqdm import tqdm
import importlib
import argparse
import json
def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", action="store_false")
    parser.add_argument("--session_name", type=str, default=None)
    parser.add_argument("--idx_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--data_path",type=str,default=None)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--lr",type=float, default=0.01)
    parser.add_argument("--batch_size",type=int, default =32)
    parser.add_argument("--model",type=str,default="resnet18")
    parser.add_argument("--rm_id",type=int,default= 0)
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

    if args.log_wandb:
        wandb.init(
            project="Abnormal_Data_FL",
            entity="aiotlab",
            group=f"Centrailize",
            name=args.session_name,
            config=args,
        )
    with open("centrailize/Without_rm_eval_loginterval_3/cluster_id_3.json","r") as f:
        rm_dict = json.load(f)

    transforms_cifar10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    with open(args.idx_path,"r") as f:
        import json
        data_idx = json.load(f)
    single_set = {"0":[]}
    
    for key in data_idx.keys():
        single_set["0"] += data_idx[key]
    args.rm_id = None

    if args.rm_id == None:
        pass
    else:
        rm_list = rm_dict[str(args.rm_id)]
        res_list = [i for i in single_set["0"] if i not in rm_list]
        single_set["0"] = res_list
    raw_train_dataset = Specific_dataset(root= args.data_path,train=True,transform=transforms_cifar10)
    # train_dataset = CustomDataset(raw_train_dataset,single_set["0"])
    train_dataset = datasets.CIFAR10("./data/cifar10/", train=True, download=True, transform=transforms_cifar10)
    test_dataset = datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=transforms_cifar10)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    device = torch.device("cuda")
    bmk_model_path = ".".join(["benchmark", "cifar10_classification", "model", args.model])
    model = getattr(importlib.import_module(bmk_model_path), "Model")().to(device)
    # 
    optimizer = torch.optim.SGD(params=model.parameters(),lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    list_training_loss=[]
    list_testing_loss = []
    for epoch in range(1,args.epochs+1):
        model.train()
        training_loss = 0
        for data, labels in tqdm(train_dataloader):
            data, labels = data.float().to(device), labels.long().to(
                device
            )
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        list_training_loss.append(training_loss / len(train_dataloader))
        testing_loss =0
        list_gr = []
        list_pd = []

        model.eval()
        with torch.no_grad():
            for data, labels in test_dataloader:
                data, labels = data.float().to(device), labels.long().to(
                    device
                )
                outputs = model(data)
                list_gr += labels.tolist()
                list_pd += outputs.max(-1)[1].tolist()
                loss = criterion(outputs, labels)
                testing_loss += loss.item() 
                # 
        testing_acc = 1.0 * sum(np.array(list_gr) == np.array(list_pd)) / len(list_gr)
        list_testing_loss.append(testing_loss / len(test_dataloader))
        print(f"Training loss: {training_loss / len(train_dataloader)}   Testing loss {testing_loss / len(test_dataloader)}  Testing acc {testing_acc}")
        if args.log_wandb:
            wandb.log({
                "Training loss": training_loss / len(train_dataloader),
                "Testing loss": testing_loss / len(test_dataloader),
                "Testing acc": testing_acc
            })

    dataset = train_dataset
    list_score = []
    list_idx = []
    list_conf = []
    list_prediction = []


    def cal_gnorm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    for idx, data in enumerate(dataset):
        optimizer.zero_grad()
        x, y = data[0].unsqueeze(0).to(device), torch.tensor(
            data[1]).unsqueeze(0).to(device)
        y_pred = model(x)
        list_conf.append(y_pred.cpu().detach().numpy())
        loss = criterion(y_pred, y)
        loss.backward()
        score_ = cal_gnorm(model)
        list_score.append(score_)
        list_idx.append(idx)
        list_prediction.append(torch.argmax(y_pred).item())
        # if idx == 10:
        #     break
    conf_array = np.concatenate(list_conf,0)
    
    saved_dict = {
        "list_score": list_score,
        "list_idx": single_set["0"],
        "list_predict": list_prediction,
    }
    import os
    # breakpoint)
    if not os.path.exists(f"centrailize/{args.session_name}"):
        os.makedirs(f"centrailize/{args.session_name}")
    with open(f"centrailize/{args.session_name}/conf_arr.npy","wb") as f:
        np.save(f, conf_array)

    with open(f"centrailize/{args.session_name}/result.json","w") as f:
        json.dump(saved_dict,f)
    torch.save(model.state_dict(),f"centrailize/{args.session_name}/model.pt")