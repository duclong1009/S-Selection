"""
This is a non-official implementation of Scaffold proposed in 'Stochastic
Controlled Averaging for Federated Learning' (ICML 2020).
"""
from torch.utils.data import DataLoader

import utils
from .fedbase import BasicServer, BasicClient
import copy
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None,device="cpu"):
        super(Server, self).__init__(option, model, clients, test_data,device)
        self.init_algo_para({'eta':1.0})
        self.cg = self.model.zeros_like()

    def pack_model(self, client_id):
        return {
            "model": copy.deepcopy(self.model),
            "cg": self.cg,
        }

    def iterate(self):
        # sample clients
        self.selected_clients = self.sample()
        # local training
        res = self.communicate(self.selected_clients)
        dys, dcs = res['dy'], res['dc']
        # aggregate
        # breakpoint()
        self.model, self.cg = self.aggregate(dys, dcs)

        return

    def aggregate(self, dys, dcs):
        # x <-- x + eta_g * dx = x + eta_g * average(dys)
        # c <-- c + |S|/N * dc = c + |S|/N * average(dcs)
        dx = fmodule._model_average(dys)
        dc = fmodule._model_average(dcs)
        new_model = self.model + self.eta * dx
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None,device="cpu"):
        super(Client, self).__init__(option, name, train_data, valid_data,device)
        self.c = None

    @fmodule.with_multi_gpus
    def train(self, model, cg):
        """
        The codes of Algorithm 1 that updates the control variate
          12:  ci+ <-- ci - c + 1 / K / eta_l * (x - yi)
          13:  communicate (dy, dc) <-- (yi - x, ci+ - ci)
          14:  ci <-- ci+
        Our implementation for efficiency
          dy = yi - x
          dc <-- ci+ - ci = -1/K/eta_l * (yi - x) - c = -1 / K /eta_l *dy - c
          ci <-- ci+ = ci + dc
          communicate (dy, dc)
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
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        cg.freeze_grad()
        if self.c is None: self.c = model.zeros_like()
        self.c.freeze_grad()

        optimizer = self.calculator.get_optimizer(
            model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )

        list_training_loss = []
        for epoch in range(self.epochs):
            training_loss = 0
            for data, labels, idxs in self.data_loader:
                data, labels = data.float().to(self.device), labels.long().to(
                    self.device
                )
                optimizer.zero_grad()
                outputs = model(data)
                loss = self.calculator.criterion(outputs, labels)
                loss.backward()
                for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()):
                    pm.grad = pm.grad - pc + pcg
                optimizer.step()
                training_loss += loss.item()
            # training_loss.append(training_loss / len(current_dataloader))
            list_training_loss.append(training_loss / len(self.data_loader))
        if not "training_loss" in utils.fmodule.LOG_DICT.keys():
            utils.fmodule.LOG_DICT["training_loss"] = {}
        utils.fmodule.LOG_DICT["training_loss"][
            f"client_{self.id}"
        ] = list_training_loss
        utils.fmodule.LOG_WANDB["mean_training_loss"] = sum(list_training_loss) / len(
            list_training_loss
        )
        
        dy = model - src_model
        dc = -1.0 / (self.num_steps * self.learning_rate) * dy - cg
        self.c = self.c + dc
        return dy, dc

    def reply(self, svr_pkg):
        model, c_g = self.unpack_model(svr_pkg)
        dy, dc = self.train(model, c_g)
        cpkg = self.pack(dy, dc)
        return cpkg

    def pack(self, dy, dc):
        return {
            "dy": dy,
            "dc": dc,
        }

    def unpack_model(self, received_pkg):
        # unpack the received package
        return received_pkg['model'], received_pkg['cg']
