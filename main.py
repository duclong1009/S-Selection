import utils.fflow as flw
import torch
import wandb
def main():
    # read options
    option = flw.read_option()
    ss_name = f'{option["algorithm"]}_ratio_{option["ratio"]}_C_{option["proportion"]}_{option["session_name"]}'
    option["session_name"] = ss_name
    # set random seed
    if option["log_wandb"]:
        wandb.init(
            project="Pill_Sample_Selection_FL",
            entity="aiotlab",
            group= option["group_name"],
            name=f"{ss_name}",
            config=option,
        )
        
    flw.setup_seed(option['seed'])
    # initialize server, clients and fedtask
    server = flw.initialize(option)
    # start federated optimization
    try:
        server.run()
    except:
        # log the exception that happens during training-time
        flw.logger.exception("Exception Logged")
        raise RuntimeError

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    main()