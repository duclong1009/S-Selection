# #!/bin/bash

# #$ -l rt_G.small=1
# #$ -l h_rt=24:00:00
# #$ -j y
# #$ -cwd

# source /etc/profile.d/modules.sh
# module load gcc/11.2.0
# module load cuda/11.5/11.5.2
# module load cudnn/8.3/8.3.3
# module load nccl/2.11/2.11.4-1
# module load python/3.10/3.10.4
# python3 -m venv ~/venv/pytorch+horovod
# source ~/venv/pytorch+horovod/bin/activate

python main.py --session_name "config2_inverse_test_threshold" --group_name "NIID_10Client_test" --idx_path Dataset_scenarios/pills/NIID_miidlerate_10clients.json --proportion 0.3 --algorithm fedalgo1 --score loss --ratio 1 --aggregate "weighted_com" --log_wandb --task medium_pilldataset --num_classes 150
# python main.py --session_name "config2_inverse_test_threshold" --group_name "NIID_10Client_test" --idx_path Dataset_scenarios/pills/NIID_miidlerate_10clients.json --proportion 0.3 --algorithm fedalgo1 --score loss --ratio 1 --aggregate "weighted_com" --log_wandb --task pill_classification --num_classes 77