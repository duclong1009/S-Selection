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

CUDA_VISIBLE_DEVICES=0 python main.py --session_name "config2_inverse_test_threshold" --group_name "NIID_10Client_test" --idx_path Dataset_scenarios/pills/NIID_miidlerate_10clients.json --proportion 0.3 --algorithm fedalgo1_dev --score all_gnorm_threshold --ratio 0.8 --aggregate "weighted_com" --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path 'test' --log_wandb
