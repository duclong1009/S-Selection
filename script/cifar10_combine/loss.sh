#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
module load python/3.10/3.10.4
python3 -m venv ~/venv/pytorch+horovod
source ~/venv/pytorch+horovod/bin/activate

python main.py --session_name "baseline_loss" --group_name "Cifar10_combine" --data_path cifar10_combine_noise/blurry_cifar10_iid_100client_1000data_combine1 --proportion 0.3 --algorithm algorithm/fedavg_v1_loss_based.py --score all_gnorm_threshold --ratio 1 --aggregate "weighted_com" --model resnet18 --task cifar10_classification --num_rounds 500