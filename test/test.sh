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


python train_centrailize.py  --rm_id 0 --session_name "Rm_cluster_0" --idx_path cifar10/gauss_cifar10_iid_100client_1000data_2/data_idx.json --data_path cifar10/gauss_cifar10_iid_100client_1000data_2 --epochs 200