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

python main.py --session_name "gauss_loss" --group_name "gauss_CIFAR10_20client_pareto" --data_path cifar10/gauss_CIFAR10_20client_pareto --proportion 0.3 --algorithm fedalgo6base --score loss --ratio 0.8 --aggregate "weighted_com" --fuzzy_config_path "fuzzylogic_config/config1" --model resnet18 --task cifar10_classification