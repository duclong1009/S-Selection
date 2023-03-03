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

python main.py --session_name "config4" --group_name "MNIST_pareto_1" --data_path mnist/MNIST-noniid-fedavg_pareto_1 --proportion 0.3 --algorithm fedalgo7 --score all_gnorm_threshold --ratio 1 --aggregate "weighted_com" --fuzzy_config_path "fuzzylogic_config/config4" --model cnn --task mnist_classification