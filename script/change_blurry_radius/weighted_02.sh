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

python main.py --session_name "weighted_sample_02" --group_name "Change_blurry_radius" --data_path cifar10/blurry_0.5_100_7 --proportion 0.3 --algorithm fedavg_weighted_each_sample --score all_gnorm_threshold --ratio 1 --aggregate "weighted_com" --model resnet18 --task cifar10_classification --num_rounds 500 --weight_for_noise 0.2
