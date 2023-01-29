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

python main.py --session_name "nopretrained" --group_name "NIID_100client_noisy" --idx_path Dataset_scenarios/NIID_100clients_random.json --proportion 0.2 --algorithm fedalgo1 --score all_gnorm_threshold --ratio 0.8 --noisy_rate_clients 0.3 0.2 0.2 0.1 0.1 0.1 0 0 0 0  --model resnet18_wopretrained