<<<<<<< HEAD
#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=4:00:00
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

python main.py --session_name "" --group_name "Test_Fedavg" --idx_path Dataset_scenarios/NIID_10clients_random.json --proportion 0.5 --algorithm fedavg --score all_gnorm_threshold --ratio 1 --aggregate "weighted_com" --seed 5959
=======
CUDA_VISIBLE_DEVICES=1 python main.py --idx_path Dataset_scenarios/NIID_10clients_random.json --proportion 0.2 --algorithm fedalgo1_fixed --score all_gnorm_threshold --ratio 0.8 --log_wandb --noisy_rate_clients 0.3 0.2 0.2 0.1 0.1 0.1 0 0 0 0
>>>>>>> ebb8cbdc8daeef473f5c2ffe6ea809bbda9bdb0c
