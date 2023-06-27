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

python main.py --num_rounds 500 --session_name "clean_dataset" --group_name "Impact_Noise" --proportion 0.3 --algorithm fedavg --ratio 1 --aggregate "weighted_com" --task medium_pilldataset --num_classes 150 --data_path ${DATA_DIR}  --save_folder_path ${LOG_DIR}--idx_path pill_dataset/medium_pilldataset/100client/clean_unequal_1.json