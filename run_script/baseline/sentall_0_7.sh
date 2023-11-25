#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -o /home/aaa10078nj/Federated_Learning/Long_SampleSelection/logs/baseline/$JOB_NAME_$JOB_ID.log
#$ -j y

source /etc/profile.d/modules.sh
#module load gcc/11.2.0
#Old gcc. Newest support is 12.2.0. See module avail
LD_LIBRARY_PATH=/apps/centos7/gcc/11.2.0/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/gcc/11.2.0/bin:${PATH}
#module load openmpi/4.1.3
#Old mpi. Use intel mpi instead
LD_LIBRARY_PATH=/apps/centos7/openmpi/4.1.3/gcc11.2.0/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/openmpi/4.1.3/gcc11.2.0/bin:${PATH}
module load cuda/11.5/11.5.2
module load cudnn/8.3/8.3.3
module load nccl/2.11/2.11.4-1
#module load python/3.10/3.10.4
#Old python. Newest support is 10.3.10.10. See module avail
LD_LIBRARY_PATH=/apps/centos7/python/3.10.4/lib:${LD_LIBRARY_PATH}
PATH=/apps/centos7/python/3.10.4/bin:${PATH}
​
source ~/venv/pytorch1.11+horovod/bin/activate
python --version
LOG_DIR="/home/aaa10078nj/Federated_Learning/Long_SampleSelection/logs/baseline/$JOB_NAME_$JOB_ID"
rm -r ${LOG_DIR}
mkdir ${LOG_DIR}

#Dataset
DATA_DIR="$SGE_LOCALDIR/$JOB_ID/"
cp -r /home/aaa10078nj/Federated_Learning/Long_SampleSelection/SampleSelection_easyFL/pill_dataset/medium_pilldataset ${DATA_DIR}

cd SampleSelection_easyFL
python main.py --num_rounds 1000 --session_name "cluster_20_alpha_0_1_" --group_name "NII_medium_pilldataset_100client_dirichlet" --proportion 0.3 --algorithm fedalgo1_sentall --ratio 0.7 --aggregate "weighted_com" --task medium_pilldataset --num_classes 150 --data_path ${DATA_DIR} --save_folder_path ${LOG_DIR} --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score all_gnorm_threshold