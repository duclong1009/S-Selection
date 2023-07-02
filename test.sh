
DATA_DIR="pill_dataset/medium_pilldataset"
LOG_DIR="Test"
python main.py --num_rounds 1000 --session_name "cluster_20_alpha_0_1_" --group_name "NII_medium_pilldataset_100client_dirichlet" --proportion 0.3 --algorithm fedfa_algo1 --ratio 0.8 --aggregate "weighted_com" --task medium_pilldataset --num_classes 150 --data_path ${DATA_DIR} --save_folder_path ${LOG_DIR} --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score all_gnorm_threshold --log_wandb