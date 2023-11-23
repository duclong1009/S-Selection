CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.3 --algorithm KAKURENBO_sentall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score kakurenbo --log_wandb --pc_threshold 0.1 --adapt_ratio
CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.3 --algorithm KAKURENBO_notall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score kakurenbo --pc_threshold 0.1 --log_wandb
CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.3 --algorithm fedalgo1_kakurenbo --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score kakurenbo --pc_threshold 0.1


CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedalgo1_sentall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss --log_wandb 

CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedalgo1_sentall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss  
CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedalgo1_notall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss  


CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedprox_algo1_sentall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss --log_wandb 
CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedprox_algo1_notall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss --log_wandb 


CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedfa --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss --log_wandb 
CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedfa_algo1_notall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss --log_wandb 
CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedfa_algo1 --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss --log_wandb 
CUDA_VISIBLE_DEVICES=0 python main.py --num_rounds 1000 --session_name cluster_20_alpha_0_1_ --group_name NII_medium_pilldataset_100client_dirichlet --proportion 0.03 --algorithm fedfa_algo1_sentall --ratio 0.7 --aggregate weighted_com --task medium_pilldataset --num_classes 150 --data_path pill_dataset/medium_pilldataset --save_folder_path log_dir_me --idx_path pill_dataset/medium_pilldataset/100client/dirichlet/data_idx_alpha_0.1_cluster_20.json --score loss --log_wandb 
