python main.py --num_rounds 500 --session_name "add_nosie4" --group_name "Impact_Noise" --proportion 0.3 --algorithm fedavg --ratio 1 --aggregate "weighted_com" --task medium_pilldataset --num_classes 150 --data_path ${DATA_DIR}  --save_folder_path ${LOG_DIR}--idx_path pill_dataset/medium_pilldataset/100client/clean_unequal_1_remove_4.json