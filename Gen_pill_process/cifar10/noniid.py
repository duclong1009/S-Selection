import json
import numpy as np


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def gen_niid_random_samples(n_clients,clean_dict, noisy_dict  ,noisy_rate):
    # 
    # assert sum(noisy_rate) == 1, "sum of noisy rate must equal 1"
    chunk_size = 20
    clean_chunks= int(len(clean_dict)/chunk_size)
    noisy_chunks = int(len(noisy_dict)/chunk_size)
    # n_chunks = 1232
    
    min_c = 15
    max_c = 35
    key = True
    
    print(f"Len clean chunks: {clean_chunks}")
    print(f"Len noisy chunks: {noisy_chunks}")
    while(key):
        try:
            clean_list_chunks = []
            noisy_list_chunks = []
            noisy_chunks_per_client = []
            for i in range(clean_chunks):
                # 
                clean_list_chunks.append(list(clean_dict.items())[chunk_size * i :chunk_size * (i+1)])
            for i in range(noisy_chunks):
                noisy_list_chunks.append(list(noisy_dict.items())[chunk_size * i :chunk_size * (i+1)])

            #init dictionary 
            idx_4client_dict = {}
            for client in range(n_clients):
                noisy_chunks_per_client.append(int(noisy_chunks * noisy_rate[client]))
                idx_4client_dict[client] = []
            
            chunk_clean_idx_list = range(len(clean_list_chunks))
            chunk_noisy_idx_list = range(len(noisy_list_chunks))
            for client in range(n_clients):
                n_c = np.random.randint(min_c, max_c)
                if n_c > len(chunk_clean_idx_list):
                    print("New session")
                    raise ValueError("")
                chunk_idx = np.random.choice(chunk_clean_idx_list, n_c, replace=False)
                chunk_clean_idx_list = list((set(chunk_clean_idx_list) - set(chunk_idx)))
                # print(f" =={n_c} == {len(chunk_clean_idx_list)}")
                for id_c in chunk_idx:
                    idx_4client_dict[client] +=[kd[1] for kd in  clean_list_chunks[id_c]]
                    
            for client in range(n_clients):
                chunk_idx = np.random.choice(chunk_noisy_idx_list, noisy_chunks_per_client[client], replace=False)
                chunk_noisy_idx_list = list((set(chunk_noisy_idx_list) - set(chunk_idx)))
                for id_c in chunk_idx:
                    idx_4client_dict[client] += [kd[1] for kd in  noisy_list_chunks[id_c]]
            key = False
        except:
            pass
    return idx_4client_dict


def gen_niid_random_samples2(n_clients,clean_dict, noisy_dict  ,noisy_rate):
    # 
    # assert sum(noisy_rate) == 1, "sum of noisy rate must equal 1"
    """_summary_

    Args:
        n_clients (_type_): _description_
        clean_dict (_type_): _description_
        noisy_dict (_type_): _description_
        noisy_rate (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    chunk_size = 25
    noisy_size = 30
    clean_chunks= int(len(clean_dict)/chunk_size)
    noisy_chunks = int(len(noisy_dict)/noisy_size)
    # n_chunks = 1232
    total_chunks = clean_chunks
    min_c = 10
    max_c = 25
   
    key= True
    print(f"Len clean chunks: {clean_chunks}")
    print(f"Len noisy chunks: {noisy_chunks}")
    while(key):
        # try:
            key1 = True
            while(key1):
                assigned_samples_list = []
                for client in range(n_clients):
                    assigned_samples_list.append(np.random.randint(min_c, max_c))
                if sum(assigned_samples_list) <= total_chunks:
                    key1=False
            print(f"{sum(assigned_samples_list)}/{total_chunks}")
            # 
            clean_list_chunks = []
            noisy_list_chunks = []
            noisy_chunks_per_client = []
            clean_chunks_per_client = [0] * n_clinets
            
            for i in range(clean_chunks):
                # 
                clean_list_chunks.append(list(clean_dict.items())[chunk_size * i :chunk_size * (i+1)])
            for i in range(noisy_chunks):
                noisy_list_chunks.append(list(noisy_dict.items())[noisy_size * i :noisy_size * (i+1)])
            
            chunk_clean_idx_list = range(len(clean_list_chunks))
            chunk_noisy_idx_list = range(len(noisy_list_chunks))
            idx_4client_dict = {}
            for client in range(n_clients):
                noisy_chunks_per_client.append(int(noisy_chunks * noisy_rate[client]))
                idx_4client_dict[client] = []
            print(noisy_chunks_per_client, assigned_samples_list)
            for client in range(n_clients):
                clean_chunks_per_client[client] = assigned_samples_list[client] - noisy_chunks_per_client[client]
                
            for client in range(n_clients):
                chunk_idx = np.random.choice(chunk_clean_idx_list, clean_chunks_per_client[client], replace=False)
                chunk_clean_idx_list = list((set(chunk_clean_idx_list) - set(chunk_idx)))
                for id_c in chunk_idx:
                    idx_4client_dict[client] +=[kd[1] for kd in  clean_list_chunks[id_c]]
                    
            for client in range(n_clients):
                chunk_idx = np.random.choice(chunk_noisy_idx_list, noisy_chunks_per_client[client], replace=False)
                chunk_noisy_idx_list = list((set(chunk_noisy_idx_list) - set(chunk_idx)))
                for id_c in chunk_idx:
                    idx_4client_dict[client] += [kd[1] for kd in  noisy_list_chunks[id_c]]
            key = False
        # except:
        #     pass
    return idx_4client_dict

def gen_niid_random_samples3(n_clients,clean_dict, noisy_dict  ,noisy_rate):
    # 
    # assert sum(noisy_rate) == 1, "sum of noisy rate must equal 1"
    """_summary_

    Args:
        n_clients (_type_): _description_
        clean_dict (_type_): _description_
        noisy_dict (_type_): _description_
        noisy_rate (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    id_path_dict = {}
    # 
    for (k,v) in clean_dict.items():
        id_path_dict[v] = k
        
    for k,v in noisy_dict.items():
        id_path_dict[v] = k
        
    chunk_size = 25
    noisy_size = 30
    clean_chunks= int(len(clean_dict)/chunk_size)
    noisy_chunks = int(len(noisy_dict)/noisy_size)
    # n_chunks = 1232
    total_chunks = clean_chunks
    min_c = 10
    max_c = 25

    key= True
    print(f"Len clean chunks: {clean_chunks}")
    print(f"Len noisy chunks: {noisy_chunks}")
    while(key):
        # try:
            key1 = True
            while(key1):
                assigned_samples_list = []
                for client in range(n_clients):
                    assigned_samples_list.append(np.random.randint(min_c, max_c))
                if sum(assigned_samples_list) <= total_chunks:
                    key1=False
            print(f"{sum(assigned_samples_list)}/{total_chunks}")
            # 
            list_noisy_img = []
            for k,v in noisy_dict.items():
                list_noisy_img.append(k)
            clean_list_chunks = []
            noisy_list_chunks = []
            noisy_samples_per_client = [0] * n_clinets
            
            for i in range(clean_chunks):
                # 
                clean_list_chunks.append(list(clean_dict.items())[chunk_size * i :chunk_size * (i+1)])
            
            chunk_clean_idx_list = range(len(clean_list_chunks))
            idx_4client_dict = {}
            # 
            for client in range(n_clients):
                
                noisy_samples_per_client[client] = int(assigned_samples_list[client] * noisy_rate[client] * chunk_size)
                idx_4client_dict[client] = []
                
            for client in range(n_clients):
                chunk_idx = np.random.choice(chunk_clean_idx_list, assigned_samples_list[client], replace=False)
                chunk_clean_idx_list = list((set(chunk_clean_idx_list) - set(chunk_idx)))
                
                for id_c in chunk_idx:
                    idx_4client_dict[client] +=[kd[1] for kd in  clean_list_chunks[id_c]]
                    
            res_n_samples = []
            for client in range(n_clients):
                removed_id = np.random.choice(idx_4client_dict[client], noisy_samples_per_client[client],replace=False)
                idx_4client_dict[client] = list(set(idx_4client_dict[client]) - set(removed_id))
                
                for id in removed_id:
                    path_ = id_path_dict[id]
                    class_ = path_.split('/')[-3]
                    # 
                    avai_list = [k for k in list_noisy_img if class_ in k][0]
                    list_noisy_img.remove(avai_list)
                    swap_id = noisy_dict[avai_list]
                    idx_4client_dict[client].append(swap_id)
                
                res_n_samples.append(len(idx_4client_dict[client]))
            print(np.array(assigned_samples_list)*chunk_size- np.array(res_n_samples))
            key = False
        # except:
            # pass
    return idx_4client_dict


if __name__ == '__main__':
    with open("train_dataset_samples.txt", "r") as f:
        train_s = f.read()
        # 
        train_samples = [data.split("~")[0]
                         for i, data in enumerate(train_s.split("\n"))]
    
    with open("Gen_pill_process/dict/clean_dict.json","r") as f:
        clean_dict = json.load(f)
    with open("Gen_pill_process/dict/noisy_dict.json","r") as f:
        noisy_dict = json.load(f)
    # -------CONFIG------------------
    seed = 10
    n_clinets = 10
    noisy_rate = [0.3, 0.25, 0.25, 0.2, 0.1, 0.1, 0, 0 , 0 , 0]
    # noisy_rate = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15, 0, 0 , 0 , 0]
    file_name = f"NIID_miidlerate_{n_clinets}clients.json"
    idx_4client = gen_niid_random_samples3(n_clinets,clean_dict, noisy_dict, noisy_rate)
    import os
    save_path = f"Dataset_scenarios"

    # seed_everything(seed)
    # -------GENDATA------------------
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(f"{save_path}/{file_name}", "w") as f:
        json.dump(idx_4client, f)