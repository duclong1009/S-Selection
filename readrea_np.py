import numpy as np
import torch
import importlib
# from 
if True:
    epochs = 30
    saved_path =  f"centrailize/Without_rm_eval_loginterval_3"
    with open(f"{saved_path}/conf_arr_{epochs}.npy","rb") as f:
        arr = np.load(f)

    with open(f"{saved_path}/result_{epochs}.json","r") as f:
        import json
        result =json.load(f)

    list_score = result["list_score"]
    list_idx = result["list_idx"]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # 
    list_score_transformed = scaler.fit_transform(np.array(list_score).reshape(-1,1))
    list_score_transformed = list_score_transformed.squeeze().tolist()

    with open("cifar10/gauss_cifar10_iid_100client_1000data_2/data_idx.json","r") as f:
            import json
            data_idx = json.load(f)
    single_set = {"0":[]}
    for key in data_idx.keys():
        single_set["0"] += data_idx[key]

    # 
    ten = torch.tensor(arr)
    sf = torch.nn.Softmax(-1)(ten)
    conf_val,max_index = torch.max(sf,-1)

    from sklearn.cluster import KMeans
    com_arr = np.array([list_score,conf_val.tolist()])
    combined_arr = np.array([list_score_transformed,conf_val.tolist()])
    # 
    com_arr = com_arr.T
    combined_arr = combined_arr.T
    n_clusters = 10

    label = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(combined_arr)

    u_labels = np.unique(label)
    # 
    #plotting the results:
    import matplotlib.pyplot as plt 
    for i in u_labels:
        plt.scatter(com_arr[label == i , 0] , com_arr[label == i , 1] , label = i)
    plt.legend()
    plt.show()
    plt.savefig(f"{saved_path}/cluster_{epochs}.png")
    plt.cla()
    cluster_id = {}

    for i in u_labels:
        
        temp_idx = np.where(label ==i)[0].tolist()
        cluster_id[str(i)] = [single_set["0"][i] for i in temp_idx]
        print(len(np.where(label ==i)[0].tolist()))
    # 
    with open(f"{saved_path}/cluster_id_{epochs}.json","w") as f:
        json.dump(cluster_id,f)
    # 