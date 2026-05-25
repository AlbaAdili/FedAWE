import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from config import *
from dataset.dataset import *
from statistic.collect_stat import stats_collector_bar as stats_collector
from util.util import data_participation_each_node, data_participation_each_node_per_round, DatasetSplit, ClientSampler, accumulation
import numpy as np
import random
from model.model import Model


def varying_dynamics_awe():
    stat = stats_collector(prefix=prefix)
    
    for seed in seeds:

        random.seed(seed)
        np.random.seed(seed)  
        torch.manual_seed(seed) 
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  
            torch.backends.cudnn.deterministic = True  

        data_train, data_test = load_data(dataset, dataset_file_path, 'cpu')
        data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=0)
        data_test_loader = DataLoader(data_test, batch_size=256, num_workers=0)
        dict_users, participation_prob_each_node_init = data_participation_each_node(data_train, num_clients)        

        step_size_local = lr_local_init

        model = Model(seed, step_size_local, model_name=model_name, device=device, flatten_weight=True)

        train_loader_list = []
        dataiter_list = []

        for n in range(num_clients):
            train_loader_list.append(
                DataLoader(DatasetSplit(data_train, dict_users[n]), batch_size=batch_size_train, shuffle=True))
            dataiter_list.append(iter(train_loader_list[n]))


        def sample_minibatch(n):
            try:
                images, labels = next(dataiter_list[n])
                if len(images) < batch_size_train:
                    dataiter_list[n] = iter(train_loader_list[n])
                    images, labels = next(dataiter_list[n])
            except StopIteration:
                dataiter_list[n] = iter(train_loader_list[n])
                images, labels = next(dataiter_list[n])

            return images, labels

        
        w_global = model.get_weight()   
        w_local = torch.stack([w_global.detach().to('cpu') for i in range(num_clients)])

        rounds = 0
        not_participate_count_at_node = []

        for n in range(num_clients):
            not_participate_count_at_node.append(1)
   
        while rounds < total_rounds:

            participation_prob_each_node = data_participation_each_node_per_round(participation_prob_each_node_init, rounds)
      
            worker_samplers = []
            for n in range(num_clients):
                worker_samplers.append(ClientSampler(participation_prob_each_node[n]))

            step_size_local_round = step_size_local / np.sqrt(rounds/10+1) 
            model.update_learning_rate(step_size_local_round)

            participation = np.array([False for i in range(num_clients)])

            accumulated = 0.
            w_accumulate_local = 0.
            w_accumulate_delta = 0.
            
            # === 收集本轮数据的临时容器 ===
            local_deltas = []
            raw_multipliers = []
            client_indices = []

            for n in range(num_clients):
                worker_sampler = worker_samplers[n]

                if worker_sampler.sample():
                    participation[n] = True
                    model.assign_weight(w_local[n].to(device))
                    model.model.train()

                    for i in range(0, 10):
                        images, labels = sample_minibatch(n)
                        images, labels = images.to(device), labels.to(device)

                        if transform_train is not None:
                            images = transform_train(images).contiguous() 

                        model.optimizer.zero_grad()
                        output = model.model(images)
                        loss = model.loss_fn(output, labels)
                        loss.backward()                     
                        model.optimizer.step()
                
                    w_accumulate_local += w_local[n].to(device)
                    raw_delta = model.get_weight() - w_local[n].to(device)
                    
                    # 缓存本地计算出的梯度和对应的过期倍数
                    local_deltas.append(raw_delta)
                    raw_multipliers.append(float(not_participate_count_at_node[n]))
                    client_indices.append(n)

                    accumulated += 1
                    not_participate_count_at_node[n] = 1
                else:
                    participation[n] = False
                    not_participate_count_at_node[n] += 1

            # === 【核心修正防御逻辑：1.8倍中位数严格阻击闸门】 ===
            if accumulated > 0:
                # 1. 计算本轮所有上线节点梯度的 L2 模长
                norms = [torch.norm(d).item() for d in local_deltas]
                median_norm = np.median(norms) if len(norms) > 0 else 0.0
                
                # 2. 遍历检查每一个客户端，执行严格防御机制
                for idx, raw_delta in enumerate(local_deltas):
                    client_idx = client_indices[idx]
                    current_norm = norms[idx]
                    
                    # 【在这里！】如果该客户端带回来的更新幅度，超过了全网中位数的 1.8 倍
                    # 判定其大概率发生严重的局部过拟合与客户端漂移（Client Drift）
                    if current_norm > 1.8 * median_norm and raw_multipliers[idx] > 1.0:
                        # 剥夺其由于长期掉线获得的过期放大系数，强行退化回 1.0 的经典无偏 FedAvg 系数保护大盘
                        adaptive_multiplier = 1.0
                    else:
                        # 否则判定为在安全范围内的良性加速更新，正常继承原论文的放大红利
                        adaptive_multiplier = raw_multipliers[idx]
                        
                    w_accumulate_delta += raw_delta * adaptive_multiplier

                w_global = accumulation(w_accumulate_local, w_accumulate_delta, accumulated, device, lr_global)
                accumulated = 0

            for n in range(num_clients):
                if participation[n]:
                    w_local[n] =  w_global.detach().to('cpu') 

            rounds = rounds + 1

            if rounds % eval_freq == 0:
                stat.collect_stat_eval(seed, rounds, model, data_train_loader, data_test_loader, w_local, w_global)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()