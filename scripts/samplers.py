from loaddata import load_seq_RSNA, make_data
from mymodels import ResNet18_pt, VGG16_pt, VGG19_pt, EfficientNet_pt, DenseNet_pt, LeNet_pt

import torch
import torch.utils.data as data
import torch.nn.functional as F
import json
import os
import numpy as np
import random
from copy import deepcopy


# def get_uncertainty(prob):
#     return torch.sum(-prob * torch.log(prob), dim=1) # entropy, 越大不确定性越大

def diversity_sample(data, k, ignore=0):
    res = []
    n = data.shape[0]
    max_index = random.randint(0, n - 1)
    res.append(max_index)
    tmp_distance = np.sum((data - np.expand_dims(data[max_index], axis=0).repeat(n, axis=0))**2, axis=1)
    distance_to_center = tmp_distance
    k -= 1
    while ignore > 0:
        max_index = distance_to_center.argmax()
        tmp_distance = np.sum((data - np.expand_dims(data[max_index], axis=0).repeat(n, axis=0))**2, axis=1)
        distance_to_center = np.minimum(tmp_distance, distance_to_center)
        ignore -= 1
    while k > 0:
        max_index = distance_to_center.argmax()
        res.append(max_index)
        tmp_distance = np.sum((data - np.expand_dims(data[max_index], axis=0).repeat(n, axis=0))**2, axis=1)
        distance_to_center = np.minimum(tmp_distance, distance_to_center)
        k -= 1
    return np.array(res)


def uniform_sample(data, k, w):
    '''
        data为新来数据+coreset, k为coreset_size, w为 (累计数据量) / (累计数据量 + 新来数据量)
    '''
    n = data.shape[0]
    weights = [(1 - w) / (n - k)] * (n-k) + [w / k] * k
    return np.random.choice(n, size=k, replace=False, p=weights)

# def uncertainty_sample(data, uncertainty, sample_size, flag=False):
#     # 正确分类，uncertainty越高越选；错误分类，uncertainty越低越选
#     if not flag:
#         uncertainty = -uncertainty
#         left = np.mean(uncertainty)
#         right = 0
#     else:
#         left = 0
#         right = np.mean(uncertainty)
#     n, d = data.shape
#     N = int(np.log(n))
#     res = np.array([], dtype=np.int32)
#     for i in range(N):
        
#         idx = np.random.choice(np.where((uncertainty >= left) & (uncertainty < right))[0], int(sample_size / N))
#         res = np.concatenate((res, idx))
#         if not flag:
#             right = left
#             left *= 2
#         else:
#             left = right
#             right *= 2

#     return res


# def get_coreset(data, sample_size):
#     with torch.no_grad():
#         ebd_correct = []
#         ebd_wrong = []
#         uncertainty_correct = []
#         uncertainty_wrong = []
#         net_ebd = torch.nn.Sequential(*list(net.children())[:-1])
#         net_ebd.eval()
#         net.eval()
#         for i, data in enumerate(loader):
            
#             inputs, labels = data
#             inputs = inputs.to(device)
#             outputs = net(inputs).cpu().squeeze()
#             prob = F.softmax(outputs, dim=1)
#             uncertain = get_uncertainty(prob).numpy()

#             ebd_all = net_ebd(inputs)
            
#             ebd_all = ebd_all.cpu().squeeze()

#             for j in range(len(labels)):
#                 if(torch.argmax(prob[j]) == labels[j]):
#                     ebd_correct.append(ebd_all[j].numpy())
#                     # ebd_correct.append(net.feature)
#                     uncertainty_correct.append(uncertain[j])
#                 else:
#                     ebd_wrong.append(ebd_all[j].numpy())
#                     # ebd_correct.append(net.feature)
#                     uncertainty_wrong.append(uncertain[j])
        
#     ebd_all = np.array(ebd_correct + ebd_wrong)
#     ebd_wrong = np.array(ebd_wrong)
#     ebd_correct = np.array(ebd_correct)
#     idx = diversity_sample(ebd_all, sample_size / 2)
#     coreset = ebd_all[idx]
#     sample_size -= sample_size / 2
#     idx = uncertainty_sample(ebd_wrong, np.array(uncertainty_wrong), sample_size / 2)
#     coreset = np.vstack((coreset, ebd_wrong[idx]))
#     sample_size -= sample_size / 2
#     idx = uncertainty_sample(ebd_correct, np.array(uncertainty_correct), sample_size, True)
#     print(idx)
#     coreset = np.vstack((coreset, ebd_correct[idx]))
#     return coreset

# coreset = get_coreset(ctn_data, 128)

def get_ebd_byclass(net, loader, opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['GPUs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ebd = [[] for _ in range(opt['class_num'])]
    with torch.no_grad():
        net.eval()
        for data in loader: 
            inputs, labels, weight = data
            inputs = inputs.to(device)
            ebd_all, _ = net(inputs)
            ebd_all = ebd_all.cpu().squeeze()
            for i in range(len(labels)):
                ebd[labels[i]].append(ebd_all[i].numpy())
    return ebd


def merge(ds, coreset, opt, weight=1):
    if coreset is None:
        return deepcopy(ds)
    
    dataset = deepcopy(ds)
    for j in range(opt['class_num']):
        dataset.tensor_data = np.concatenate((dataset.tensor_data, coreset[j]))
        dataset.tensor_targets = torch.cat((dataset.tensor_targets, (torch.ones(len(coreset[j]), dtype=int) * j)), 0)
        dataset.weight = torch.cat((dataset.weight, torch.ones(len(coreset[j]))*weight), 0)
    dataset.update_classw()

    # for j in range(opt['class_num']):
    #     merge_data = np.concatenate((dataset.tensor_data[dataset.tensor_targets == j], coreset[j]))
    #     merge_weight = torch.cat((torch.ones(len(dataset.tensor_data[dataset.tensor_targets == j])), torch.ones(len(coreset[j]))*weight), 0)
    #     if not j:
    #         tot_data = merge_data
    #         tot_targets = torch.zeros(len(merge_data), dtype=int)
    #         tot_weight = merge_weight
    #     else:
    #         tot_data = np.concatenate((tot_data, merge_data))
    #         tot_targets = torch.cat((tot_targets, (torch.ones(len(merge_data), dtype=int) * j)), 0)
    #         tot_weight = torch.cat((tot_weight, merge_weight), 0)    
    return dataset


def Gonzalez(net, dataset, opt, tag):
    coreset = []
    loader = data.DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=4)
    ebd = get_ebd_byclass(net, loader, opt)
    for j in range(opt['class_num']):
        ebd[j] = np.array(ebd[j])
        if tag == 'train_mem':
            if ebd[j].shape[0] <= opt['coreset_size']:
                coreset.append(dataset.tensor_data[dataset.tensor_targets == j])
                continue
            idx = diversity_sample(ebd[j], opt['coreset_size'], ignore=int(opt['ignore']*len(ebd[j])))
        elif tag == 'val_mem':
            if ebd[j].shape[0] <= opt['val_coreset_size']:
                coreset.append(dataset.tensor_data[dataset.tensor_targets == j])
                continue
            idx = diversity_sample(ebd[j], opt['val_coreset_size'], ignore=4)
        coreset.append(dataset.tensor_data[dataset.tensor_targets == j][idx])

    return coreset


def get_Gonzalez(net, memory, opt):
    coreset = []
    ans = 0
    for j in range(opt['class_num']):
        merge_data = memory[j]
        if not j:
            tot_data = merge_data
            tot_targets = torch.zeros(len(merge_data), dtype=int)
        else:
            tot_data = np.concatenate((tot_data, merge_data))
            tot_targets = torch.cat((tot_targets, (torch.ones(len(merge_data), dtype=int) * j)), 0)
    dataset = make_data(tot_data, tot_targets, None)

    loader = data.DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=4)
    ebd = get_ebd_byclass(net, loader, opt)
    for j in range(opt['class_num']):
        ebd[j] = np.array(ebd[j])
        if ebd[j].shape[0] <= opt['coreset_size']:
            coreset.append(dataset.tensor_data[dataset.tensor_targets == j])
            ans += len(dataset.tensor_data[dataset.tensor_targets == j])
            continue
        idx = diversity_sample(ebd[j], opt['coreset_size'], ignore=int(opt['ignore']*len(ebd[j])))
        ans += opt['coreset_size']
        coreset.append(dataset.tensor_data[dataset.tensor_targets == j][idx])

    return coreset, ans


def get_Uniform(net, dataset, cnt, opt, id, tag="train"):
    memory = []
    loader = data.DataLoader(dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=4)
    ebd = get_ebd_byclass(net, loader, opt)
    if tag == "val":
        size = opt['val_coreset_size']
    else:
        size = opt['mem_size']

    for j in range(opt['class_num']):
        ebd[j] = np.array(ebd[j])

        if ebd[j].shape[0] <= size:
            memory.append(dataset.tensor_data[dataset.tensor_targets == j])
            cnt[j] += ebd[j].shape[0]
            continue

        if cnt[j] <= size:
            tmp = size
            cnt[j] = ebd[j].shape[0]

        elif id:
            tmp = cnt[j]
            cnt[j] += ebd[j].shape[0] - size
    
        else:
            print("Wrong!!!")

        idx = uniform_sample(ebd[j], size, tmp / cnt[j])        
        memory.append(dataset.tensor_data[dataset.tensor_targets == j][idx])
    return memory