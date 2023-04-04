from loaddata import load_seq_RSNA
from mymodels import ResNet18_pt, VGG16_pt, VGG19_pt, EfficientNet_pt, DenseNet_pt, LeNet_pt

import torch
import torch.utils.data as data
import torch.nn.functional as F
import json
import os
import numpy as np
import random


# with open("../opt-json/RSNA-binary.json") as f:
#     opt = json.load(f)

# ctn_data = load_seq_RSNA(data_info_path=opt['data_info_path'], data_path=opt['data_path'],
#                                modify_size=opt['modify_size'], train='play', day=opt['day'], reshuffle=False)

# loader = data.DataLoader(ctn_data, batch_size=opt['batch_size'], shuffle=True)

# os.environ["CUDA_VISIBLE_DEVICES"] = opt['GPUs']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# if opt['model'] == 'resnet18-pt':
#     net = ResNet18_pt(opt['class_num']).to(device)
# elif opt['model'] == 'resnet18':
#     net = ResNet18(opt['class_num']).to(device)
# elif opt['model'] == 'vgg16-pt':
#     net = VGG16_pt(opt['class_num']).to(device)
# elif opt['model'] == 'efficient-pt':
#     net = EfficientNet_pt(opt['class_num']).to(device)
# elif opt['model'] == 'densenet-pt':
#     net = DenseNet_pt(opt['class_num']).to(device)
# elif opt['model'] == 'vgg19-pt':
#     net = VGG19_pt(opt['class_num']).to(device)
# elif opt['model'] == 'lenet-pt':
#     net = LeNet_pt(opt['class_num']).to(device)


# net.load_state_dict(torch.load('../model/resnet18-pt-balanced-96', map_location=device))


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
            inputs, labels = data
            inputs = inputs.to(device)
            ebd_all, _ = net(inputs)
            ebd_all = ebd_all.cpu().squeeze()
            for i in range(len(labels)):
                ebd[labels[i]].append(ebd_all[i].numpy())
    return ebd