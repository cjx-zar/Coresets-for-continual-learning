import json
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import os
import numpy as np
from copy import deepcopy
import time
import random

from loaddata import load_seq_RSNA
from mymodels import ResNet18_pt
from samplers import merge, get_Uniform


with open("../opt-json/SVRG.json") as f:
    opt = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = opt['GPUs']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, val_data = load_seq_RSNA(data_info_path=opt['data_info_path'], data_path=opt['data_path'],
                            modify_size=opt['modify_size'], train='train', day=opt['day'], reshuffle=False, design=opt['design'])

test_data = load_seq_RSNA(data_info_path=opt['data_info_path'], data_path=opt['data_path'],
                              modify_size=opt['modify_size'], train='test', reshuffle=False, design=opt['design'])
test_loader = data.DataLoader(test_data[0], batch_size=opt['batch_size'], shuffle=False, num_workers=4)

def cac_grad(net, criterion, eles):
    # 返回平均grad
    if isinstance(eles, list):
        k = len(eles)
    else:
        eles = [eles]
        k = 1
    net.zero_grad()
    net.train()
    for ele in eles:
        inputs, labels, _ = ele
        if inputs.dim() == 3:
            inputs = torch.unsqueeze(inputs, dim=0)
        if labels.dim() == 0:
            labels = torch.as_tensor([labels])
        inputs, labels = inputs.to(device), labels.to(device)
        _, outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

    grad = np.array([x.grad.cpu() / k for x in net.parameters() if x.grad is not None])
    return grad

class ReservoirSampling:
    def __init__(self, k):
        self.k = k # 蓄水池的容量
        self.count = 0 # 已经接收的元素个数
        self.reservoir = [] # 蓄水池
        self.grads = [] # 每个池中元素的梯度信息(numpy)
        self.avggrad = None

    def update_avggrad(self, x, plus=True):
        if self.avggrad is None:
            self.avggrad = x / self.k
        else:
            if plus:
                self.avggrad += x / self.k
            else:
                self.avggrad -= x / self.k

    def sample(self, net, criterion, idx):
        dataset = train_data[idx]
        for i in range(len(dataset)):
            element = dataset[i] # tuple(x, y, w)
            self.count += 1
            if len(self.reservoir) < self.k: # 蓄水池未满，直接加入
                self.reservoir.append(element)
                newgrad = cac_grad(net, criterion, element)
                self.grads.append(newgrad)
                self.update_avggrad(newgrad)
            else: # 蓄水池已满，以k/count的概率替换
                j = random.randint(0, self.count - 1)
                if j < self.k:
                    self.reservoir[j] = element
                    newgrad = cac_grad(net, criterion, element)
                    self.update_avggrad(newgrad)
                    self.update_avggrad(self.grads[j], False)
                    self.grads[j] = newgrad

    def get_reservoir(self):
        return self.reservoir
    
    def get_grad(self):
        return self.avggrad
    
    def empty(self):
        return self.count < self.k

Memory = ReservoirSampling(opt['mem_size'])

class ToT_grad:
    def __init__(self):
        self.count = 0 # 已经见过的元素个数
        self.history_grad = None

    def update_history(self, net, criterion, idx): # 做成类，动态平均
        tdata = train_data[idx]
        m = len(tdata)
        if self.history_grad is not None:
            self.history_grad *= (self.count / (self.count + m))

        train_loader = data.DataLoader(tdata, batch_size=opt['batch_size'], shuffle=True, num_workers=4)
        criterion_sum = deepcopy(criterion)
        criterion_sum.reduction = 'sum'
        net.zero_grad()
        net.train()
        for i, ddata in enumerate(train_loader):
            inputs, labels, _ = ddata
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = net(inputs)
            loss = criterion_sum(outputs, labels)
            loss.backward()

        curgrad = np.array([x.grad.cpu() for x in net.parameters() if x.grad is not None])
        if self.history_grad is None:
            self.history_grad = curgrad / (self.count + m)
        else:
            self.history_grad += curgrad / (self.count + m)
        self.count += m
    
    def get_history(self):
        return self.history_grad
    
Tot_grad = ToT_grad()


def model_test(net):
    summ = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for i, data in enumerate(test_loader):
            inputs, labels, weight = data
            inputs = inputs.to(device)

            _, validation_output = net(inputs)
            val_y = torch.max(validation_output, 1)[1].cpu().squeeze()
            correct += (val_y == labels).sum().item()
            summ += len(val_y)
    val_accuracy = float(correct / summ)
    print('test accuracy: %.3f (%d / %d).' % (val_accuracy, correct, summ))
    return val_accuracy


def model_val(net):
    summ = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for i in range(opt['day']):
            loader = data.DataLoader(val_data[i], batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            for j, ddata in enumerate(loader):
                inputs, labels, weight = ddata
                inputs = inputs.to(device)

                _, validation_output = net(inputs)
                val_y = torch.max(validation_output, 1)[1].cpu().squeeze()
                correct += (val_y == labels).sum().item()
                summ += len(val_y)
    val_accuracy = float(correct / summ)
    print('val accuracy: %.3f (%d / %d).' % (val_accuracy, correct, summ))
    return val_accuracy


def eva_and_save_model(net, method):
    test_acc = model_test(net)
    forget_acc = model_val(net)
    test_acc = ('%.2f'%test_acc).split('.')[-1]
    forget_acc = ('%.2f'%forget_acc).split('.')[-1]
    acc_info = '-' + test_acc + '-' + forget_acc

    if opt['design']:
        borib = '-ib'
    else:
        borib = '-b'

    # torch.save(net.state_dict(), '../model/' + opt['model'] + '-' + str(opt['coreset_size']) + '/' + opt['loss'] + '-seq' + str(opt['day']) + acc_info + method + borib + '-t')


def train(idx, criterion, val_loader, net, mode="train", tolerance=4):
    tdata = train_data[idx]
    if mode == "train":
        optimizer = optim.SGD(net.parameters(), lr=opt['lr'])
        epochs = opt['epochs']
    else:
        optimizer = optim.SGD(net.parameters(), lr=opt['finetune_lr'])
        epochs = opt['finetune_epochs']
    train_loader = data.DataLoader(tdata, batch_size=opt['batch_size'], shuffle=True, num_workers=4)
    best_acc = 0.0
    last_acc = 0.0
    tol = 0
    for epoch in range(epochs):
        loss100 = 0.0
        cnt = 0
        net.train()
        for i, ddata in enumerate(train_loader):
            inputs, labels, weights = ddata
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            _, outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            loss100 += loss.item()
            cnt += 1

            # 利用Memory和history_grad按照SVRG的思想也进行一步梯度下降(要不要设计频率?)
            # 求memory中数据在当前模型下的梯度
            if not Memory.empty():
                now = 0
                cur_grad = cac_grad(net, criterion, Memory.get_reservoir())
                cur_grad += Tot_grad.get_history() - Memory.get_grad()
                for param in net.parameters():
                    if param.grad is None: continue
                    param.grad += cur_grad[now].to(device)
                    now += 1
            optimizer.step()

        if epoch % 5 == 4:      
            # 验证集
            net.eval()
            with torch.no_grad(): 
                correct = 0
                summ = 0
                for i, ddata in enumerate(val_loader):
                    inputs, labels, weights = ddata
                    inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)

                    _, validation_output = net(inputs)
                    val_y = torch.max(validation_output, 1)[1].cpu().squeeze()
                    for i in range(len(labels)):
                        if val_y[i] == labels[i]:
                            correct += 1
                        summ += 1
                val_accuracy = float(correct / summ)
                if last_acc > val_accuracy:
                    tol += 1
                else:
                    tol = 0
                last_acc = val_accuracy
                if best_acc < val_accuracy:
                    best_acc = val_accuracy
                    best_model = deepcopy(net.state_dict())
                if epoch % 10 == 9:
                    print('[Seq %d, Epoch %d] val accuracy: %.2f (%d / %d), avg loss: %.2f .' % (idx+1, epoch+1, val_accuracy, correct, summ, loss100 / cnt))
                if tol >= tolerance:
                        print("!!!Early Stop!!!")
                        break

    return best_model


def SVRG():
    print("--------Train with SVRG replay--------")
    net = ResNet18_pt(opt['class_num']).to(device)
    val_coreset = None
    best_prev = 0.0
    cnt_val = [0, 0]
    F = []
    Intrans = []
    start_time = time.time()
    for i in range(opt['day']):
        # 确定val_loader不变
        # new_val_data = val_data[i]
        new_val_data = merge(val_data[i], val_coreset, opt, 1)
        new_val_loader = data.DataLoader(new_val_data, batch_size=opt['batch_size'], shuffle=False, num_workers=4)

        if opt['loss'] == 'balanced':
            criterion = nn.CrossEntropyLoss(weight=train_data[i].class_weight.to(device))
        best_model = train(i, criterion, new_val_loader, net, "train")
        net.load_state_dict(best_model)
        # 记录net的test_acc, 计算Last Forgetting
        cur_acc = model_test(net)
        best_prev = max(best_prev, cur_acc)
        print("Current test acc : %.3f, Best previous test acc : %.3f" % (cur_acc, best_prev))
        F.append(best_prev - cur_acc)
        Intrans.append(opt['non_CIL'] - cur_acc)

        # 更新Memory, history_grad和val_coreset
        Memory.sample(net, criterion, i)
        Tot_grad.update_history(net, criterion, i)
        val_coreset = get_Uniform(net, new_val_data, cnt_val, opt, i, "val")
    
    end_time = time.time()
    print("Training time: %.3f" % (end_time - start_time))
    print("Last Forgetting : %.3f" % (sum(F)/len(F)))
    print("Last Intransigence : %.3f" % (sum(Intrans)/len(Intrans)))
    eva_and_save_model(net, '-SVRG')


if __name__ == '__main__':
    SVRG()