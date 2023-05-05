import json
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.autograd.functional as F
import torchvision.models as models
from torch.utils.data.dataset import Dataset
from functorch import make_functional_with_buffers, vmap, grad
import os
import numpy as np
from copy import deepcopy
import time
import random
import gc

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

class SimpleDT(Dataset):
    def __init__(self, dt):
       self.dt = dt

    def __getitem__(self, index):
        return (self.dt[index][0], self.dt[index][1], self.dt[index][2])

    def __len__(self):
        return len(self.dt)

def cac_grad(net, criterion, eles, times):
    train_loader = data.DataLoader(SimpleDT(eles), batch_size=opt['batch_size'], shuffle=False, num_workers=4)
    net.zero_grad()
    for i, ddata in enumerate(train_loader):
        inputs, labels, _ = ddata
        inputs, labels = inputs.to(device), labels.to(device)
        _, outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

    ans = np.array([x.grad.cpu() * times for x in net.parameters()])
    net.zero_grad()
    return ans
    

class ReservoirSampling:
    def __init__(self, k):
        self.k = k # 蓄水池的容量
        self.count = 0 # 已经接收的元素个数
        self.reservoir = [] # 蓄水池
        self.idx = np.arange(opt['batch_size'])

    def sample(self, dataset):
        for i in range(len(dataset)):
            element = dataset[i] # tuple(x, y, w)
            self.count += 1
            if len(self.reservoir) < self.k: # 蓄水池未满，直接加入
                self.reservoir.append(element)
            else: # 蓄水池已满，以k/count的概率替换
                j = random.randint(0, self.count - 1)
                if j < self.k:
                    self.reservoir[j] = element
    
    def update_idx(self):
        self.idx = np.random.choice(self.k, opt['batch_size'], replace=False)
    
    def get_batch_reservoir(self):
        return np.take(self.reservoir, self.idx, axis=0)
    
    def empty(self):
        return self.count < self.k

Memory = ReservoirSampling(opt['mem_size'])
val_Memory = ReservoirSampling(opt['val_coreset_size'])

class ToT_grad:
    def __init__(self):
        self.count = 0
        self.history_grad = None
        self.record = None
        self.best = None

    def update_history(self, net, criterion, idx):
        tdata = train_data[idx]
        train_loader = data.DataLoader(tdata, batch_size=opt['batch_size'], shuffle=False, num_workers=4)
        net.zero_grad()
        net.train()
        m = 0
        for i, ddata in enumerate(train_loader):
            inputs, labels, _ = ddata
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            m += 1

        curgrad = np.array([x.grad.cpu() for x in net.parameters()])
        if self.history_grad is None:
            self.history_grad = curgrad / m
        else:
            self.history_grad += curgrad / m
    
    def update_record(self, g):
        self.count += 1
        if self.record is None:
            self.record = deepcopy(g)
        else:
            self.record += g

    def update_epoch(self):
        if self.record is not None:
            self.history_grad += self.record / self.count
            self.count = 0
            self.record = None

    def update_best(self):
        self.best = deepcopy(self.history_grad)

    def recover_from_best(self):
        self.history_grad = self.best

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


def train(idx, criterion, val_loader, net, mode="train", tolerance=400):
    lastnet = deepcopy(net)
    tdata = train_data[idx]
    if mode == "train":
        optimizer = optim.SGD(net.parameters(), lr=opt['lr'] / (idx + 1))
        schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        epochs = opt['epochs']
    else:
        optimizer = optim.SGD(net.parameters(), lr=opt['finetune_lr'] / (idx + 1))
        schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
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

            # 利用Memory和history_grad按照SVRG的思想也进行一步梯度下降
            if not Memory.empty():
                now = 0
                Memory.update_idx()
                new_grad = np.array([x.grad.cpu() for x in net.parameters()])

                cur_grad = cac_grad(net, criterion, Memory.get_batch_reservoir(), idx)
                
                last_grad = cac_grad(lastnet, criterion, Memory.get_batch_reservoir(), idx)
                
                Tot_grad.update_record(cur_grad - last_grad)

                cur_grad += (Tot_grad.get_history() - last_grad) * opt['alpha']

                for param in net.parameters():
                    param.grad += cur_grad[now].to(device) + new_grad[now].to(device)
                    now += 1

            optimizer.step()

        Tot_grad.update_epoch()    

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
                    Tot_grad.update_best()
                if epoch % 10 == 9:
                    print('[Seq %d, Epoch %d] val accuracy: %.2f (%d / %d), avg loss: %.2f .' % (idx+1, epoch+1, val_accuracy, correct, summ, loss100 / cnt))
                if tol >= tolerance:
                        print("!!!Early Stop!!!")
                        break
        
        lastnet = deepcopy(net)
        schedule.step()

    return best_model


def SVRG():
    print("--------Train with SVRG replay--------")
    net = ResNet18_pt(opt['class_num']).to(device)
    val_coreset = None
    best_prev = 0.0
    cnt_val = [0, 0]
    F = []
    Intrans = []
    class_num = torch.zeros(opt['class_num'])
    start_time = time.time()
    for i in range(opt['day']):
        # 确定val_loader不变
        # new_val_data = val_data[i]
        # new_val_data = merge(val_data[i], val_coreset, opt, 1)
        # val_Memory.sample(val_data[i])
        new_val_loader = data.DataLoader(val_data[i], batch_size=opt['batch_size'], shuffle=False, num_workers=4)
        # new_val_loader = data.DataLoader(SimpleDT(val_Memory.reservoir), batch_size=opt['batch_size'], shuffle=False, num_workers=4)

        if opt['loss'] == 'balanced':
            class_num += train_data[i].class_num
            criterion = nn.CrossEntropyLoss(weight=(1 - class_num / sum(class_num)).to(device))
            # criterion = nn.CrossEntropyLoss()
        best_model = train(i, criterion, new_val_loader, net, "train")
        net.load_state_dict(best_model)
        Tot_grad.recover_from_best()

        # 记录net的test_acc, 计算Last Forgetting
        cur_acc = model_test(net)
        best_prev = max(best_prev, cur_acc)
        print("Current test acc : %.3f, Best previous test acc : %.3f" % (cur_acc, best_prev))
        F.append(best_prev - cur_acc)
        Intrans.append(opt['non_CIL'] - cur_acc)

        # 更新Memory, history_grad和val_coreset
        Memory.sample(train_data[i])
        print("!sample done!")
        Tot_grad.update_history(net, criterion, i)
        print("!update history done!")
        # val_coreset = get_Uniform(net, new_val_data, cnt_val, opt, i, "val")
    
    end_time = time.time()
    print("Training time: %.3f" % (end_time - start_time))
    print("Last Forgetting : %.3f" % (sum(F)/len(F)))
    print("Last Intransigence : %.3f" % (sum(Intrans)/len(Intrans)))
    eva_and_save_model(net, '-SVRG')


if __name__ == '__main__':
    SVRG()