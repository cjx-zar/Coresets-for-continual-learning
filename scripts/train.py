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

from loaddata import load_seq_RSNA, make_data
from mymodels import ResNet18_pt, ResNet18_npt, VGG16_pt, VGG19_pt, EfficientNet_pt, DenseNet_pt, LeNet_pt
from samplers import diversity_sample, get_ebd_byclass, uniform_sample, merge, get_Gonzalez, get_Uniform


class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, pt, target):
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        # probs[probs < 0.3] = 1 - probs[probs < 0.3]
        probs[probs < 0.2] = 0.8
        log_p = probs.log()
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


with open("../opt-json/RSNA-binary.json") as f:
    opt = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = opt['GPUs']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, val_data = load_seq_RSNA(data_info_path=opt['data_info_path'], data_path=opt['data_path'],
                            modify_size=opt['modify_size'], train='train', day=opt['day'], reshuffle=False, design=opt['design'])

test_data = load_seq_RSNA(data_info_path=opt['data_info_path'], data_path=opt['data_path'],
                              modify_size=opt['modify_size'], train='test', reshuffle=False, design=opt['design'])
test_loader = data.DataLoader(test_data[0], batch_size=opt['batch_size'], shuffle=False, num_workers=4)

def init_train():
    if opt['model'] == 'resnet18-pt':
        net = ResNet18_pt(opt['class_num']).to(device)
    elif opt['model'] == 'vgg16-pt':
        net = VGG16_pt(opt['class_num']).to(device)
    elif opt['model'] == 'vgg19-pt':
        net = VGG19_pt(opt['class_num']).to(device)
    elif opt['model'] == 'densenet-pt':
        net = DenseNet_pt(opt['class_num']).to(device)
    elif opt['model'] == 'resnet18':
        net = ResNet18_npt(opt['class_num']).to(device)
    # elif opt['model'] == 'lenet-pt':
    #     net = LeNet_pt(opt['class_num']).to(device)
    # elif opt['model'] == 'efficient-pt':
    #     net = EfficientNet_pt(opt['class_num']).to(device)

    return net


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


def update_weight(net, criterion, tot_dataset, new_datalen):
    # 算coreset和train_data[i]的loss
    loss = None
    loader = data.DataLoader(tot_dataset, batch_size=opt['batch_size'], shuffle=False, num_workers=4)
    with torch.no_grad(): 
        for i, ddata in enumerate(loader):
            # 测试模式
            net.eval()
            inputs, labels, weights = ddata
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            _, outputs = net(inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            if loss is None:
                loss = criterion(log_probs, labels).cpu()
            else:
                loss = torch.cat((loss, criterion(log_probs, labels).cpu()), 0)
    # 根据loss重新分配权重
        new_w = deepcopy(tot_dataset.weight)
        new_data_cntw = 0.0
        new_data_cntloss = 0.0
        for j in range(new_datalen):
            new_data_cntw += tot_dataset.weight[j]
            new_data_cntloss += loss[j]
        for j in range(new_datalen):
            for k in range(new_datalen, len(tot_dataset)):
                new_w[j] += loss[j] / (new_data_cntloss + loss[k]) * (new_data_cntw + tot_dataset.weight[k])
        
        for k in range(new_datalen, len(tot_dataset)):
            new_w[k] += loss[k] / (new_data_cntloss + loss[k]) * (new_data_cntw + tot_dataset.weight[k])
        new_w = new_w * 2 / (len(tot_dataset) - new_datalen + 3)
        return new_w


def train(idx, train_data, memory, val_loader, net, mode="train", tolerance=4):
    if mode == "train":
        optimizer = optim.SGD(net.parameters(), lr=opt['lr'])
        # optimizer = optim.Adagrad(net.parameters(), lr=opt['lr'])
        epochs = opt['epochs']
    else:
        optimizer = optim.SGD(net.parameters(), lr=opt['finetune_lr'])
        # optimizer = optim.Adagrad(net.parameters(), lr=opt['finetune_lr'])
        epochs = opt['finetune_epochs']

    if memory is not None:
        coreset, coreset_len = get_Gonzalez(net, memory, opt)
        new_train_data = merge(train_data, coreset, opt, len(train_data)/coreset_len)
    else:
        new_train_data = deepcopy(train_data)
    if opt['loss'] == 'balanced':
        criterion = nn.NLLLoss(weight=new_train_data.class_weight.to(device), reduction='none')
        # criterion = nn.CrossEntropyLoss(weight=new_train_data.class_weight.to(device))
    elif opt['loss'] == 'focal':
        criterion = MultiCEFocalLoss(opt['class_num'], 1, new_train_data.class_weight.to(device), 'none')
    train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=4)

    best_acc = 0.0
    last_acc = 0.0
    tol = 0
    for epoch in range(epochs):
        loss100 = 0.0
        cnt = 0
        for i, ddata in enumerate(train_loader):
            net.train()

            inputs, labels, weights = ddata
            inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            _, outputs = net(inputs)

            if opt['loss'] == 'balanced':
                log_probs = F.log_softmax(outputs, dim=1) # NLLLoss
            elif opt['loss'] == 'focal':
                log_probs = F.softmax(outputs, dim=1) # focal
            
            loss = criterion(log_probs, labels)
            weighted_loss = loss * weights
            final_loss = weighted_loss.mean() 
            final_loss.backward()
            optimizer.step()
            loss100 += final_loss.item()
            cnt += 1
        
        if epoch % opt['reweight'] == opt['reweight']-1 and memory is not None:
            new_weight = update_weight(net, criterion, new_train_data, len(train_data))
            # 生成新的train_loader
            new_train_data.weight = new_weight
            train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=4)

        if epoch % opt['new_replay'] == opt['new_replay']-1 and memory is not None and epoch != epochs - 1:
            # 从memory中更新coreset
            coreset, coreset_len = get_Gonzalez(net, memory, opt)
            new_train_data = merge(train_data, coreset, opt, len(train_data)/coreset_len)
            train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=4)

        if epoch % 5 == 4:      
            # 验证集
            with torch.no_grad(): 
                correct = 0
                summ = 0
                for i, ddata in enumerate(val_loader):

                    # 测试模式
                    net.eval()

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


def forgetting_train():
    if opt['day'] == 1:
        print("--------Train with all data--------")
    else:
        print("--------Train with " + str(opt['day']) + " Seqs data with no memory--------") 
    net = init_train()

    best_prev = 0.0
    F = []
    Intrans = []
    start_time = time.time()
    for i in range(opt['day']):
        # 确定val_loader不变
        new_val_data = val_data[i]
        new_val_loader = data.DataLoader(new_val_data, batch_size=opt['batch_size'], shuffle=False, num_workers=4)

        best_model = train(i, train_data[i], None, new_val_loader, net, "train")
        net.load_state_dict(best_model)
        # 记录net的test_acc, 计算Last Forgetting
        cur_acc = model_test(net)
        best_prev = max(best_prev, cur_acc)
        print("Current test acc : %.3f, Best previous test acc : %.3f" % (cur_acc, best_prev))
        F.append(best_prev - cur_acc)
        Intrans.append(opt['non_CIL'] - cur_acc)
    
    end_time = time.time()
    print("Training time: %.3f" % (end_time - start_time))
    print("Last Forgetting : %.3f" % (sum(F)/len(F)))
    print("Last Intransigence : %.3f" % (sum(Intrans)/len(Intrans)))
    
    # eva_and_save_model(net, '-forget')


def continual_train_withCandN():
    print("--------Train with Gonzalez Coreset and new data--------")
    net = init_train()

    memory = None
    val_coreset = None
    best_prev = 0.0
    cnt = [0, 0]
    cnt_val = [0, 0]
    F = []
    Intrans = []
    start_time = time.time()
    for i in range(opt['day']):
        # 确定val_loader不变
        # new_val_data = val_data[i]
        new_val_data = merge(val_data[i], val_coreset, opt, 1)
        new_val_loader = data.DataLoader(new_val_data, batch_size=opt['batch_size'], shuffle=False, num_workers=4)

        best_model = train(i, train_data[i], memory, new_val_loader, net, "train")
        net.load_state_dict(best_model)
        # 记录net的test_acc, 计算Last Forgetting
        cur_acc = model_test(net)
        best_prev = max(best_prev, cur_acc)
        print("Current test acc : %.3f, Best previous test acc : %.3f" % (cur_acc, best_prev))
        F.append(best_prev - cur_acc)
        Intrans.append(opt['non_CIL'] - cur_acc)

        # 更新memory和val_coreset
        memory = get_Uniform(net, merge(train_data[i], memory, opt, 1), cnt, opt, i)
        val_coreset = get_Uniform(net, new_val_data, cnt_val, opt, i, "val")
    
    end_time = time.time()
    print("Training time: %.3f" % (end_time - start_time))
    print("Last Forgetting : %.3f" % (sum(F)/len(F)))
    print("Last Intransigence : %.3f" % (sum(Intrans)/len(Intrans)))
    eva_and_save_model(net, '-c+n')

if __name__ == '__main__':
    # forgetting_train()
    continual_train_withCandN()
