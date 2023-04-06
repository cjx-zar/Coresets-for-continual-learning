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

from loaddata import load_seq_RSNA, make_data
from mymodels import ResNet18_pt, ResNet18_npt, VGG16_pt, VGG19_pt, EfficientNet_pt, DenseNet_pt, LeNet_pt
from samplers import diversity_sample, get_ebd_byclass, uniform_sample


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        # input: N x C
        # target: N
        # get probability of true class
        prob = F.softmax(input, dim=1)
        prob = prob[range(input.size(0)), target]
        # compute focal weight
        weight = self.alpha * (1 - prob) ** self.gamma
        # compute cross entropy loss with focal weight
        loss = F.cross_entropy(input, target, reduction='none')
        loss = weight * loss
        # return average loss over batch size
        return loss.mean()


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

    net = nn.DataParallel(net)
    if opt['loss'] == 'focal':
        criterion = FocalLoss()
    elif not opt['loss']: criterion = nn.CrossEntropyLoss()
    else: criterion = None

    return net, criterion


def model_test(net, test_loader, fg_acc=False):
    summ = 0
    correct = 0
    with torch.no_grad():
        net.eval()
        for loader in test_loader:
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                _, validation_output = net(inputs)
                val_y = torch.max(validation_output, 1)[1].cpu().squeeze()
                for i in range(len(labels)):
                    if val_y[i] == labels[i]:
                        correct += 1
                    summ += 1
    val_accuracy = float(correct / summ)
    if fg_acc:
        print('forget accuracy: %.3f (%d / %d).' % (val_accuracy, correct, summ))
    else:
        print('test accuracy: %.3f (%d / %d).' % (val_accuracy, correct, summ))

    return val_accuracy


def eva_and_save_model(net, val_loaders, method):
    test_acc = model_test(net, [test_loader])
    forget_acc = model_test(net, val_loaders, fg_acc=True)
    test_acc = ('%.2f'%test_acc).split('.')[-1]
    forget_acc = ('%.2f'%forget_acc).split('.')[-1]

    acc_info = '-' + test_acc + '-' + forget_acc

    if opt['design']:
        borib = '-ib'
    else:
        borib = '-b'

    torch.save(net.state_dict(), '../model/' + opt['model'] + '/' + opt['loss'] + '-seq' + str(opt['day']) + acc_info + method + borib)


def train(idx, train_loader, val_loader, net, criterion, mode="train", tolerance=4):
    if mode == "train":
        optimizer = optim.SGD(net.parameters(), lr=opt['lr'])
        # optimizer = optim.Adagrad(net.parameters(), lr=opt['lr'])
        epochs = opt['epochs']
    else:
        optimizer = optim.SGD(net.parameters(), lr=opt['finetune_lr'])
        # optimizer = optim.Adagrad(net.parameters(), lr=opt['finetune_lr'])
        epochs = opt['finetune_epochs']

    best_acc = 0.0
    last_acc = 0.0
    tol = 0
    for epoch in range(epochs):
        loss100 = 0.0
        cnt = 0
        for i, data in enumerate(train_loader):
            net.train()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            _, outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss100 += loss.item()
            cnt += 1
        
        if epoch % 5 == 4:      
            # 验证集
            with torch.no_grad(): 
                correct = 0
                summ = 0
                for i, data in enumerate(val_loader):

                    # 测试模式
                    net.eval()

                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

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

    return best_acc, best_model
    

def continual_train_withC():
    net, criterion = init_train()

    coreset = []
    val_loaders = []
    for i in range(opt['day']):
        
        val_loader = data.DataLoader(val_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
        val_loaders.append(val_loader)
        # 新来的数据先和memory coreset变成新的coreset, 然后用coreset微调
        if i:
            #更新coreset
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            new_ebd = get_ebd_byclass(net, train_loader, opt)
            for j in range(opt['class_num']):
                tot_ebd = np.vstack((np.array(new_ebd[j]), ebd[j]))
                tot_data = np.concatenate((train_data[i].tensor_data[train_data[i].tensor_targets == j], coreset[j]))
                idx = diversity_sample(tot_ebd, opt['coreset_size'], ignore=int(opt['ignore']*len(tot_ebd)))
                print(np.array(new_ebd[j]).shape, tot_ebd.shape, np.sum(idx < len(new_ebd[j])))
                coreset[j] = tot_data[idx]
                if not j:
                    coreset_data = coreset[j]
                    coreset_targets = torch.zeros(len(coreset[j]), dtype=int)
                else:
                    coreset_data = np.concatenate((coreset_data, coreset[j]))
                    coreset_targets = torch.cat((coreset_targets, (torch.ones(len(coreset[j]), dtype=int) * j)), 0)
            new_train_data = make_data(coreset_data, coreset_targets)
            new_train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            best_acc, best_model = train(i, new_train_loader, val_loader, net, nn.CrossEntropyLoss(), "fine-tune")
            ebd = get_ebd_byclass(net, new_train_loader, opt)

        else:
            #初始化coreset
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            if opt['loss'] == 'balanced':
                criterion = nn.CrossEntropyLoss(weight=train_data[i].weight.to(device))
            best_acc, best_model = train(i, train_loader, val_loader, net, criterion, "train")
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            ebd = get_ebd_byclass(net, train_loader, opt)
            for j in range(opt['class_num']):
                ebd[j] = np.array(ebd[j])
                idx = diversity_sample(ebd[j], opt['coreset_size'], ignore=int(opt['ignore']*len(ebd[j])))
                coreset.append(train_data[i].tensor_data[train_data[i].tensor_targets == j][idx])
                ebd[j] = ebd[j][idx]
        
        print(best_acc)
        net.load_state_dict(best_model)
    
    eva_and_save_model(net, val_loaders, '-c')


def forgetting_train():
    if opt['day'] == 1:
        print("--------Train with all data--------")
    else:
        print("--------Train with " + str(opt['day']) + " Seqs data with no memory--------") 
    net, criterion = init_train()
    val_loaders = []
    for i in range(opt['day']):
        train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
        val_loader = data.DataLoader(val_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
        val_loaders.append(val_loader)
        if opt['loss'] == 'balanced':
            criterion = nn.CrossEntropyLoss(weight=train_data[i].weight.to(device))
        if i:
            best_acc, best_model = train(i, train_loader, val_loader, net, criterion, 'fine-tune')
        else:
            best_acc, best_model = train(i, train_loader, val_loader, net, criterion, 'train')

        print(best_acc)
        net.load_state_dict(best_model)
    
    eva_and_save_model(net, val_loaders, '-forget')


def continual_train_withCandN():
    print("--------Train with Gonzalez Coreset and new data--------")
    net, criterion = init_train()

    coreset = []
    val_loaders = []
    for i in range(opt['day']):
        
        val_loader = data.DataLoader(val_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
        val_loaders.append(val_loader)

        # 新来的数据直接和memory coreset一起微调，然后生成新的coreset
        if i:
            # 融合新数据和memory
            for j in range(opt['class_num']):
                merge_data = np.concatenate((train_data[i].tensor_data[train_data[i].tensor_targets == j], coreset[j]))
                if not j:
                    tot_data = merge_data
                    tot_targets = torch.zeros(len(merge_data), dtype=int)
                else:
                    tot_data = np.concatenate((tot_data, merge_data))
                    tot_targets = torch.cat((tot_targets, (torch.ones(len(merge_data), dtype=int) * j)), 0)
            new_train_data = make_data(tot_data, tot_targets)
            new_train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            if opt['loss'] == 'balanced':
                criterion = nn.CrossEntropyLoss(weight=new_train_data.weight.to(device))
            best_acc, best_model = train(i, new_train_loader, val_loader, net, criterion, "fine-tune")

            # 生成新的coreset
            new_train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            ebd = get_ebd_byclass(net, new_train_loader, opt)
            for j in range(opt['class_num']):
                ebd[j] = np.array(ebd[j])
                idx = diversity_sample(ebd[j], opt['coreset_size'], ignore=int(opt['ignore']*len(ebd[j])))
                coreset[j] = new_train_data.tensor_data[new_train_data.tensor_targets == j][idx]

        else:
            #初始化coreset
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            if opt['loss'] == 'balanced':
                criterion = nn.CrossEntropyLoss(weight=train_data[i].weight.to(device))
            best_acc, best_model = train(i, train_loader, val_loader, net, criterion, "train")
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            ebd = get_ebd_byclass(net, train_loader, opt)
            for j in range(opt['class_num']):
                ebd[j] = np.array(ebd[j])
                idx = diversity_sample(ebd[j], opt['coreset_size'], ignore=int(opt['ignore']*len(ebd[j])))
                coreset.append(train_data[i].tensor_data[train_data[i].tensor_targets == j][idx])
        
        print(best_acc)
        net.load_state_dict(best_model)
    
    eva_and_save_model(net, val_loaders, '-c+n')


def continual_train_withNthenC():
    net, criterion = init_train()

    coreset = []
    val_loaders = []
    for i in range(opt['day']):
        
        val_loader = data.DataLoader(val_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
        val_loaders.append(val_loader)

        # 新来的数据微调，然后生成新的coreset再微调
        if i:
            # 新来的数据train
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            if opt['loss'] == 'balanced':
                criterion = nn.CrossEntropyLoss(weight=train_data[i].weight.to(device))
            train(i, train_loader, val_loader, net, criterion, "train")

            # 融合新数据和coreset
            for j in range(opt['class_num']):
                merge_data = np.concatenate((train_data[i].tensor_data[train_data[i].tensor_targets == j], coreset[j]))
                if not j:
                    tot_data = merge_data
                    tot_targets = torch.zeros(len(merge_data), dtype=int)
                else:
                    tot_data = np.concatenate((tot_data, merge_data))
                    tot_targets = torch.cat((tot_targets, (torch.ones(len(merge_data), dtype=int) * j)), 0)
            new_train_data = make_data(tot_data, tot_targets)
            # 生成新的coreset
            new_train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            ebd = get_ebd_byclass(net, new_train_loader, opt)
            for j in range(opt['class_num']):
                ebd[j] = np.array(ebd[j])
                idx = diversity_sample(ebd[j], opt['coreset_size'], ignore=int(opt['ignore']*len(ebd[j])))
                coreset[j] = new_train_data.tensor_data[new_train_data.tensor_targets == j][idx]
                if not j:
                    tot_data = coreset[j]
                    tot_targets = torch.zeros(len(coreset[j]), dtype=int)
                else:
                    tot_data = np.concatenate((tot_data, coreset[j]))
                    tot_targets = torch.cat((tot_targets, (torch.ones(len(coreset[j]), dtype=int) * j)), 0)
            # 新coreset微调
            new_train_data = make_data(tot_data, tot_targets)
            new_train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            if opt['loss'] == 'balanced':
                criterion = nn.CrossEntropyLoss(weight=new_train_data.weight.to(device))
            best_acc, best_model = train(i, new_train_loader, val_loader, net, criterion, "fine-tune")
            
        else:
            #初始化coreset
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            if opt['loss'] == 'balanced':
                criterion = nn.CrossEntropyLoss(weight=train_data[i].weight.to(device))
            best_acc, best_model = train(i, train_loader, val_loader, net, criterion, "train")
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            ebd = get_ebd_byclass(net, train_loader, opt)
            for j in range(opt['class_num']):
                ebd[j] = np.array(ebd[j])
                idx = diversity_sample(ebd[j], opt['coreset_size'], ignore=int(opt['ignore']*len(ebd[j])))
                coreset.append(train_data[i].tensor_data[train_data[i].tensor_targets == j][idx])
            
        print(best_acc)
        net.load_state_dict(best_model)
    
    eva_and_save_model(net, val_loaders, '-n->c')


def continual_train_withCandN_uniform():
    print("--------Train with Uniform Coreset and new data--------")
    net, criterion = init_train()

    coreset = []
    cnt = [0, 0]
    val_loaders = []
    for i in range(opt['day']):
        
        val_loader = data.DataLoader(val_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
        val_loaders.append(val_loader)

        # 新来的数据直接和memory coreset一起微调，然后生成新的coreset
        if i:
            # 融合新数据和memory
            for j in range(opt['class_num']):
                merge_data = np.concatenate((train_data[i].tensor_data[train_data[i].tensor_targets == j], coreset[j]))
                if not j:
                    tot_data = merge_data
                    tot_targets = torch.zeros(len(merge_data), dtype=int)
                else:
                    tot_data = np.concatenate((tot_data, merge_data))
                    tot_targets = torch.cat((tot_targets, (torch.ones(len(merge_data), dtype=int) * j)), 0)
            new_train_data = make_data(tot_data, tot_targets)
            new_train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            if opt['loss'] == 'balanced':
                criterion = nn.CrossEntropyLoss(weight=new_train_data.weight.to(device))
            best_acc, best_model = train(i, new_train_loader, val_loader, net, criterion, "fine-tune")

            # 生成新的coreset
            new_train_loader = data.DataLoader(new_train_data, batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            ebd = get_ebd_byclass(net, new_train_loader, opt)
            for j in range(opt['class_num']):
                ebd[j] = np.array(ebd[j])
                tmp = cnt[j]
                cnt[j] += ebd[j].shape[0] - opt['coreset_size']
                idx = uniform_sample(ebd[j], opt['coreset_size'], tmp / cnt[j])
                coreset[j] = new_train_data.tensor_data[new_train_data.tensor_targets == j][idx]

        else:
            #初始化coreset
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=True, num_workers=4)
            if opt['loss'] == 'balanced':
                criterion = nn.CrossEntropyLoss(weight=train_data[i].weight.to(device))
            best_acc, best_model = train(i, train_loader, val_loader, net, criterion, "train")
            train_loader = data.DataLoader(train_data[i], batch_size=opt['batch_size'], shuffle=False, num_workers=4)
            ebd = get_ebd_byclass(net, train_loader, opt)
            for j in range(opt['class_num']):
                ebd[j] = np.array(ebd[j])
                cnt[j] += ebd[j].shape[0]
                idx = uniform_sample(ebd[j], opt['coreset_size'], opt['coreset_size'] / cnt[j])
                coreset.append(train_data[i].tensor_data[train_data[i].tensor_targets == j][idx])

        print(best_acc)
        net.load_state_dict(best_model)
    
    eva_and_save_model(net, val_loaders, '-uc+n')


if __name__ == '__main__':
    
    # forgetting_train()
    # continual_train_withCandN_uniform()
    continual_train_withCandN()
    

    # continual_train_withC()
    # continual_train_withNthenC()