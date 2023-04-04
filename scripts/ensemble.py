import torch
import torch.utils.data as data
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
import numpy as np
import json
from loaddata import load_RSNA
from mymodels import ResNet18, ResNet18_pt, VGG16_pt


with open("../opt-json/RSNA-binary.json") as f:
    opt = json.load(f)

train_data = load_RSNA(
    data_info_path=opt['data_info_path'],
    data_path=opt['data_path'],
    classes=opt['class_num'],
    modify_size=opt['modify_size'],
    train='train')
test_data = load_RSNA(
    data_info_path=opt['data_info_path'],
    data_path=opt['data_path'],
    classes=opt['class_num'],
    modify_size=opt['modify_size'],
    train='test')

train_loader = data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)
val_loader = data.DataLoader(test_data, batch_size=opt['batch_size'], shuffle=True)

device = torch.device("cuda:" + str(opt['GPU_rank']))

lr = LinearRegression()

net1 = ResNet18_pt(opt['class_num']).to(device)
net1.load_state_dict(torch.load('../model/'))
net2 = VGG16_pt(opt['class_num']).to(device)
net2.load_state_dict(torch.load('../model/'))
nets = [net1, net2]

print("Start Training...")
for epoch in range(10):
    for i, data in enumerate(train_loader):
        S_train = np.zeros((len(data[0]), len(nets)))
        for j, net in enumerate(nets):
            net.eval()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            prob = torch.max(outputs, 1)[0].data.cpu().numpy().squeeze()
            S_train[:, j] = prob[:]
        
        lr.fit(S_train, labels.cpu().numpy().squeeze())
print("Done Training!")


print("Start Testing...")
correct = 0
correct12 = [0, 0]
summ = 0
for i, data in enumerate(val_loader):
    S_test = np.zeros((len(data[0]), len(nets)))
    for j, net in enumerate(nets):
        net.eval()

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        prob = torch.max(outputs, 1)[0].data.cpu().numpy().squeeze()


        val_y = torch.max(outputs, 1)[1].data.cpu().numpy().squeeze()
        for i in range(len(labels)):
            if val_y[i] == labels[i]:
                correct12[j] += 1
    

        S_test[:, j] = prob[:]
        
    y_pred = lr.predict(S_test)[:]
    correct += np.sum(y_pred == labels.cpu().numpy().squeeze())
    summ += np.sum(y_pred != labels.cpu().numpy().squeeze()) + np.sum(y_pred == labels.cpu().numpy().squeeze())
print("Done Testing!")


val_accuracy = float(correct / summ)
print('val accuracy: %.2f (%d / %d).' % (val_accuracy, correct, summ))
print('val accuracy: %.2f (%d / %d).' % (float(correct12[0] / summ), correct12[0], summ))
print('val accuracy: %.2f (%d / %d).' % (float(correct12[1] / summ), correct12[1], summ))