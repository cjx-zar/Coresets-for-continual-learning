import json
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import os

from loaddata import load_seq_RSNA
from mymodels import ResNet18_pt, VGG16_pt, VGG19_pt, EfficientNet_pt, DenseNet_pt, LeNet_pt


with open("../opt-json/RSNA-binary.json") as f:
    opt = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = opt['GPUs']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if opt['model'] == 'resnet18-pt':
    net = ResNet18_pt(opt['class_num']).to(device)
elif opt['model'] == 'vgg16-pt':
    net = VGG16_pt(opt['class_num']).to(device)
elif opt['model'] == 'efficient-pt':
    net = EfficientNet_pt(opt['class_num']).to(device)
elif opt['model'] == 'densenet-pt':
    net = DenseNet_pt(opt['class_num']).to(device)
elif opt['model'] == 'vgg19-pt':
    net = VGG19_pt(opt['class_num']).to(device)
elif opt['model'] == 'lenet-pt':
    net = LeNet_pt(opt['class_num']).to(device)
net = nn.DataParallel(net)
net.load_state_dict(torch.load('../model/vgg16-pt/vgg16-pt-balanced-seq5-84-c+n', map_location=device))


test_data = load_seq_RSNA(data_info_path=opt['data_info_path'], data_path=opt['data_path'],
                              modify_size=opt['modify_size'], train='train')
loader = data.DataLoader(test_data[0], batch_size=opt['batch_size'], shuffle=False, num_workers=4)

summ = 0
correct = 0
with torch.no_grad():
    net.eval()
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
    print('val accuracy: %.2f (%d / %d).' % (val_accuracy, correct, summ))