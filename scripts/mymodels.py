import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class ResNet18(nn.Module):
    def __init__(self, num_classes, in_channel, pre):
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=pre)
        self.net.conv1 = torch.nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = torch.nn.Linear(512, num_classes)
        self.ebd = nn.Sequential(*list(self.net.children())[:-1])
        
    def forward(self, x):
        x = self.ebd(x)
        feature = x.view(x.size(0), -1)
        res = self.net.fc(feature)
        return feature, res

class VGG16(nn.Module):
    def __init__(self, num_classes, in_channel=1):
        super(VGG16, self).__init__()
        self.net = models.vgg16_bn(pretrained=True)
        self.net.features[0] = torch.nn.Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.net.classifier[6] = torch.nn.Linear(4096, num_classes)
        self.ebd = nn.Sequential(*list(self.net.children())[:-1])
        
    def forward(self, x):
        x = self.ebd(x)
        feature = x.view(x.size(0), -1)
        res = self.net.classifier(feature)
        return feature, res
    
class VGG19(nn.Module):
    def __init__(self, num_classes, in_channel=1):
        super(VGG19, self).__init__()
        self.net = models.vgg19_bn(pretrained=True)
        self.net.features[0] = torch.nn.Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.net.classifier[6] = torch.nn.Linear(4096, num_classes)
        self.ebd = nn.Sequential(*list(self.net.children())[:-1])
        
    def forward(self, x):
        x = self.ebd(x)
        feature = x.view(x.size(0), -1)
        res = self.net.classifier(feature)
        return feature, res

class DenseNet(nn.Module):
    def __init__(self, num_classes, in_channel=1):
        super(DenseNet, self).__init__()
        self.net = models.densenet121(pretrained=True)
        self.net.features.conv0 = torch.nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        self.net.classifier = torch.nn.Linear(50176, num_classes)
        
        self.ebd = nn.Sequential(*list(self.net.children())[:-1])
        
    def forward(self, x):
        x = self.ebd(x)
        feature = x.view(x.size(0), -1)
        res = self.net.classifier(feature)
        return feature, res


def ResNet18_npt(num_classes, in_channel=1):
    return ResNet18(num_classes, in_channel, pre=False)

def ResNet18_pt(num_classes, in_channel=1):
    return ResNet18(num_classes, in_channel, pre=True)

def VGG16_pt(num_classes, in_channel=1):
    return VGG16(num_classes, in_channel)

def VGG19_pt(num_classes, in_channel=1):
    return VGG19(num_classes, in_channel)

def DenseNet_pt(num_classes, in_channel=1):
    return DenseNet(num_classes, in_channel)


# 下面还没改好

def EfficientNet_pt(num_classes):
    net = EfficientNet.from_pretrained('efficientnet-b0')
    net._change_in_channels(1)
    net._fc = nn.Linear(net._fc.in_features, num_classes)
    return net

def LeNet_pt(num_classes, in_channel=1):
    net = models.googlenet(pretrained=True)
    net.fc = nn.Linear(1024, num_classes)
    return net