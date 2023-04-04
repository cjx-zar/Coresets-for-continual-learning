import torch
import json
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from loaddata import load_RSNA


with open("../opt-json/RSNA-binary.json") as f:
    opt = json.load(f)

train_data = load_RSNA(
    data_info_path=opt['data_info_path'],
    data_path=opt['data_path'],
    classes=opt['class_num'],
    modify_size=opt['modify_size'],
    train=True)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# 初始化均值和标准差
mean = 0.
std = 0.

# 遍历所有图像并累加均值和方差
for images, _ in dataloader:
    # 计算当前批次中每个通道的均值和方差，并乘以批次大小（N）
    batch_mean = torch.mean(images, dim=(0, 2, 3)) * images.shape[0]
    batch_std = torch.std(images, dim=(0, 2 ,3)) * images.shape[0]

    # 累加到总体均值和方差中
    mean += batch_mean 
    std += batch_std

# 计算总体均值和方差，并除以样本数量（N）
mean /= len(train_data)
std /= len(train_data)

# 打印结果
print('Mean:', mean)
print('Std:', std)