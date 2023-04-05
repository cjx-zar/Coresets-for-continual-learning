from torchvision import datasets
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
import torch
import pydicom as dicom
import random


def train_val_split(data_path, df, transforms, val_ratio):
    tensor_data = []
    tensor_targets = []
    for idx in range(len(df)):
        img_path = os.path.join(data_path, (df['patientId'].iloc[idx] + '.dcm'))
        ds = dicom.dcmread(img_path)
        img = Image.fromarray(ds.pixel_array)
        if transforms is not None:
            img = transforms(img)
        tensor_data.append(img)
        tensor_targets.append(torch.tensor(df['Target'].iloc[idx]))
    
    combined = list(zip(tensor_data, tensor_targets))
    random.shuffle(combined)
    split = int(val_ratio * len(combined))
    part1 = combined[:split]
    part2 = combined[split:]
    val_data, val_targets = zip(*part1)
    train_data, train_targets = zip(*part2)

    return np.array(train_data), torch.tensor(train_targets), np.array(val_data), torch.tensor(val_targets)




class MyCustomDataset(Dataset):
    def __init__(self, data_path, df, transforms=None):
        self.data_path = data_path
        self.df = df
        self.transforms = transforms

        self.weight = torch.tensor(1 - self.df['Target'].value_counts() / len(self.df), dtype=torch.float)
        self.data_size = len(self.df)
        self.tensor_data = []
        self.tensor_targets = []
        # self.data = []
        # self.targets = []
        for idx in range(len(self.df)):
            img_path = os.path.join(self.data_path, (self.df['patientId'].iloc[idx] + '.dcm'))
            ds = dicom.dcmread(img_path)
            img = Image.fromarray(ds.pixel_array)
            if self.transforms is not None:
                img = self.transforms(img)
            self.tensor_data.append(img)
            self.tensor_targets.append(torch.tensor(self.df['Target'].iloc[idx]))
        self.tensor_data = np.array(self.tensor_data)
        self.tensor_targets = np.array(self.tensor_targets)
        #     self.data.append(img.numpy())
        #     self.targets.append(self.df['Target'].iloc[idx])
        # self.data = np.array(self.data)
        # self.targets = np.array(self.targets)

    def __getitem__(self, index):
        # img_path = os.path.join(
        #     self.data_path, (self.df['patientId'][index] + '.dcm'))
        # ds = dicom.dcmread(img_path)
        # img = Image.fromarray(ds.pixel_array)
        # if self.transforms is not None:
        #     img = self.transforms(img)
        # return (img, torch.tensor(self.df['Target'][index]))

        return (self.tensor_data[index], self.tensor_targets[index])

    def __len__(self):
        return len(self.tensor_targets)


class MySimpleDataset(Dataset):
    def __init__(self, data, targets):
       self.tensor_data = data
       self.tensor_targets = targets
       self.weight = torch.tensor(1 - torch.bincount(self.tensor_targets) / len(self.tensor_targets), dtype=torch.float)

    def __getitem__(self, index):
        return (self.tensor_data[index], self.tensor_targets[index])

    def __len__(self):
        return len(self.tensor_targets)


def make_data(data, targets):
    return MySimpleDataset(data, targets)


def init_df(data_info_path, train, day, reshuffle, design):
    df_list = []

    if reshuffle:
        # 删除现有的train/test划分（如果有）
        for file in os.listdir(data_info_path):
            if file.endswith('.txt'):
                os.remove(os.path.join(data_info_path, file))
        # 生成新的train/test划分
        flag = True
        for file in os.listdir(data_info_path):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(
                    data_info_path, file), sep=',')
                df = shuffle(df)
                e = int(len(df) * 0.7)

                df[:e].to_csv(os.path.join(
                    data_info_path, "train.txt"), mode='a', header=flag, index=False)
                df[e:].to_csv(os.path.join(
                    data_info_path, "test.txt"), mode='a', header=flag, index=False)
                flag = False

    tot_df = pd.read_csv(os.path.join(data_info_path, train + '.txt'), sep=',')
    if not design:
        tot_df = tot_df.sample(frac=1)
    left = 0
    gap = int(len(tot_df) / day)
    for i in range(day):
        df_list.append(tot_df.iloc[left:left+gap])
        left += gap
    return df_list


def load_seq_RSNA(data_info_path, data_path, modify_size, train, day=1, reshuffle=False, design=False, val_ratio=0.1):
    df_list = init_df(data_info_path, train, day, reshuffle, design)
    
    if train == 'train' or train == 'play':
        trainset_list = []
        valset_list = []
        for i in range(day):
            train_data, train_targets, val_data, val_targets = train_val_split(data_path,df_list[i],transforms.Compose([
                                transforms.Resize(modify_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4885, ),(0.2446, ))]), val_ratio)
            trainset_list.append(make_data(train_data, train_targets))
            valset_list.append(make_data(val_data, val_targets))

        return trainset_list, valset_list

    elif train == 'test':
        testset_list = []
        for i in range(day):
            testset_list.append(MyCustomDataset(data_path=data_path, df=df_list[i],
                    transforms=transforms.Compose([
                        transforms.Resize(modify_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4885, ),(0.2446, ))])))
        return testset_list