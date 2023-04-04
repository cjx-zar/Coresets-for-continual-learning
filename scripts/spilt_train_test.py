import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from PIL import Image
import pydicom as dicom
import torch

# all_data = pd.read_csv('/home/cjx/workspace/RSNA/stage_2_detailed_class_info.csv', sep=',')
# label_list = all_data['Target'].unique()
# all_data = pd.read_csv('/home/cjx/workspace/RSNA/stage_2_train_labels.csv', sep=',')
# label_list = all_data['Target'].unique()
# print(len(all_data))

# m = {'No Lung Opacity / Not Normal' : 'NLONN', 'Normal' : 'N', 'Lung Opacity': 'LO'}

# for y in label_list:
#     all_data[all_data['Target'] == y].to_csv('../detailed/detailed_' + m[y] + '.csv', index=False)
#     print(len(all_data[all_data['Target'] == y]))

# for y in label_list:
#     all_data[all_data['Target'] == y].to_csv('../binary/binary_' + str(y) + '.csv', index=False)
#     print(len(all_data[all_data['Target'] == y]))

path = '/home/cjx/workspace/RSNA/binary/'
train = 'train'
