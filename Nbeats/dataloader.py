import os
from torch.utils import data
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
from dataset import WindowDataset
from utils import MeanScaler


def get_dataloader(target_feature, data_root, data_name, batch_size, input_window, output_window, train_rate, stride=1):
    '''
    ---Return--- 
    
    train_dataLoader
    test_dataloader
    ------------
    ''' 

    data_path = os.path.join(data_root, data_name)
    data = pd.read_csv(data_path, parse_dates=['index'])
    data.rename(columns={'index':'date'}, inplace=True)

    data = data[['date', 'open']]
    data.set_index('date', inplace=True)

    # train / test split
    train_periods = int(data.shape[0] * train_rate)

    data_tr = data.iloc[:train_periods, :]
    data_te = data.iloc[train_periods:, :]

    # scaling 
    yscaler = MeanScaler()
    tmp = yscaler.fit_transform(data_tr['open'])
    data_tr['open'] = tmp

    tmp = yscaler.transform(data_te['open'])
    data_te['open'] = tmp

    custom_dataset = WindowDataset(data_te, input_window, output_window, stride=stride)
    test_dataloader = DataLoader(custom_dataset, 1)         

    custom_dataset = WindowDataset(data_tr, input_window, output_window, stride=stride)
    train_dataloader = DataLoader(custom_dataset, batch_size) 

    return train_dataloader, test_dataloader, yscaler