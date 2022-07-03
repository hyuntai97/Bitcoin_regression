import torch
from torch.nn import functional as F
import time 
import datetime
import numpy as np 
from tqdm import tqdm 
from progressbar import *

from model import *


class ModelTrain:
    '''
    Model Training
    '''

    def __init__(
        self, 
        model, 
        train_dataloader, 
        epochs, 
        optimizer, 
        ):

        self.model = model
        self.trainloader = train_dataloader
        self.optimizer = optimizer

        #-- Training time check
        total_start = time.time()

        #-- Train history
        train_loss_lst = []

        #-- Train iteration 
        progress = ProgressBar()

        for epoch in progress(range(epochs)):
            train_loss = self.train()            

            train_loss_lst.append(train_loss)
        
        end = time.time() - total_start
        total_time = datetime.timedelta(seconds=end)
        print('\nFinish Train: Training Time: {}\n'.format(total_time))    

        #-- Save history 
        self.history = {}
        self.history['train'] = []
        self.history['train'].append({'train_loss':train_loss_lst})
        
        self.history['time'] = []
        self.history['time'].append({
            'epoch':epochs,
            'total':str(total_time)
        })      

    def train(self):
        self.model.train()

        train_loss = []
        
        for x_train_batch, y_train_batch in self.trainloader:
            self.optimizer.zero_grad()
            _, forecast, _, _ = self.model(x_train_batch.to(self.model.device))
            # loss customizing 가능 
            loss = F.mse_loss(forecast, y_train_batch.squeeze(-1).to(self.model.device))
            train_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
        train_loss = np.mean(train_loss)

        return train_loss        