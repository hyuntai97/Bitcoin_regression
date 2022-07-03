import torch
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

        for X, Y, Xf, Yf in self.trainloader:
            ypred, mu, sigma = self.model(X, Y, Xf)
            ytrain_tensor = torch.cat([Y, Yf], dim=1)
            z = ytrain_tensor.squeeze(2)
            loss = gaussian_likelihood_loss(z, mu, sigma)
            train_loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        train_loss = np.mean(train_loss)

        return train_loss



