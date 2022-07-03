import torch
import torch.nn as nn
import torch.nn.functional as F 
import random
import numpy as np


class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)
    
    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t

def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation
    gaussian maximum likelihood using log 
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.sample()
    if ypred.dim() == 1:
        ypred = ypred.unsqueeze(0)
    return ypred

def gaussian_likelihood_loss(z, mu, sigma):
    '''
    Gaussian Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)
    likelihood: 
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))
    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
    '''
    # negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
    negative_likelihood = 1/2 * torch.log(2 * np.pi * (sigma**2)) + (z - mu) ** 2 / (2 * sigma ** 2)
    return negative_likelihood.mean()



class DeepAR(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, likelihood="g"):
        super(DeepAR, self).__init__()

        # network
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size+input_size, hidden_size, \
                num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(hidden_size, 1)

        self.likelihood = likelihood

    def forward(self, X, y, Xf):
        num_ts, seq_len, _ = X.size() # 64, 168, 2
        _, output_horizon, num_features = Xf.size() # 64, 60, 2

        ynext = None
        ypred = []
        mus = []
        sigmas = []       
        h, c = None, None
        for s in range(seq_len + output_horizon):
            if s < seq_len:
                ynext = y[:, s].view(-1, 1) # (batch_size, 1) 
                yembed = self.input_embed(ynext).view(num_ts, -1) # (batch_size, embed_size)
                x = X[:, s, :].view(num_ts, -1) # (batch_size, num_feature)
            else:
                yembed = self.input_embed(ynext).view(num_ts, -1) # (batch_size, embed_size)
                x = Xf[:, s-seq_len, :].view(num_ts, -1) # (batch_size, num_feature)
            x = torch.cat([x, yembed], dim=1) # (batch_size, num_feature+embed_size)
            inp = x.unsqueeze(1) # (batch_size, 1, num_feature+embed_size)
            if h is None and c is None:
                out, (h, c) = self.encoder(inp) # h size (num_layers, num_ts, hidden_size)
            else:
                out, (h, c) = self.encoder(inp, (h, c))

            hs = h[-1, :, :] # (batch_size, hidden_size)
            hs = F.relu(hs)
            mu, sigma = self.likelihood_layer(hs)
            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))
            if self.likelihood == 'g':
                ynext = gaussian_sample(mu, sigma)

            if s >= seq_len - 1 and s < output_horizon + seq_len - 1:
                ypred.append(ynext)
                
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)
        return ypred, mu, sigma   
        