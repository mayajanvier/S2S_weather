import numpy as np 
import torch
import torch.nn as nn 
from torch.distributions import Normal
import pandas as pd
from torch.utils.data import DataLoader
from processings.dataset import PandasDataset


class MOS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, mu, sigma, features, truth):
        # linear
        theta = self.layer(features) 
        #print("theta", theta)
        
        # MOS
        mu_pred = mu*theta[:,0] + theta[:,1]

        #if not torch.any(torch.eq(sigma, 0)):
        #    sigma_pred = torch.log(sigma)*theta[:,2] + torch.exp(theta[:,3])
        #else:
        sigma_pred = torch.exp(theta[:,2]) # to preserve positiveness

        #print("mu", mu_pred)
        #print("sigma", sigma_pred)

        # out distribution is a normal distribution of mean mu_pred and std sigma_pred
        distrib = Normal(mu_pred, sigma_pred)
        return distrib
    

if __name__== "__main__":
    # check sizes:
    #import torchsummary
    linear = MOS(50,4)
    #torchsummary.summary(linear, input_size=(1,50))

    # one forward pass
    data_folder = "../scratch/"
    train_data = pd.read_json(data_folder+'data_2m_temperature.json')
    train_data = PandasDataset(train_data, "2m_temperature")
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    for batch in train_loader:
        mu = batch['mu']
        sigma = batch['sigma']
        X = batch['input']
        y = batch['truth']
        out_distrib = linear(mu, sigma, X, y)
        print(out_distrib.sample())
        break