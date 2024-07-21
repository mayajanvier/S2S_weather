import numpy as np 
import torch
import torch.nn as nn 
from torch.distributions import Normal
import pandas as pd
from torch.utils.data import DataLoader
from processings.dataset import PandasDataset, WeatherDataset
from processings.format_data import compute_wind_speed


class MOS(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, mu, sigma, features, truth):
        # linear
        theta = self.layer(features) 
        
        # MOS
        mu_pred = mu*theta[:,0] + theta[:,1]
        # TODO : manage theta of shape 4 (sigma non zero)
        sigma_pred = torch.exp(theta[:,2]) # to preserve positiveness

        # out distribution is a normal distribution of mean mu_pred and std sigma_pred
        distrib = Normal(mu_pred, sigma_pred)
        return distrib
    
    

class SpatialMOS(nn.Module):
    """ Manage all latitude and longitude at once """
    def __init__(self, feature_dim, lat_dim, lon_dim, out_dim):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(out_dim, feature_dim, lat_dim, lon_dim)*0.01)

    def forward(self, mu, sigma, features, truth):
        # Perform element-wise multiplication and sum over the shared dimension (features, i)
        theta = torch.einsum('bijk,mijk->bmjk', features, self.matrix) # shape (batch, out_dim, lat_dim, lon_dim)
        
        # MOS
        mu_pred = mu*theta[:,0,:,:] + theta[:,1,:,:]
        # TODO : manage theta of shape 4 (sigma non zero)
        sigma_pred = torch.exp(theta[:,2,:,:]) # to preserve positiveness

        # out distribution is a normal distribution of mean mu_pred and std sigma_pred
        distrib = Normal(mu_pred, sigma_pred)
        return distrib

class SpatialEMOS(nn.Module):
    """ Manage all latitude and longitude at once """
    def __init__(self, feature_dim, lat_dim, lon_dim, out_dim):
        super().__init__()
        # TODO remettre scaling 0.01 apres test
        self.matrix = nn.Parameter(torch.randn(out_dim, feature_dim, lat_dim, lon_dim)*0.0001) 

    def forward(self, mu, sigma, features, truth):
        # Perform element-wise multiplication and sum over the shared dimension (features, i)
        theta = torch.einsum('bijk,mijk->bmjk', features, self.matrix) # shape (batch, out_dim, lat_dim, lon_dim)
        
        # MOS
        print("theta")
        print(theta[:,0,:,:].mean(), theta[:,1,:,:].mean(), theta[:,2,:,:].mean(), theta[:,3,:,:].mean())
        print(torch.isnan(theta).sum())
        print((theta==0).sum())
        print(torch.isinf(theta).sum())
        print(theta[:,0,:,:].min(), theta[:,0,:,:].max())
        print(theta[:,1,:,:].min(), theta[:,1,:,:].max())
        print(theta[:,2,:,:].min(), theta[:,2,:,:].max())
        print(theta[:,3,:,:].min(), theta[:,3,:,:].max())

        mu_pred = mu*(theta[:,0,:,:] +1)+ theta[:,1,:,:] # test prior a, c=1 
        print("sigma")
        sigma_values = sigma*(theta[:,2,:,:]+1) + theta[:,3,:,:]
        print(sigma_values.min(), sigma_values.max())
        print(torch.isinf(sigma*(theta[:,2,:,:]+1)).sum())

        sigma_pred = torch.exp(sigma*(theta[:,2,:,:]+1) + theta[:,3,:,:]) # to preserve positiveness
        # count number of nan
        print(torch.isnan(mu_pred).sum())
        print(torch.isnan(sigma_pred).sum())
        #count number of zeros 
        print((mu_pred==0).sum())
        print((sigma_pred==0).sum())

        # out distribution is a normal distribution of mean mu_pred and std sigma_pred
        distrib = Normal(mu_pred, sigma_pred)
        return distrib

class SpatialEMOSprior(nn.Module):
    """ Manage all latitude and longitude at once """
    def __init__(self, feature_dim, lat_dim, lon_dim, out_dim):
        super().__init__()
        # TODO remettre scaling 0.01 apres test
        self.matrix = nn.Parameter(torch.randn(out_dim, feature_dim, lat_dim, lon_dim)*0.01) 

    def forward(self, mu, sigma, features, truth):
        # Perform element-wise multiplication and sum over the shared dimension (features, i)
        theta = torch.einsum('bijk,mijk->bmjk', features, self.matrix) # shape (batch, out_dim, lat_dim, lon_dim)
        
        # MOS
        print(theta[:,0,:,:].mean(), theta[:,1,:,:].mean(), theta[:,2,:,:].mean(), theta[:,3,:,:].mean())
        mu_pred = mu*(theta[:,0,:,:]+1) + theta[:,1,:,:] # test prior a, c=1 
        sigma_pred = torch.exp(sigma*(theta[:,2,:,:]+1) + theta[:,3,:,:]) + 1e-3 # to preserve positiveness
        breakpoint

        # out distribution is a normal distribution of mean mu_pred and std sigma_pred
        distrib = Normal(mu_pred, sigma_pred)
        return distrib

# class SpatialEMOSnoexp(nn.Module):
#     """ Manage all latitude and longitude at once """
#     def __init__(self, feature_dim, lat_dim, lon_dim, out_dim):
#         super().__init__()
#         # TODO voir si on a besoin du scaling 0.01
#         self.matrix = nn.Parameter(torch.randn(out_dim, feature_dim, lat_dim, lon_dim)*0.01) 

#     def forward(self, mu, sigma, features, truth):
#         # Perform element-wise multiplication and sum over the shared dimension (features, i)
#         theta = torch.einsum('bijk,mijk->bmjk', features, self.matrix) # shape (batch, out_dim, lat_dim, lon_dim)
        
#         # MOS
#         mu_pred = mu*theta[:,0,:,:] + theta[:,1,:,:]
#         sigma_pred = torch.abs(sigma*theta[:,2,:,:] + theta[:,3,:,:]) # to preserve positiveness

#         # out distribution is a normal distribution of mean mu_pred and std sigma_pred
#         distrib = Normal(mu_pred, sigma_pred)
#         return distrib


# class SpatialEMOSinit(nn.Module):
#     """ Manage all latitude and longitude at once """
#     def __init__(self, feature_dim, lat_dim, lon_dim, out_dim):
#         super().__init__()
#         # TODO voir si on a besoin du scaling 0.01
#         self.matrix = nn.Parameter(torch.randn(out_dim, feature_dim, lat_dim, lon_dim)*0.0001) 

#     def forward(self, mu, sigma, features, truth):
#         # Perform element-wise multiplication and sum over the shared dimension (features, i)
#         theta = torch.einsum('bijk,mijk->bmjk', features, self.matrix) # shape (batch, out_dim, lat_dim, lon_dim)
        
#         # MOS
#         mu_pred = mu*theta[:,0,:,:] + theta[:,1,:,:]
#         sigma_pred = torch.exp(sigma*theta[:,2,:,:] + theta[:,3,:,:]) # to preserve positiveness

#         # out distribution is a normal distribution of mean mu_pred and std sigma_pred
#         distrib = Normal(mu_pred, sigma_pred)
#         return distrib

class SpatioTemporalEMOS(nn.Module):
    """ Manage all latitude, longitude and time (months) at once,
    and moving average mean of MOS parameters over time dimension """
    def __init__(self, time_dim, feature_dim, lat_dim, lon_dim, out_dim):
        super().__init__()
        self.time_dim = time_dim
        self.matrix = nn.Parameter(torch.randn(time_dim, out_dim, feature_dim, lat_dim, lon_dim))

    def forward(self, mu, sigma, features, truth):
        # Perform element-wise multiplication and sum over the shared dimension (features, i)
        theta = torch.einsum('btijk,tmijk->btmjk', features, self.matrix) # shape (batch, time_dim, out_dim, lat_dim, lon_dim)

        # Circular rolling window mean of the out_dim over time
        theta_rolled = torch.zeros_like(theta)
        for k in range(self.time_dim):
            indices = [(k-1) % self.time_dim, k, (k+1) % self.time_dim]
            theta_rolled[:, k, :, :, :] = torch.mean(theta[:, indices, :, :, :], dim=1) # shape unchanged
        
        # MOS
        mu_pred = mu*theta[:,:,0,:,:] + theta[:,:,1,:,:]
        sigma_pred = torch.exp(sigma*theta[:,:,2,:,:] + theta[:,:,3,:,:]) # to preserve positiveness

        # out distribution is a normal distribution of mean mu_pred and std sigma_pred
        distrib = Normal(mu_pred, sigma_pred)
        return distrib


if __name__== "__main__":
    ### SINGLE MOS
    # check sizes:
    #import torchsummary
    # linear = MOS(50,4)
    # #torchsummary.summary(linear, input_size=(1,50))

    # # one forward pass
    # data_folder = "../scratch/"
    # train_data = pd.read_json(data_folder+'data_2m_temperature.json')
    # train_data = PandasDataset(train_data, "2m_temperature")
    # train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    # for batch in train_loader:
    #     mu = batch['mu']
    #     sigma = batch['sigma']
    #     X = batch['input']
    #     y = batch['truth']
    #     out_distrib = linear(mu, sigma, X, y)
    #     print(out_distrib.sample())
    #     break

    ### SPATIAL MOS
    linear = SpatialMOS(47, 121, 240, 3)

    # one forward pass
    data_folder = "/home/majanvie/scratch/data/raw"
    train_folder = f"{data_folder}/train"
    obs_folder = f"{data_folder}/obs"
    
    train_dataset = WeatherDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable="10m_wind_speed",
        lead_time_idx=28,
        valid_years=[1996,2017],
        valid_months=[1,1],
        subset="train")

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    print("Nb of training examples:",len(train_loader.dataset.data_index))
    for batch in train_loader:
        mu = batch['mu']
        sigma = batch['sigma']
        X = batch['input']
        y = batch['truth']
        out_distrib = linear(mu, sigma, X, y)
        print(out_distrib.sample())
        print(out_distrib.sample().shape)
        break