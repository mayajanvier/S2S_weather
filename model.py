import numpy as np 
import torch
import torch.nn as nn 
from torch.distributions import Normal
import pandas as pd
from torch.utils.data import DataLoader

### EMOS 
class SpatialEMOS(nn.Module):
    """ Manage all latitude and longitude at once """
    def __init__(self, feature_dim, lat_dim, lon_dim, out_dim):
        super().__init__()
        # scaling 0.01 because of exponential
        self.matrix = nn.Parameter(torch.randn(out_dim, feature_dim, lat_dim, lon_dim)*0.01) 

    def forward(self, mu, sigma, features, truth):
        # Perform element-wise multiplication and sum over the shared dimension (features, i)
        theta = torch.einsum('bijk,mijk->bmjk', features, self.matrix) # shape (batch, out_dim, lat_dim, lon_dim)
        
        # MOS
        mu_pred = mu*(theta[:,0,:,:])+ theta[:,1,:,:] # test prior a, c=1 
        sigma_pred = torch.exp(torch.log(sigma)*(theta[:,2,:,:]) + theta[:,3,:,:]) # to preserve positiveness

        # out distribution is a normal distribution of mean mu_pred and std sigma_pred
        distrib = Normal(mu_pred, sigma_pred)
        return distrib

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

### DRUnets
class DRUnet(nn.Module):
    """ Both variables, forecasting mode
    Distributional regression U-Net from Pic et al.:
    Inputs: size 121x240 with 77 channels
    Architecture:
        - 2 decreasing resolution blocks, 2 increasing resolution blocks 
        with skip connections
        - Latent layers: 3x3 conv, BN, ReLU
        - Down blocks: twice 3x3 conv, BN, ReLU + 2x2 max pooling
        - Up blocks: bilinear upsampling+ twice 3x3 conv, BN, ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # layers
        self.layer1_down = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer2_down = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.latent = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3_up = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4_up = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # up and down scalers
        self.down1 = nn.MaxPool2d(2) # max pooling
        self.down2 = nn.MaxPool2d(2) # max pooling
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling 
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling

    def forward(self, x):
        # down
        x1 = self.layer1_down(x) # keep for skip connection
        x2 = self.down1(x1)
        x3 = self.layer2_down(x2) # keep for skip connection
        x4 = self.down2(x3)
        # latent
        x5 = self.latent(x4)
        # up
        x6 = self.up1(x5)
        x7 = self.layer3_up(x6 + x3) # skip connection
        x8 = self.up2(x7)
        x9 = self.layer4_up(x8+ x1) # skip connection
        
        # distributional outputs
        mu_temp, std_temp = x9[:,0,:,:], torch.exp(x9[:,1,:,:]) # to preserve positiveness
        mu_wind, std_wind = x9[:,2,:,:], torch.exp(x9[:,3,:,:]) # to preserve positiveness
        temp_dist = Normal(mu_temp, std_temp)
        wind_dist = Normal(mu_wind, std_wind)
        return x9, temp_dist, wind_dist

class DRUnetVar(nn.Module):
    """ One variable, forecasting mode
    Distributional regression U-Net from Pic et al.:
    Inputs: size 121x240 with 77 channels
    Architecture:
        - 2 decreasing resolution blocks, 2 increasing resolution blocks 
        with skip connections
        - Latent layers: 3x3 conv, BN, ReLU
        - Down blocks: twice 3x3 conv, BN, ReLU + 2x2 max pooling
        - Up blocks: bilinear upsampling+ twice 3x3 conv, BN, ReLU"""
    def __init__(self, in_channels, out_channels, variable):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable = variable
        # layers
        self.layer1_down = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer2_down = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.latent = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3_up = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4_up = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # up and down scalers
        self.down1 = nn.MaxPool2d(2) # max pooling
        self.down2 = nn.MaxPool2d(2) # max pooling
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling 
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling

    def forward(self, x):
        # down
        x1 = self.layer1_down(x) # keep for skip connection
        x2 = self.down1(x1)
        x3 = self.layer2_down(x2) # keep for skip connection
        x4 = self.down2(x3)
        # latent
        x5 = self.latent(x4)
        # up
        x6 = self.up1(x5)
        x7 = self.layer3_up(x6 + x3) # skip connection
        x8 = self.up2(x7)
        x9 = self.layer4_up(x8+ x1) # skip connection
        
        # distributional outputs
        if self.variable == "2m_temperature":
            mu_var = x9[:,0,:,:]
            std_var = torch.exp(x9[:,1,:,:]) # to preserve positiveness
        elif self.variable == "10m_wind_speed":
            mu_var = x9[:,0,:,:] 
            std_var = torch.exp( x9[:,1,:,:]) #torch.exp(torch.log(sigma) * x9[:,2,:,:] + x9[:,3,:,:]) # to preserve positiveness

        var_dist = Normal(mu_var, std_var)
        return x9, var_dist

class DRUnetPrior(nn.Module):
    """ Both, post-processing mode
    Distributional regression U-Net from Pic et al.:
    Inputs: size 121x240 with 77 channels
    Architecture:
        - 2 decreasing resolution blocks, 2 increasing resolution blocks 
        with skip connections
        - Latent layers: 3x3 conv, BN, ReLU
        - Down blocks: twice 3x3 conv, BN, ReLU + 2x2 max pooling
        - Up blocks: bilinear upsampling+ twice 3x3 conv, BN, ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # layers
        self.layer1_down = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer2_down = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.latent = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3_up = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4_up = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # up and down scalers
        self.down1 = nn.MaxPool2d(2) # max pooling
        self.down2 = nn.MaxPool2d(2) # max pooling
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling 
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling

    def forward(self, x, mu, sigma):
        # down
        x1 = self.layer1_down(x) # keep for skip connection
        x2 = self.down1(x1)
        x3 = self.layer2_down(x2) # keep for skip connection
        x4 = self.down2(x3)
        # latent
        x5 = self.latent(x4)
        # up
        x6 = self.up1(x5)
        x7 = self.layer3_up(x6 + x3) # skip connection
        x8 = self.up2(x7)
        x9 = self.layer4_up(x8+ x1) # skip connection
        
        # distributional outputs
        mu_temp, std_temp = mu[:,0,:,:] + x9[:,0,:,:], torch.exp(x9[:,1,:,:] + torch.log(sigma[:,0,:,:])) # to preserve positiveness
        mu_wind, std_wind = mu[:,1,:,:] + x9[:,2,:,:], torch.exp(x9[:,3,:,:] + torch.log(sigma[:,1,:,:])) # to preserve positiveness
        temp_dist = Normal(mu_temp, std_temp)
        wind_dist = Normal(mu_wind, std_wind)
        return x9, temp_dist, wind_dist

class DRUnetPriorVar(nn.Module):
    """ One variable, post-processing mode
    Distributional regression U-Net from Pic et al.:
    Inputs: size 121x240 with 77 channels
    Architecture:
        - 2 decreasing resolution blocks, 2 increasing resolution blocks 
        with skip connections
        - Latent layers: 3x3 conv, BN, ReLU
        - Down blocks: twice 3x3 conv, BN, ReLU + 2x2 max pooling
        - Up blocks: bilinear upsampling+ twice 3x3 conv, BN, ReLU"""
    def __init__(self, in_channels, out_channels, variable):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable = variable
        # layers
        self.layer1_down = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer2_down = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.latent = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3_up = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4_up = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # up and down scalers
        self.down1 = nn.MaxPool2d(2) # max pooling
        self.down2 = nn.MaxPool2d(2) # max pooling
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling 
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling

    def forward(self, x, mu, sigma):
        # down
        x1 = self.layer1_down(x) # keep for skip connection
        x2 = self.down1(x1)
        x3 = self.layer2_down(x2) # keep for skip connection
        x4 = self.down2(x3)
        # latent
        x5 = self.latent(x4)
        # up
        x6 = self.up1(x5)
        x7 = self.layer3_up(x6 + x3) # skip connection
        x8 = self.up2(x7)
        x9 = self.layer4_up(x8+ x1) # skip connection
        
        # distributional outputs
        if self.variable == "2m_temperature":
            mu_var = mu  + x9[:,0,:,:]
            std_var = torch.exp(torch.log(sigma)  + x9[:,1,:,:]) # to preserve positiveness
        elif self.variable == "10m_wind_speed":
            mu_var = mu +  x9[:,0,:,:] 
            std_var = torch.exp(torch.log(sigma) + x9[:,1,:,:]) #torch.exp(torch.log(sigma) * x9[:,2,:,:] + x9[:,3,:,:]) # to preserve positiveness

        var_dist = Normal(mu_var, std_var)
        return x9, var_dist

class DRUnetAll(nn.Module):
    """Encompasses all architectures of DRUnet:
        - 2m temperature, wind speed or both
        - with or without prior
    """
    def __init__(self, in_channels, out_channels, variable, type):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable = variable # 2m_temperature, 10m_wind_speed or both
        self.type = type # with or without prior
        # layers
        self.layer1_down = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer2_down = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.latent = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3_up = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4_up = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # up and down scalers
        self.down1 = nn.MaxPool2d(2) # max pooling
        self.down2 = nn.MaxPool2d(2) # max pooling
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling 
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #bilinear upsampling

    def forward(self, x, mu, sigma):
        # down
        x1 = self.layer1_down(x) # keep for skip connection
        x2 = self.down1(x1)
        x3 = self.layer2_down(x2) # keep for skip connection
        x4 = self.down2(x3)
        # latent
        x5 = self.latent(x4)
        # up
        x6 = self.up1(x5)
        x7 = self.layer3_up(x6 + x3) # skip connection
        x8 = self.up2(x7)
        x9 = self.layer4_up(x8+ x1) # skip connection
        
        # distributional outputs
        if self.type == "prior":
            if self.variable == "both":
                mu_temp, std_temp = mu[:,0,:,:] + x9[:,0,:,:], torch.exp(x9[:,1,:,:] + torch.log(sigma[:,0,:,:])) # to preserve positiveness
                mu_wind, std_wind = mu[:,1,:,:] + x9[:,2,:,:], torch.exp(x9[:,3,:,:] + torch.log(sigma[:,1,:,:])) # to preserve positiveness
                temp_dist = Normal(mu_temp, std_temp)
                wind_dist = Normal(mu_wind, std_wind)
                return x9, temp_dist, wind_dist
                
            elif self.variable == "2m_temperature":
                mu_var = mu  + x9[:,0,:,:]
                std_var = torch.exp(torch.log(sigma)  + x9[:,1,:,:]) # to preserve positiveness
                var_dist = Normal(mu_var, std_var)
                return x9, var_dist
            elif self.variable == "10m_wind_speed":
                mu_var = mu +  x9[:,0,:,:] #mu * x9[:,0,:,:] + x9[:,1,:,:]
                std_var = torch.exp(torch.log(sigma) + x9[:,1,:,:]) #torch.exp(torch.log(sigma) * x9[:,2,:,:] + x9[:,3,:,:]) # to preserve positiveness
                var_dist = Normal(mu_var, std_var)
                return x9, var_dist

        elif self.type == "no_prior":
            if self.variable == "both":
                mu_temp, std_temp = x9[:,0,:,:], torch.exp(x9[:,1,:,:]) # to preserve positiveness
                mu_wind, std_wind = x9[:,2,:,:], torch.exp(x9[:,3,:,:]) # to preserve positiveness
                temp_dist = Normal(mu_temp, std_temp)
                wind_dist = Normal(mu_wind, std_wind)
                return x9, temp_dist, wind_dist

            elif self.variable == "2m_temperature":
                mu_var = x9[:,0,:,:]
                std_var = torch.exp(x9[:,1,:,:]) # to preserve positiveness
                var_dist = Normal(mu_var, std_var)
                return x9, var_dist
            elif self.variable == "10m_wind_speed":
                mu_var = x9[:,0,:,:] 
                std_var = torch.exp( x9[:,1,:,:]) #torch.exp(torch.log(sigma) * x9[:,2,:,:] + x9[:,3,:,:]) # to preserve positiveness
                var_dist = Normal(mu_var, std_var)
                return x9, var_dist


if __name__== "__main__":
    # count number of parameters
    model = DRUnetPrior(74,4)
    print(sum(p.numel() for p in model.parameters()))
    model = SpatialEMOS(67, 120, 240, 4)
    print(sum(p.numel() for p in model.parameters()))
