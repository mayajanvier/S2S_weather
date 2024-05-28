import numpy as np
from model import MOS
import torch
import torch.nn as nn
import torch.optim as optim 
#from weatherbench2.metrics import GaussianCRPS
from metrics import crps_normal
import pandas as pd 
import os
from torch.utils.data import DataLoader, TensorDataset
from processings.dataset import PandasDataset

def create_training_folder(name):
    # Define a base directory 
    base_dir = 'training_results'
    # Create a new directory with a unique name
    new_folder = os.path.join(base_dir, 'training_' + str(len(os.listdir(base_dir)) + 1) + '_' + name)
    # Create the directory if it doesn't exist
    os.makedirs(new_folder, exist_ok=True)  
    return new_folder


def train(train_loader, model, nb_epoch, lr, criterion, result_folder, save_every=50):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0

        for batch in train_loader:
            mu = batch['mu']
            sigma = batch['sigma']
            X = batch['input']
            y = batch['truth']
            optimizer.zero_grad()
            out_distrib = model(mu, sigma, X, y) # parameters 

            loss = criterion(out_distrib, y).mean()  # batch loss
            loss.backward()
            optimizer.step()

            running_loss += loss

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}')

        # save epoch loss to csv file
        with open(result_folder+'/loss.csv', 'a') as f:
            f.write(f'{epoch},{epoch_loss}\n')

        # save model every save_every epochs
        if epoch > 0:
            if epoch % save_every == 0:
                torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')

    # save final model        
    torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')
        
if __name__== "__main__":
    # run pipeline for training
    # load data
    data_folder = "../scratch/"
    train_data = pd.read_json(data_folder+'data_2m_temperature.json')

    # build dataloader
    train_data = PandasDataset(train_data, "2m_temperature")
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    # model setup and training
    folder = create_training_folder("t2m")
    criterion = crps_normal
    model = MOS(50,3)
    train(train_loader, model, 100, 0.01, criterion, folder)




