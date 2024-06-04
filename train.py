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
from processings.dataset import PandasDataset, compute_wind_speed
from sklearn.model_selection import train_test_split
import wandb

def create_training_folder(name):
    # Define a base directory 
    base_dir = 'training_results'
    # Create a new directory with a unique name
    new_folder = os.path.join(base_dir, 'training_' + str(len(os.listdir(base_dir)) + 1) + '_' + name)
    # Create the directory if it doesn't exist
    os.makedirs(new_folder, exist_ok=True)  
    return new_folder


def train(train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, target_column, batch_size, save_every=50):
    # Weight and Biases setup
    wandb.init(
    project = f"S2S_train_MOS", # set the wandb project where this run will be logged
    name = name_experiment,     # give the run a name
    config={                    # track hyperparameters and run metadata
    "learning_rate": lr,
    "architecture": "MOS",
    "variable": target_column,
    "epochs": nb_epoch,
    "batch_size": batch_size
    }
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0.0

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

        # validation each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_batch in val_loader:
                val_mu = val_batch['mu']
                val_sigma = val_batch['sigma']
                val_X = val_batch['input']
                val_y = val_batch['truth']
                val_out_distrib = model(val_mu, val_sigma, val_X, val_y)
                val_loss += criterion(val_out_distrib, val_y).mean()      
        model.train()

        val_loss = val_loss / len(val_loader)
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # log metrics to wandb
        wandb.log({"Train loss": epoch_loss, "Validation loss": val_loss})

        # save epoch losses to csv file
        val_losses.append(val_loss.item())
        train_losses.append(epoch_loss.item())
        L = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
        L.to_csv(result_folder+'/loss.csv', index=False)

        # save model every save_every epochs
        if epoch > 0:
            if epoch % save_every == 0:
                torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')

    # save final model        
    torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')
    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth")) # in wandb

    wandb.finish()
   
if __name__== "__main__":
    # run pipeline for training
    # load data
    data_folder = "../scratch/data/train/"
    train_data = pd.read_json(data_folder+'PPE_OPT_lat=-90.0_lon=0.0_lead=24h.json')
    train_data = compute_wind_speed(train_data)

    # separate train and validation randomly
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True)

    # build dataloaders
    train_data = PandasDataset(train_data, "2m_temperature")
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    val_data = PandasDataset(val_data, "2m_temperature")
    val_loader = DataLoader(val_data, batch_size=10, shuffle=True)

    # model setup and training
    folder = create_training_folder("t2m")
    criterion = crps_normal
    model = MOS(50,3)
    train(train_loader, val_loader, model, 10, 0.01, criterion, folder)




