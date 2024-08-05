import numpy as np
from model import MOS, SpatialMOS
import torch
import torch.nn as nn
import torch.optim as optim 
#from weatherbench2.metrics import GaussianCRPS
from metrics import crps_normal
import pandas as pd 
import os
from torch.utils.data import DataLoader
from processings.dataset import PandasDataset, WeatherDataset
from processings.format_data import compute_wind_speed
from sklearn.model_selection import train_test_split
import wandb

def create_training_folder(name, base_dir='training_results'):
    # Create a new directory with a unique name
    new_folder = os.path.join(base_dir, 'training_' + str(len(os.listdir(base_dir)) + 1) + '_' + name)
    # Create the directory if it doesn't exist
    os.makedirs(new_folder, exist_ok=True)  
    return new_folder


def train(train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, target_column, batch_size, val_mask, save_every=50):
    # Weight and Biases setup
    if "spatial" in name_experiment:
        project = "S2S_SpatialEMOS_ensemble"
        architecture = "SpatialEMOS"
    else:
        project = "S2S_train_MOS"
        architecture = "MOS"

    wandb.init(
    project = project, # set the wandb project where this run will be logged
    name = name_experiment,     # give the run a name
    config={                    # track hyperparameters and run metadata
    "learning_rate": lr,
    "architecture": architecture,
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
            print("min max")
            print(torch.min(X), torch.max(X))
            print(torch.min(mu), torch.max(mu))
            print(torch.min(sigma), torch.max(sigma))
            # print nan in input
            print("nan")
            print(torch.isnan(X).sum())
            print(torch.isnan(mu).sum())
            print(torch.isnan(sigma).sum())

            # print zeros
            print("null")
            print(torch.sum(X == 0))
            print(torch.sum(mu == 0))
            print(torch.sum(sigma == 0))

            # inf
            print('inf')
            print(torch.isinf(X).sum())
            print(torch.isinf(mu).sum())
            print(torch.isinf(sigma).sum())
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
                val_loss_masked = criterion(val_out_distrib, val_y) * val_mask # land sea mask
                val_loss += val_loss_masked.mean()     
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

def trainUNet(train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, batch_size, val_mask, weights, save_every=50):
    """ training in the normalized and detrented space """
    # Weight and Biases setup
    project = "S2S_Unet_ensemble"
    architecture = "Unet"

    wandb.init(
    project = project, # set the wandb project where this run will be logged
    name = name_experiment,     # give the run a name
    config={                    # track hyperparameters and run metadata
    "learning_rate": lr,
    "architecture": architecture,
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
            X = batch['input']
            y = batch['truth']
            optimizer.zero_grad()
            maps, out_distrib_temp, out_distrib_wind = model(X) # parameters 

            
            loss = criterion(out_distrib_temp, y[:,0,:,:]) +  criterion(out_distrib_wind, y[:,1,:,:]) # batch loss
            # weight latitudes
            loss = loss * weights
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss

        # validation each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_batch in val_loader:
                val_X = val_batch['input']
                val_y = val_batch['truth']
                val_maps, val_out_distrib_temp, val_out_distrib_wind = model(val_X)
                # weight latitudes and mask
                val_loss_masked = (criterion(val_out_distrib_temp, val_y[:,0,:,:]) + criterion(val_out_distrib_wind, y[:,1,:,:])) * weights * val_mask # land sea mask
                val_loss += val_loss_masked.mean()     
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

def train_sched(train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, target_column, batch_size, val_mask, save_every=50, use_sgd=False, momentum=0.9, weight_decay=0):
    # Weight and Biases setup
    if "spatial" in name_experiment:
        project = "S2S_SpatialEMOS_ensemble"
        architecture = "SpatialEMOS"
    else:
        project = "S2S_train_MOS"
        architecture = "MOS"

    wandb.init(
        project=project,  # set the wandb project where this run will be logged
        name=name_experiment,  # give the run a name
        config={  # track hyperparameters and run metadata
            "learning_rate": lr,
            "architecture": architecture,
            "variable": target_column,
            "epochs": nb_epoch,
            "batch_size": batch_size,
            "optimizer": "SGD" if use_sgd else "Adam",
            "momentum": momentum if use_sgd else None,
            "weight_decay": weight_decay
        }
    )
    
    if use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)
    
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
            out_distrib = model(mu, sigma, X, y)  # parameters 

            loss = criterion(out_distrib, y).mean()  # batch loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

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
                val_loss_masked = criterion(val_out_distrib, val_y) * val_mask  # land sea mask
                val_loss += val_loss_masked.mean().item()
        model.train()

        val_loss = val_loss / len(val_loader)
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # log metrics to wandb
        wandb.log({"Train loss": epoch_loss, "Validation loss": val_loss})

        # save epoch losses to csv file
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)
        L = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
        L.to_csv(result_folder + '/loss.csv', index=False)

        # save model every save_every epochs
        if epoch > 0 and epoch % save_every == 0:
            torch.save(model.state_dict(), result_folder + f'/model_{epoch}.pth')

        # Step the scheduler
        scheduler.step(val_loss)

    # save final model        
    torch.save(model.state_dict(), result_folder + f'/model_{epoch}.pth')

if __name__== "__main__":
    ### SINGLE MOS 
    # run pipeline for training
    # load data
    # data_folder = "../scratch/data/train/"
    # train_data = pd.read_json(data_folder+'PPE_OPT_lat=-90.0_lon=0.0_lead=24h.json')
    # train_data = compute_wind_speed(train_data)

    # # separate train and validation randomly
    # train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True)

    # # build dataloaders
    # train_data = PandasDataset(train_data, "2m_temperature")
    # train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    # val_data = PandasDataset(val_data, "2m_temperature")
    # val_loader = DataLoader(val_data, batch_size=10, shuffle=True)

    # # model setup and training
    # folder = create_training_folder("t2m")
    # criterion = crps_normal
    # model = MOS(50,3)
    # train(train_loader, val_loader, model, 10, 0.01, criterion, folder)

    ### SPATIAL MOS 
    # data 
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

    val_dataset = WeatherDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable="10m_wind_speed",
        lead_time_idx=28,
        valid_years=[1996,2017],
        valid_months=[1,1],
        subset="val")
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    # model setup and training
    name_experiment = "spatial_wind_speed"
    folder = create_training_folder(name_experiment)
    model = SpatialMOS(47, 121, 240, 3)
    criterion = crps_normal
    train(
        train_loader,
        val_loader,
        model,
        10,
        0.01,
        criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        target_column="10m_wind_speed",
        batch_size=128,
        save_every=50)
    




