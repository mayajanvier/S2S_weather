import numpy as np
import pandas as pd 
import os
import torch
import torch.nn as nn
import torch.optim as optim 
from metrics import crps_normal
from torch.utils.data import DataLoader
import wandb
import psutil

# utils for training setup
def create_training_folder(name, base_dir='training_results'):
    # Create a new directory with a unique name
    new_folder = os.path.join(base_dir, 'training_' + str(len(os.listdir(base_dir)) + 1) + '_' + name)
    # Create the directory if it doesn't exist
    os.makedirs(new_folder, exist_ok=True)  
    return new_folder

### EMOS 
def train(train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, target_column, batch_size, val_mask, save_every=5):
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

            optimizer.zero_grad()
            out_distrib = model(mu, sigma, X, y) # parameters 

            loss = criterion(out_distrib, y).mean()  # batch loss
            loss.backward()
            optimizer.step()

            running_loss += loss
            del mu, sigma, X, y, out_distrib # free memory

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
                del val_mu, val_sigma, val_X, val_y, val_out_distrib # free memory  
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

def train_sched(train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, target_column, batch_size, val_mask, save_every=5, use_sgd=False, momentum=0.9, weight_decay=0):
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

# DRUnets 
## BOTH
def trainUNet(train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, batch_size, val_mask, weights, device, save_every=5):
    """ Both, forecasting mode"""
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

    print("val mask", val_mask.shape)
    print("weights", weights.shape)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    train_losses_temp = []
    train_losses_wind = []
    val_losses_temp = []
    val_losses_wind = []
    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0.0
        running_loss_temp = 0.0
        running_loss_wind = 0.0
        # test 
        # for _ in range(20):
        #     X = torch.rand((64,70,120,240)).to(device)
        #     y = torch.rand((64,70,120,240)).to(device)
        #     optimizer.zero_grad()
        #     maps, out_distrib_temp, out_distrib_wind = model(X) # parameters 

        #     print()
        #     loss = criterion(out_distrib_temp, y[:,0,:,:]) +  criterion(out_distrib_wind, y[:,1,:,:]) # batch loss
        #     # weight latitudes
        #     loss = loss * weights
        #     loss = loss.mean()
        #     loss.backward()
        #     optimizer.step()

        #     running_loss += loss
        #     print("Loss: ", running_loss)
        i = 0
        for batch in train_loader:
            X = batch['input'].to(device)
            y = batch['truth'].to(device)
            optimizer.zero_grad()
            maps, out_distrib_temp, out_distrib_wind = model(X) # parameters 
            
            temp_loss = criterion(out_distrib_temp, y[:,0,:,:]) #/ 2.1 # weight with initial loss value
            wind_loss = criterion(out_distrib_wind, y[:,1,:,:]) #/ 1.5 # weight with initial loss value
            loss = temp_loss + wind_loss # batch loss
            # weight latitudes and normalize 
            #loss = (loss * weights).sum() / weights.sum()
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # convert to float to avoid accumulation of the graph
            running_loss_temp += temp_loss.mean().item()
            running_loss_wind += wind_loss.mean().item()
            # running_loss_temp += ((temp_loss * weights).sum()/ weights.sum()).item()
            # running_loss_wind += ((wind_loss * weights).sum()/ weights.sum()).item()
        
            if i% 100 == 0:
                print(i, psutil.Process().memory_info().rss / (1024 * 1024), "temp", out_distrib_temp.loc.mean().item(), out_distrib_temp.scale.mean().item(), "wind", out_distrib_wind.loc.mean().item(), out_distrib_wind.scale.mean().item())
                print(f'Loss{i}', running_loss)
                print(f'Loss_temp{i}', running_loss_temp, "Loss_wind", running_loss_wind)
            i += 1
            del maps, out_distrib_temp, out_distrib_wind # free memory
            del temp_loss, wind_loss # free memory

        # validation each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_loss_temp = 0.0
            val_loss_wind = 0.0
            for val_batch in val_loader:
                val_X = val_batch['input'].to(device)
                val_y = val_batch['truth'].to(device)
                val_maps, val_out_distrib_temp, val_out_distrib_wind = model(val_X)
                # weight latitudes and mask
                val_temp_loss = criterion(val_out_distrib_temp, val_y[:,0,:,:])
                val_wind_loss = criterion(val_out_distrib_wind, val_y[:,1,:,:])
                val_weights = weights * val_mask
                #val_loss_masked = ((val_temp_loss + val_wind_loss)* val_weights).sum() / val_weights.sum() # land sea mask
                val_loss_masked = (val_temp_loss + val_wind_loss) * val_weights
                val_loss += val_loss_masked.mean().item()
                val_loss_temp += (val_temp_loss * val_weights).mean().item()
                val_loss_wind += (val_wind_loss * val_weights).mean().item() 
                del val_maps, val_out_distrib_temp, val_out_distrib_wind # free memory 
                del val_temp_loss, val_wind_loss, val_weights, val_loss_masked # free memory
        model.train()

        val_loss = val_loss / len(val_loader)
        val_loss_temp = val_loss_temp / len(val_loader)
        val_loss_wind = val_loss_wind / len(val_loader)
        epoch_loss = running_loss / len(train_loader)
        epoch_loss_temp = running_loss_temp / len(train_loader)
        epoch_loss_wind = running_loss_wind / len(train_loader)
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # log metrics to wandb
        wandb.log({
            "Train loss": epoch_loss,
            "Validation loss": val_loss,
            "Train temp loss": epoch_loss_temp,
            "Train wind loss": epoch_loss_wind,
            "Validation temp loss": val_loss_temp,
            "Validation wind loss": val_loss_wind})

        # save epoch losses to csv file
        val_losses.append(val_loss)
        val_losses_temp.append(val_loss_temp)
        val_losses_wind.append(val_loss_wind)
        train_losses.append(epoch_loss)
        train_losses_temp.append(epoch_loss_temp)
        train_losses_wind.append(epoch_loss_wind)
        L = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_temp_loss': train_losses_temp,
            'val_temp_loss': val_losses_temp,
            'train_wind_loss': train_losses_wind,
            'val_wind_loss': val_losses_wind})
        L.to_csv(result_folder+'/loss.csv', index=False)

        # save model every save_every epochs
        if epoch > 0:
            if epoch % save_every == 0:
                torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')


    # save final model        
    torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')
    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth")) # in wandb

    wandb.finish()

def trainUNetPrior(train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, batch_size, val_mask, weights, device, save_every=5):
    """ Both, post-processing mode """
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

    print("val mask", val_mask.shape)
    print("weights", weights.shape)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    train_losses_temp = []
    train_losses_wind = []
    val_losses_temp = []
    val_losses_wind = []
    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0.0
        running_loss_temp = 0.0
        running_loss_wind = 0.0
        i = 0
        for batch in train_loader:
            X = batch['input'].to(device)
            y = batch['truth'].to(device)
            mu = batch['mu'].to(device)
            sigma = batch['sigma'].to(device)
            optimizer.zero_grad()
            maps, out_distrib_temp, out_distrib_wind = model(X, mu, sigma) # parameters 
            
            temp_loss = criterion(out_distrib_temp, y[:,0,:,:])
            wind_loss = criterion(out_distrib_wind, y[:,1,:,:])
            loss = temp_loss + wind_loss # batch loss
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # convert to float to avoid accumulation of the graph
            running_loss_temp += temp_loss.mean().item()
            running_loss_wind += wind_loss.mean().item()
        
            if i% 100 == 0:
                print(i, psutil.Process().memory_info().rss / (1024 * 1024), "temp", out_distrib_temp.loc.mean().item(), out_distrib_temp.scale.mean().item(), "wind", out_distrib_wind.loc.mean().item(), out_distrib_wind.scale.mean().item())
                print(f'Loss{i}', running_loss)
                print(f'Loss_temp{i}', running_loss_temp, "Loss_wind", running_loss_wind)
            i += 1
            del maps, out_distrib_temp, out_distrib_wind # free memory
            del temp_loss, wind_loss # free memory

        # validation each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_loss_temp = 0.0
            val_loss_wind = 0.0
            for val_batch in val_loader:
                val_X = val_batch['input'].to(device)
                val_y = val_batch['truth'].to(device)
                val_mu = val_batch['mu'].to(device)
                val_sigma = val_batch['sigma'].to(device)
                val_maps, val_out_distrib_temp, val_out_distrib_wind = model(val_X, val_mu, val_sigma)
                # weight latitudes and mask
                val_temp_loss = criterion(val_out_distrib_temp, val_y[:,0,:,:])
                val_wind_loss = criterion(val_out_distrib_wind, val_y[:,1,:,:])
                val_weights = weights * val_mask
                val_loss_masked = (val_temp_loss + val_wind_loss) * val_weights
                val_loss += val_loss_masked.mean().item()
                val_loss_temp += (val_temp_loss * val_weights).mean().item()
                val_loss_wind += (val_wind_loss * val_weights).mean().item() 
                del val_maps, val_out_distrib_temp, val_out_distrib_wind # free memory 
                del val_temp_loss, val_wind_loss, val_weights, val_loss_masked # free memory
        model.train()

        val_loss = val_loss / len(val_loader)
        val_loss_temp = val_loss_temp / len(val_loader)
        val_loss_wind = val_loss_wind / len(val_loader)
        epoch_loss = running_loss / len(train_loader)
        epoch_loss_temp = running_loss_temp / len(train_loader)
        epoch_loss_wind = running_loss_wind / len(train_loader)
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # log metrics to wandb
        wandb.log({
            "Train loss": epoch_loss,
            "Validation loss": val_loss,
            "Train temp loss": epoch_loss_temp,
            "Train wind loss": epoch_loss_wind,
            "Validation temp loss": val_loss_temp,
            "Validation wind loss": val_loss_wind})

        # save epoch losses to csv file
        val_losses.append(val_loss)
        val_losses_temp.append(val_loss_temp)
        val_losses_wind.append(val_loss_wind)
        train_losses.append(epoch_loss)
        train_losses_temp.append(epoch_loss_temp)
        train_losses_wind.append(epoch_loss_wind)
        L = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_temp_loss': train_losses_temp,
            'val_temp_loss': val_losses_temp,
            'train_wind_loss': train_losses_wind,
            'val_wind_loss': val_losses_wind})
        L.to_csv(result_folder+'/loss.csv', index=False)

        # save model every save_every epochs
        if epoch > 0:
            if epoch % save_every == 0:
                torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')


    # save final model        
    torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')
    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth")) # in wandb

    wandb.finish()

def trainUNetPriorSep(variable, train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, batch_size, val_mask, weights, device, save_every=5):
    """ Training with both, only temperature or only wind speed losses
    within both architecture, post-processing mode """
    # Weight and Biases setup
    project = "S2S_Unet_ensemble"
    architecture = "Unet"
    print(variable)


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

    print("val mask", val_mask.shape)
    print("weights", weights.shape)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    train_losses_temp = []
    train_losses_wind = []
    val_losses_temp = []
    val_losses_wind = []
    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0.0
        running_loss_temp = 0.0
        running_loss_wind = 0.0
        i = 0
        for batch in train_loader:
            X = batch['input'].to(device)
            y = batch['truth'].to(device)
            mu = batch['mu'].to(device)
            sigma = batch['sigma'].to(device)
            optimizer.zero_grad()
            maps, out_distrib_temp, out_distrib_wind = model(X, mu, sigma) # parameters 
            
            temp_loss = criterion(out_distrib_temp, y[:,0,:,:])
            wind_loss = criterion(out_distrib_wind, y[:,1,:,:])
            if variable == "2m_temperature":
                loss = temp_loss
            elif variable == "10m_wind_speed":
                loss = wind_loss
            elif variable == "both":
                loss = temp_loss + wind_loss # batch loss
            # weight latitudes and normalize 
            #loss = (loss * weights).sum() / weights.sum()
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # convert to float to avoid accumulation of the graph
            running_loss_temp += temp_loss.mean().item()
            running_loss_wind += wind_loss.mean().item()
            # running_loss_temp += ((temp_loss * weights).sum()/ weights.sum()).item()
            # running_loss_wind += ((wind_loss * weights).sum()/ weights.sum()).item()
        
            if i% 100 == 0:
                print(i, psutil.Process().memory_info().rss / (1024 * 1024), "temp", out_distrib_temp.loc.mean().item(), out_distrib_temp.scale.mean().item(), "wind", out_distrib_wind.loc.mean().item(), out_distrib_wind.scale.mean().item())
                print(f'Loss{i}', running_loss)
                print(f'Loss_temp{i}', running_loss_temp, "Loss_wind", running_loss_wind)
            i += 1
            del maps, out_distrib_temp, out_distrib_wind # free memory
            del temp_loss, wind_loss # free memory

        # validation each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_loss_temp = 0.0
            val_loss_wind = 0.0
            for val_batch in val_loader:
                val_X = val_batch['input'].to(device)
                val_y = val_batch['truth'].to(device)
                val_mu = val_batch['mu'].to(device)
                val_sigma = val_batch['sigma'].to(device)
                val_maps, val_out_distrib_temp, val_out_distrib_wind = model(val_X, val_mu, val_sigma)
                # weight latitudes and mask
                val_temp_loss = criterion(val_out_distrib_temp, val_y[:,0,:,:])
                val_wind_loss = criterion(val_out_distrib_wind, val_y[:,1,:,:])
                val_weights = weights * val_mask
                if variable == "2m_temperature":
                    val_loss_masked = val_temp_loss * val_weights
                elif variable == "10m_wind_speed":
                    val_loss_masked = val_wind_loss * val_weights
                elif variable == "both":
                    val_loss_masked = (val_temp_loss + val_wind_loss) * val_weights
                val_loss += val_loss_masked.mean().item()
                val_loss_temp += (val_temp_loss * val_weights).mean().item()
                val_loss_wind += (val_wind_loss * val_weights).mean().item() 
                del val_maps, val_out_distrib_temp, val_out_distrib_wind # free memory 
                del val_temp_loss, val_wind_loss, val_weights, val_loss_masked # free memory
        model.train()

        val_loss = val_loss / len(val_loader)
        val_loss_temp = val_loss_temp / len(val_loader)
        val_loss_wind = val_loss_wind / len(val_loader)
        epoch_loss = running_loss / len(train_loader)
        epoch_loss_temp = running_loss_temp / len(train_loader)
        epoch_loss_wind = running_loss_wind / len(train_loader)
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # log metrics to wandb
        wandb.log({
            "Train loss": epoch_loss,
            "Validation loss": val_loss,
            "Train temp loss": epoch_loss_temp,
            "Train wind loss": epoch_loss_wind,
            "Validation temp loss": val_loss_temp,
            "Validation wind loss": val_loss_wind})

        # save epoch losses to csv file
        val_losses.append(val_loss)
        val_losses_temp.append(val_loss_temp)
        val_losses_wind.append(val_loss_wind)
        train_losses.append(epoch_loss)
        train_losses_temp.append(epoch_loss_temp)
        train_losses_wind.append(epoch_loss_wind)
        L = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_temp_loss': train_losses_temp,
            'val_temp_loss': val_losses_temp,
            'train_wind_loss': train_losses_wind,
            'val_wind_loss': val_losses_wind})
        L.to_csv(result_folder+'/loss.csv', index=False)

        # save model every save_every epochs
        if epoch > 0:
            if epoch % save_every == 0:
                torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')


    # save final model        
    torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')
    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth")) # in wandb

    wandb.finish()

## SINGLE
def trainUNetVar(variable, train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, batch_size, val_mask, weights, device, save_every=5):
    """ Single, forecasting mode """
    # Weight and Biases setup
    project = "S2S_Unet_ensemble"
    architecture = "Unet"
    print(variable)


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

    print("val mask", val_mask.shape)
    print("weights", weights.shape)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    train_losses_temp = []
    train_losses_wind = []
    val_losses_temp = []
    val_losses_wind = []
    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0.0
        running_loss_temp = 0.0
        running_loss_wind = 0.0
        i = 0
        for batch in train_loader:
            X = batch['input'].to(device)
            if variable == "2m_temperature":
                y = batch['truth'].to(device)[:,0,:,:]
                #mu = batch['mu'].to(device)[:,0,:,:]
                #sigma = batch['sigma'].to(device)[:,0,:,:]
            elif variable == "10m_wind_speed":
                y = batch['truth'].to(device)[:,1,:,:]
                #mu = batch['mu'].to(device)[:,1,:,:]
                #sigma = batch['sigma'].to(device)[:,1,:,:]

            optimizer.zero_grad()
            maps, out_distrib = model(X) # parameters 
            loss = criterion(out_distrib, y)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # convert to float to avoid accumulation of the graph
        
            if i% 100 == 0:
                print(f'Loss{i}', running_loss)
            i += 1
            del maps, out_distrib # free memory

        # validation each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_loss_temp = 0.0
            val_loss_wind = 0.0
            for val_batch in val_loader:
                val_X = val_batch['input'].to(device)
                if variable == "2m_temperature":
                    val_y = val_batch['truth'].to(device)[:,0,:,:]
                    #val_mu = val_batch['mu'].to(device)[:,0,:,:]
                    #val_sigma = val_batch['sigma'].to(device)[:,0,:,:]
                elif variable == "10m_wind_speed":
                    val_y = val_batch['truth'].to(device)[:,1,:,:]
                    #val_mu = val_batch['mu'].to(device)[:,1,:,:]
                    #val_sigma = val_batch['sigma'].to(device)[:,1,:,:]
                val_maps, val_out_distrib = model(val_X)
                # weight latitudes and mask
                val_loss_out = criterion(val_out_distrib, val_y)
                val_weights = weights * val_mask
                #val_loss_masked = ((val_temp_loss + val_wind_loss)* val_weights).sum() / val_weights.sum() # land sea mask
                val_loss_masked = val_loss_out * val_weights
                val_loss += val_loss_masked.mean().item()

                del val_maps, val_out_distrib # free memory 
                del val_weights, val_loss_masked # free memory
        model.train()

        val_loss = val_loss / len(val_loader)
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # log metrics to wandb
        wandb.log({
            "Train loss": epoch_loss,
            "Validation loss": val_loss})

        # save epoch losses to csv file
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)
        L = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses
            })
        L.to_csv(result_folder+'/loss.csv', index=False)

        # save model every save_every epochs
        if epoch > 0:
            if epoch % save_every == 0:
                torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')


    # save final model        
    torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')
    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth")) # in wandb

    wandb.finish()

def trainUNetPriorVar(variable, train_loader, val_loader, model, nb_epoch, lr, criterion, result_folder, name_experiment, batch_size, val_mask, weights, device, save_every=5):
    """ Single, post-processing mode """
    # Weight and Biases setup
    project = "S2S_Unet_ensemble"
    architecture = "Unet"
    print(variable)


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

    print("val mask", val_mask.shape)
    print("weights", weights.shape)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    train_losses_temp = []
    train_losses_wind = []
    val_losses_temp = []
    val_losses_wind = []
    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0.0
        running_loss_temp = 0.0
        running_loss_wind = 0.0
        i = 0
        for batch in train_loader:
            X = batch['input'].to(device)
            if variable == "2m_temperature":
                y = batch['truth'].to(device)[:,0,:,:]
                mu = batch['mu'].to(device)[:,0,:,:]
                sigma = batch['sigma'].to(device)[:,0,:,:]
            elif variable == "10m_wind_speed":
                y = batch['truth'].to(device)[:,1,:,:]
                mu = batch['mu'].to(device)[:,1,:,:]
                sigma = batch['sigma'].to(device)[:,1,:,:]

            optimizer.zero_grad()
            maps, out_distrib = model(X, mu, sigma) # parameters 
            loss = criterion(out_distrib, y)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # convert to float to avoid accumulation of the graph
        
            if i% 100 == 0:
                print(f'Loss{i}', running_loss)
            i += 1
            del maps, out_distrib # free memory

        # validation each epoch
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_loss_temp = 0.0
            val_loss_wind = 0.0
            for val_batch in val_loader:
                val_X = val_batch['input'].to(device)
                if variable == "2m_temperature":
                    val_y = val_batch['truth'].to(device)[:,0,:,:]
                    val_mu = val_batch['mu'].to(device)[:,0,:,:]
                    val_sigma = val_batch['sigma'].to(device)[:,0,:,:]
                elif variable == "10m_wind_speed":
                    val_y = val_batch['truth'].to(device)[:,1,:,:]
                    val_mu = val_batch['mu'].to(device)[:,1,:,:]
                    val_sigma = val_batch['sigma'].to(device)[:,1,:,:]
                val_maps, val_out_distrib = model(val_X, val_mu, val_sigma)
                # weight latitudes and mask
                val_loss_out = criterion(val_out_distrib, val_y)
                val_weights = weights * val_mask
                val_loss_masked = val_loss_out * val_weights
                val_loss += val_loss_masked.mean().item()

                del val_maps, val_out_distrib # free memory 
                del val_weights, val_loss_masked # free memory
        model.train()

        val_loss = val_loss / len(val_loader)
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # log metrics to wandb
        wandb.log({
            "Train loss": epoch_loss,
            "Validation loss": val_loss})

        # save epoch losses to csv file
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)
        L = pd.DataFrame({
            'train_loss': train_losses,
            'val_loss': val_losses
            })
        L.to_csv(result_folder+'/loss.csv', index=False)

        # save model every save_every epochs
        if epoch > 0:
            if epoch % save_every == 0:
                torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')


    # save final model        
    torch.save(model.state_dict(), result_folder+f'/model_{epoch}.pth')
    #torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth")) # in wandb

    wandb.finish()

    




