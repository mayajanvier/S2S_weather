import argparse
import pandas as pd
from model import SpatialEMOS, DRUnet, DRUnetPrior, DRUnetVar, DRUnetPriorVar
from processings.dataset import WeatherYearEnsembleDataset, WeatherEnsembleDatasetMM, WeatherYearEnsembleDatasetNorm
from train import train, create_training_folder, train_sched, trainUNet, trainUNetPrior, trainUNetPriorSep, trainUNetVar, trainUNetPriorVar
from torch.utils.data import DataLoader
from metrics import crps_normal
import torch
import json
from sklearn.model_selection import train_test_split
import xarray as xr
import numpy as np

### EMOS ### 
def main_spatial_ens(variable, lead_time, valid_years, valid_months, batch_size, lr, nb_epoch, name_experiment, base_dir):
    """ Train models for this variable, lead_time, 
    on valid months data, for all latitude/longitude """
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"

    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask, dtype=torch.float)
    folder = create_training_folder(name_experiment, base_dir)
    
    train_dataset = WeatherEnsembleDatasetMM(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = WeatherEnsembleDatasetMM(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # model setup and training
    model = SpatialEMOS(67, 121, 240, 3) #log_nostd
    criterion = crps_normal

    # write setup in json
    params = {
        "variable": variable,
        "lead_time": lead_time,
        "valid_years": valid_years,
        "valid_months": valid_months,
        "batch_size": batch_size,
        "lr": lr,
        "nb_epoch": nb_epoch,
        "name_experiment": name_experiment,
        "base_dir": base_dir
    }
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)

    # training
    train(
        train_loader,
        val_loader,
        model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        target_column=variable,
        batch_size=batch_size,
        val_mask=land_sea_mask,
        save_every=5)    
    
### DRUNet BOTH ###
# DRUNet both [Month Lead agg]
def main_Unet_ens(valid_years, batch_size, lr, nb_epoch, name_experiment, device, base_dir):
    """ Train Unet for wind and temperature, all lead_times, 
    on valid months data, spatially"""
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    folder = create_training_folder(name_experiment, base_dir)
    print(folder)
    
    train_dataset = WeatherYearEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )

    val_dataset = WeatherYearEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )
    
    lat_beg = train_dataset.latitude_beg
    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask[lat_beg:,:], dtype=torch.float) #120x240
    land_sea_mask = land_sea_mask.to(device) 

    # model setup and training
    model = DRUnet(74,4).to(device)
    print("device", device)
    criterion = crps_normal
    weights = np.cos(np.deg2rad(train_dataset.latitude)) # (120,)
    #weights = weights[1:] # (120,)
    weights = weights.reshape(-1, 1) # (120, 1) for broadcasting during multiplication
    # repeat to match the shape of the grid
    #weights = np.repeat(weights, 240).reshape(121, 240)
    #weights = weights[1:,:] # 120x240
    # batch size: actually not needed, spatial weighting only 
    #weights = np.repeat(weights[np.newaxis,:,:], batch_size, axis=0) # 120x240 -> 1x120x240 -> batch_sizex120x240
    weights = torch.tensor(weights, dtype=torch.float) # batch_sizex120x240
    weights = weights.to(device)

    # write setup in json
    params = {
        "valid_years": valid_years,
        "batch_size": batch_size,
        "lr": lr,
        "nb_epoch": nb_epoch,
        "name_experiment": name_experiment,
        "base_dir": base_dir
    }
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)

    # training
    trainUNet(
        train_loader,
        val_loader,
        model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        batch_size=batch_size,
        val_mask=land_sea_mask,
        weights = weights,
        device = device,
        save_every=5)    

# DRUNet both [General Agg]
def main_Unet_ensNorm(valid_years, batch_size, lr, nb_epoch, name_experiment, device, base_dir):
    """ Train Unet for wind and temperature, all lead_times, 
    on valid months data, spatially"""
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    folder = create_training_folder(name_experiment, base_dir)
    print(folder)
    
    train_dataset = WeatherYearEnsembleDatasetNorm(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )

    val_dataset = WeatherYearEnsembleDatasetNorm(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )
    
    lat_beg = train_dataset.latitude_beg
    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask[lat_beg:,:], dtype=torch.float) #120x240
    land_sea_mask = land_sea_mask.to(device) 

    # model setup and training
    model = DRUnet(74,4).to(device)
    print("device", device)
    criterion = crps_normal
    weights = np.cos(np.deg2rad(train_dataset.latitude)) # (120,)
    #weights = weights[1:] # (120,)
    weights = weights.reshape(-1, 1) # (120, 1) for broadcasting during multiplication
    # repeat to match the shape of the grid
    #weights = np.repeat(weights, 240).reshape(121, 240)
    #weights = weights[1:,:] # 120x240
    # batch size: actually not needed, spatial weighting only 
    #weights = np.repeat(weights[np.newaxis,:,:], batch_size, axis=0) # 120x240 -> 1x120x240 -> batch_sizex120x240
    weights = torch.tensor(weights, dtype=torch.float) # batch_sizex120x240
    weights = weights.to(device)

    # write setup in json
    params = {
        "valid_years": valid_years,
        "batch_size": batch_size,
        "lr": lr,
        "nb_epoch": nb_epoch,
        "name_experiment": name_experiment,
        "base_dir": base_dir
    }
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)

    # training
    trainUNet(
        train_loader,
        val_loader,
        model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        batch_size=batch_size,
        val_mask=land_sea_mask,
        weights = weights,
        device = device,
        save_every=5)    

# DRUNet+prior both [Month Lead Agg] 
def main_Unet_ensPrior(valid_years, batch_size, lr, nb_epoch, name_experiment, device, base_dir):
    """ Train Unet for wind and temperature, all lead_times, 
    on valid months data, spatially"""
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    folder = create_training_folder(name_experiment, base_dir)
    print(folder)
    
    train_dataset = WeatherYearEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )

    val_dataset = WeatherYearEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )
    
    lat_beg = train_dataset.latitude_beg
    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask[lat_beg:,:], dtype=torch.float) #120x240
    land_sea_mask = land_sea_mask.to(device) 

    # model setup and training
    model = DRUnetPrior(74,4).to(device)
    print("device", device)
    criterion = crps_normal
    weights = np.cos(np.deg2rad(train_dataset.latitude)) # (120,)
    #weights = weights[1:] # (120,)
    weights = weights.reshape(-1, 1) # (120, 1) for broadcasting during multiplication
    # repeat to match the shape of the grid
    #weights = np.repeat(weights, 240).reshape(121, 240)
    #weights = weights[1:,:] # 120x240
    # batch size: actually not needed, spatial weighting only 
    #weights = np.repeat(weights[np.newaxis,:,:], batch_size, axis=0) # 120x240 -> 1x120x240 -> batch_sizex120x240
    weights = torch.tensor(weights, dtype=torch.float) # batch_sizex120x240
    weights = weights.to(device)

    # write setup in json
    params = {
        "valid_years": valid_years,
        "batch_size": batch_size,
        "lr": lr,
        "nb_epoch": nb_epoch,
        "name_experiment": name_experiment,
        "base_dir": base_dir
    }
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)

    # training
    trainUNetPrior(
        train_loader,
        val_loader,
        model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        batch_size=batch_size,
        val_mask=land_sea_mask,
        weights = weights,
        device = device,
        save_every=5)    

# exp both structure, post-processing, month lead agg 
def main_Unet_ensPrior_sep(valid_years, batch_size, lr, nb_epoch, name_experiment, device, base_dir, variable):
    """ Train Unet for wind and temperature, all lead_times, 
    on valid months data, spatially"""
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    folder = create_training_folder(name_experiment, base_dir)
    print(folder)
    
    train_dataset = WeatherYearEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )

    val_dataset = WeatherYearEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )
    
    lat_beg = train_dataset.latitude_beg
    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask[lat_beg:,:], dtype=torch.float) #120x240
    land_sea_mask = land_sea_mask.to(device) 

    # model setup and training
    model = DRUnetPrior(74,4).to(device)
    print("device", device)
    criterion = crps_normal
    weights = np.cos(np.deg2rad(train_dataset.latitude)) # (120,)
    #weights = weights[1:] # (120,)
    weights = weights.reshape(-1, 1) # (120, 1) for broadcasting during multiplication
    # repeat to match the shape of the grid
    #weights = np.repeat(weights, 240).reshape(121, 240)
    #weights = weights[1:,:] # 120x240
    # batch size: actually not needed, spatial weighting only 
    #weights = np.repeat(weights[np.newaxis,:,:], batch_size, axis=0) # 120x240 -> 1x120x240 -> batch_sizex120x240
    weights = torch.tensor(weights, dtype=torch.float) # batch_sizex120x240
    weights = weights.to(device)

    # write setup in json
    params = {
        "valid_years": valid_years,
        "batch_size": batch_size,
        "lr": lr,
        "nb_epoch": nb_epoch,
        "name_experiment": name_experiment,
        "base_dir": base_dir
    }
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)

    # training
    trainUNetPriorSep(
        variable,
        train_loader,
        val_loader,
        model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        batch_size=batch_size,
        val_mask=land_sea_mask,
        weights = weights,
        device = device,
        save_every=5)    

### DRUNet SINGLE ###
# DRUNet single [General Agg]
def main_Unet_ensNorm_single(valid_years, batch_size, lr, nb_epoch, name_experiment, device, base_dir, variable):
    """ Train Unet for wind and temperature, all lead_times, 
    on valid months data, spatially"""
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    folder = create_training_folder(name_experiment, base_dir)
    print(folder)
    
    train_dataset = WeatherYearEnsembleDatasetNorm(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )

    val_dataset = WeatherYearEnsembleDatasetNorm(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )
    
    lat_beg = train_dataset.latitude_beg
    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask[lat_beg:,:], dtype=torch.float) #120x240
    land_sea_mask = land_sea_mask.to(device) 

    # model setup and training
    # TODO: DRUnetVar(77,2, variable)
    model = DRUnetVar(77,4, variable).to(device)
    print("device", device)
    criterion = crps_normal
    weights = np.cos(np.deg2rad(train_dataset.latitude)) # (120,)
    #weights = weights[1:] # (120,)
    weights = weights.reshape(-1, 1) # (120, 1) for broadcasting during multiplication
    # repeat to match the shape of the grid
    #weights = np.repeat(weights, 240).reshape(121, 240)
    #weights = weights[1:,:] # 120x240
    # batch size: actually not needed, spatial weighting only 
    #weights = np.repeat(weights[np.newaxis,:,:], batch_size, axis=0) # 120x240 -> 1x120x240 -> batch_sizex120x240
    weights = torch.tensor(weights, dtype=torch.float) # batch_sizex120x240
    weights = weights.to(device)

    # write setup in json
    params = {
        "valid_years": valid_years,
        "batch_size": batch_size,
        "lr": lr,
        "nb_epoch": nb_epoch,
        "name_experiment": name_experiment,
        "base_dir": base_dir
    }
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)

    # training
    trainUNetVar(
        variable,
        train_loader,
        val_loader,
        model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        batch_size=batch_size,
        val_mask=land_sea_mask,
        weights = weights,
        device = device,
        save_every=5)    

# DRUNet+prior single [Month Lead Agg]
def main_Unet_ensPrior_single(valid_years, batch_size, lr, nb_epoch, name_experiment, device, base_dir, variable):
    """ Train Unet for wind and temperature, all lead_times, 
    on valid months data, spatially"""
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    folder = create_training_folder(name_experiment, base_dir)
    print(folder)
    
    train_dataset = WeatherYearEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )

    val_dataset = WeatherYearEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        #pin_memory=True
        )
    
    lat_beg = train_dataset.latitude_beg
    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask[lat_beg:,:], dtype=torch.float) #120x240
    land_sea_mask = land_sea_mask.to(device) 

    # model setup and training
    model = DRUnetPriorVar(77,2, variable).to(device)
    print("device", device)
    criterion = crps_normal
    weights = np.cos(np.deg2rad(train_dataset.latitude)) # (120,)
    #weights = weights[1:] # (120,)
    weights = weights.reshape(-1, 1) # (120, 1) for broadcasting during multiplication
    # repeat to match the shape of the grid
    #weights = np.repeat(weights, 240).reshape(121, 240)
    #weights = weights[1:,:] # 120x240
    # batch size: actually not needed, spatial weighting only 
    #weights = np.repeat(weights[np.newaxis,:,:], batch_size, axis=0) # 120x240 -> 1x120x240 -> batch_sizex120x240
    weights = torch.tensor(weights, dtype=torch.float) # batch_sizex120x240
    weights = weights.to(device)

    # write setup in json
    params = {
        "valid_years": valid_years,
        "batch_size": batch_size,
        "lr": lr,
        "nb_epoch": nb_epoch,
        "name_experiment": name_experiment,
        "base_dir": base_dir
    }
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)
  
    # training
    trainUNetPriorVar(
        variable,
        train_loader,
        val_loader,
        model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        batch_size=batch_size,
        val_mask=land_sea_mask,
        weights = weights,
        device = device,
        save_every=5)    

if __name__ == "__main__":
    # example usage
    ### Spatial month EMOS
    parser = argparse.ArgumentParser(description="Run spatial month experiment.")
    parser.add_argument('-lead', '--lead_idx', type=int, required=True, help="Lead index to use in the experiment")
    args = parser.parse_args()
    lead_idx = args.lead_idx

    base_dir = f"training_results/spatial_month_ensemble_detrend/lead{lead_idx}"
    for month in range(1,13):
        print(f"Month {month}")
        for variable in ["2m_temperature", "10m_wind_speed"]:
            print(f"Variable {variable}")
            if variable == "2m_temperature":
                nb_epochs = 10
            elif variable == "10m_wind_speed":
                nb_epochs = 15
            main_spatial_ens(
                variable,
                lead_time=lead_idx,
                valid_years=[1996,2017],
                valid_months=[month, month],
                batch_size=128,
                lr=0.01,
                nb_epoch=nb_epochs,
                name_experiment=f"spatial_month{month}_{variable}_lead={lead_idx}",
                base_dir=base_dir)

    # DRUnet both
    base_dir = f"training_results/DRUnet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    main_Unet_ens(
        valid_years=[1996,2017],
        batch_size=128, # to compare with EMOS
        lr=0.01,
        nb_epoch=10,
        name_experiment=f"DRUnet",
        device=device,
        base_dir=base_dir)
    
    # DRUnet+prior single 
    parser = argparse.ArgumentParser(description="Run spatial month experiment.")
    parser.add_argument('-var', '--var', type=str, required=True, help="Lead index to use in the experiment")
    args = parser.parse_args()
    variable = args.var
    # training
    base_dir = f"training_results/DRUnet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    main_Unet_ensPrior_var(
        valid_years=[1996,2017],
        batch_size=128, # to compare with EMOS
        lr=0.01,
        nb_epoch=25,
        name_experiment=f"DRUnet_prior_{variable}",
        device=device,
        base_dir=base_dir,
        variable=variable)
