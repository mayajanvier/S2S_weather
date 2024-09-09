import argparse
import pandas as pd
from model import MOS, SpatialMOS, SpatialEMOS, DRUnet, SpatialEMOSprior
from processings.dataset import PandasDataset, WeatherDataset, WeatherEnsembleDataset, WeatherYearEnsembleDataset, WeatherEnsembleDatasetMM
from train import train, create_training_folder, train_sched, trainUNet
from torch.utils.data import DataLoader
from metrics import crps_normal
import torch
import json
from sklearn.model_selection import train_test_split
import xarray as xr
import numpy as np




def main():
    # get training parameters
    param_folder = 'parameters/' 
    with open(param_folder+"train.json") as fp:
        params = json.load(fp)
    
    name_experiment = params["name_experiment"]
    target_column = params["target_variable"]
    batch_size = params["batch_size"]
    nb_epoch = params["nb_epoch"]
    lr = params["lr"]
    out_dim = params["out_dim"]
    save_every = params["save_every"]

    # create training folder
    folder = create_training_folder(name_experiment)

    # load data
    data_folder = "../scratch/data/train/"
    # TODO: change this to the correct path
    data_path = data_folder+f'PPE_OPT_lat=-90.0_lon=0.0_lead=24h.json'
    train_data  = pd.read_json(data_path)
    train_data = compute_wind_speed(train_data)
    feature_dim = len(train_data["input"][0])

    # separate train and validation randomly
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True)
    train_indices, val_indices = train_data.index.tolist(), val_data.index.tolist()

    # build dataloaders
    train_data   = PandasDataset(train_data, target_column=target_column)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = PandasDataset(val_data, target_column=target_column)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # model setup and training
    criterion = crps_normal
    model = MOS(feature_dim, out_dim)

    # save parametrization of training
    params["data_path"] = data_path
    params["feature_dim"] = feature_dim
    params["metric"] = "CRPS_Normal" # TODO les noms faire liste lien comme a dreem, pareil pour model 
    params["train_indices"] = train_indices
    params["val_indices"] = val_indices
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)

    # training
    train(
        train_loader,
        val_loader,
        model=model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        name_experiment=name_experiment,
        target_column=target_column,
        batch_size=batch_size,
        save_every=save_every
        )


def main_spatial(variable, lead_time, valid_years, valid_months, batch_size, lr, nb_epoch, name_experiment, base_dir):
    """ Train models for this variable, lead_time, 
    on valid months data, for all latitude/longitude """

    data_folder = "/home/majanvie/scratch/data/raw"
    train_folder = f"{data_folder}/train"
    obs_folder = f"{data_folder}/obs"

    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask, dtype=torch.float)
    folder = create_training_folder(name_experiment, base_dir)
    
    train_dataset = WeatherDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = WeatherDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # model setup and training
    model = SpatialMOS(47, 121, 240, 3)
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
    
def main_Unet_ens(valid_years, batch_size, lr, nb_epoch, name_experiment, device, base_dir):
    """ Train Unet for wind and temperature, all lead_times, 
    on valid months data, spatially"""
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"

    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask[1:,:], dtype=torch.float) #120x240
    land_sea_mask = land_sea_mask.to(device) 
    folder = create_training_folder(name_experiment, base_dir)
    
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

    # model setup and training
    model = DRUnet(70,4).to(device)
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

def main_spatial_ens2(variable, lead_time, valid_years, valid_months, batch_size, lr, nb_epoch, name_experiment, base_dir):
    """ Train models for this variable, lead_time, 
    on valid months data, for all latitude/longitude """
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"

    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask, dtype=torch.float)
    folder = create_training_folder(name_experiment, base_dir)
    
    train_dataset = WeatherEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = WeatherEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # model setup and training
    model = SpatialEMOS(66, 121, 240, 4)
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
    train_sched(
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

def main_spatial_ens3(variable, lead_time, valid_years, valid_months, batch_size, lr, nb_epoch, name_experiment, base_dir):
    """ Train models for this variable, lead_time, 
    on valid months data, for all latitude/longitude """
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"

    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask, dtype=torch.float)
    folder = create_training_folder(name_experiment, base_dir)
    
    train_dataset = WeatherEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = WeatherEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # model setup and training
    model = SpatialEMOSprior(66, 121, 240, 4)
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

def main_spatial_ens4(variable, lead_time, valid_years, valid_months, batch_size, lr, nb_epoch, name_experiment, base_dir):
    """ Train models for this variable, lead_time, 
    on valid months data, for all latitude/longitude """
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"

    land_sea_mask = xr.open_dataset(f"{obs_folder}/land_sea_mask.nc").land_sea_mask.values.T # (lat, lon)
    land_sea_mask = torch.tensor(land_sea_mask, dtype=torch.float)
    folder = create_training_folder(name_experiment, base_dir)
    
    train_dataset = WeatherEnsembleDataset2(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = WeatherEnsembleDataset2(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=lead_time,
        valid_years=valid_years,
        valid_months=valid_months,
        subset="val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # model setup and training
    model = SpatialEMOS(66, 121, 240, 4)
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

if __name__ == "__main__":
    ### Spatial year 
    # base_dir = "training_results/spatial_year"
    # for variable in ["2m_temperature", "10m_wind_speed"]:
    #     main_spatial(
    #         variable,
    #         lead_time=28,
    #         valid_years=[1996,2017],
    #         valid_months=[1,12],
    #         batch_size=128,
    #         lr=0.01,
    #         nb_epoch=15,
    #         name_experiment=f"spatial_year_{variable}_lead={28}",
    #         base_dir=base_dir)
    
    ### Spatial month MOS
    # parser = argparse.ArgumentParser(description="Run spatial month experiment.")
    # parser.add_argument('-lead', '--lead_idx', type=int, required=True, help="Lead index to use in the experiment")
    # args = parser.parse_args()
    # lead_idx = args.lead_idx

    # base_dir = f"training_results/spatial_month/lead{lead_idx}"
    # for month in range(1,13):
    #     print(f"Month {month}")
    #     for variable in ["2m_temperature", "10m_wind_speed"]:
    #         print(f"Variable {variable}")
    #         if variable == "2m_temperature":
    #             nb_epochs = 10
    #         elif variable == "10m_wind_speed":
    #             nb_epochs = 15
    #         main_spatial(
    #             variable,
    #             lead_time=lead_idx,
    #             valid_years=[1996,2017],
    #             valid_months=[month, month],
    #             batch_size=128,
    #             lr=0.01,
    #             nb_epoch=nb_epochs,
    #             name_experiment=f"spatial_month{month}_{variable}_lead={lead_idx}",
    #             base_dir=base_dir)


    ### Spatial month EMOS
    # parser = argparse.ArgumentParser(description="Run spatial month experiment.")
    # parser.add_argument('-lead', '--lead_idx', type=int, required=True, help="Lead index to use in the experiment")
    # args = parser.parse_args()
    # lead_idx = args.lead_idx

    # base_dir = f"training_results/spatial_month_ensemble/lead{lead_idx}"
    # for month in range(1,2):
    #     print(f"Month {month}")
    #     for variable in ["2m_temperature","10m_wind_speed"]:
    #         print(f"Variable {variable}")
    #         if variable == "2m_temperature":
    #             nb_epochs = 20
    #         elif variable == "10m_wind_speed":
    #             nb_epochs = 15
    #         main_spatial_ens(
    #             variable,
    #             lead_time=lead_idx,
    #             valid_years=[1996,2017],
    #             valid_months=[month, month],
    #             batch_size=128,
    #             lr=0.01,
    #             nb_epoch=nb_epochs,
    #             name_experiment=f"spatial_month{month}_{variable}_lead={lead_idx}_log_nostd_MM_lead",
    #             base_dir=base_dir)
    
    # DRUnet
    base_dir = f"training_results/DRUnet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    main_Unet_ens(
        valid_years=[1996,2017],
        batch_size=64,
        lr=0.0001,
        nb_epoch=50,
        name_experiment=f"DRUnet",
        device=device,
        base_dir=base_dir)
