import argparse
import pandas as pd
from model import MOS, SpatialMOS
from processings.dataset import PandasDataset, WeatherDataset
from train import train, create_training_folder
from torch.utils.data import DataLoader
from metrics import crps_normal
import torch
import json
from sklearn.model_selection import train_test_split
import xarray as xr


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
    
    ### Spatial month
    parser = argparse.ArgumentParser(description="Run spatial month experiment.")
    parser.add_argument('-lead', '--lead_idx', type=int, required=True, help="Lead index to use in the experiment")
    args = parser.parse_args()
    lead_idx = args.lead_idx

    base_dir = f"training_results/spatial_month/lead{lead_idx}"
    for month in range(1,13):
        print(f"Month {month}")
        for variable in ["2m_temperature", "10m_wind_speed"]:
            print(f"Variable {variable}")
            if variable == "2m_temperature":
                nb_epochs = 10
            elif variable == "10m_wind_speed":
                nb_epochs = 15
            main_spatial(
                variable,
                lead_time=lead_idx,
                valid_years=[1996,2017],
                valid_months=[month, month],
                batch_size=128,
                lr=0.01,
                nb_epoch=nb_epochs,
                name_experiment=f"spatial_month{month}_{variable}_lead={lead_idx}",
                base_dir=base_dir)


    ### Spatial month rolling window 