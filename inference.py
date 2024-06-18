import pandas as pd 
import numpy as np
import xarray as xr
from metrics import compute_crps_normal
import torch
import json
from processings.dataset import PandasDataset, WeatherDataset, compute_wind_speed
from torch.utils.data import DataLoader
from model import MOS, SpatialMOS

def MOS_inference(model, batch):
    param_folder = "parameters/"
    # Load parameters
    with open(param_folder + "test.json") as fp: # test 
        test_params = json.load(fp)

    model_folder = test_params["model_folder"]
    with open(model_folder + "params.json") as fp: # train
        train_params = json.load(fp)
    out_dim = train_params["out_dim"]

    # Load and format test data 
    test = pd.read_json(test_params["test_file_path"])
    test = compute_wind_speed(test)
    var = train_params["target_variable"] # one model per variable 
    test_data = PandasDataset(test, var)
    test_data = DataLoader(test_data, batch_size=1, shuffle=True)
        
    # Load model weights
    feature_dim = len(test["input"][0])
    epoch = test_params["model_epoch"]
    model = MOS(feature_dim,out_dim)
    model.load_state_dict(torch.load(model_folder + f"model_{epoch}.pth"))
    model.eval()

    # Compute performance metrics
    results_file_path = model_folder + f"results_model_{epoch}.json"
    with open(results_file_path, 'w') as f:
        f.write("[\n")  # Start of JSON array
        first_entry = True

        crps_list = []
        for batch in test_data:
            # TODO: add metrics 
            crps_var = compute_crps_normal(model, batch).detach().numpy()[0]
            dict_date = {
                "forecast_time": batch["forecast_time"],
                "crps": float(crps_var)
                }
            crps_list.append(crps_var)
            if not first_entry:
                f.write(",\n")
            json.dump(dict_date, f)
            first_entry = False

        # Compute mean CRPS
        dict_mean = {
            "forecast_time": "mean",
            "crps": float(sum(crps_list)/len(crps_list))
            }
        if not first_entry:
            f.write(",\n")
        json.dump(dict_mean, f)

        # End of JSON array
        f.write("\n]")  


def SpatialMOS_inference(lead_time, valid_years, epoch=14):
    data_folder = "/home/majanvie/scratch/data/raw"
    test_folder = f"{data_folder}/test"
    obs_folder = f"{data_folder}/obs"

    base_dir = "training_results/spatial_month"
    train_index = 2
    full_results = []   
    for month in range(1,13):
        print(f"Month {month}")
        for variable in ["2m_temperature", "10m_wind_speed"]:
            print(f"Variable {variable}")
            model_folder = f"{base_dir}/training_{train_index}_spatial_month{month}_{variable}_lead={lead_time}"

            test_dataset = WeatherDataset(
                data_path=test_folder,
                obs_path=obs_folder,
                target_variable=variable,
                lead_time_idx=lead_time,
                valid_years=valid_years,
                valid_months=[month,month],
                subset="test")

            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

            # load model weights
            model = SpatialMOS(47, 121, 240, 3)
            model.load_state_dict(torch.load(
                f"{model_folder}/model_{epoch}.pth"
                ))
            model.eval()

            # Compute performance metrics
            results = []
            for batch in test_loader:
                mu = batch['mu']
                sigma = batch['sigma']
                X = batch['input']
                y = batch['truth']
                out_distrib = model(mu, sigma, X, y)
                
                crps_var = compute_crps_normal(model, batch).detach().numpy()  # shape (1, lat, lon)
                crps_climato = np.random.rand(*crps_var.shape)  # shape (1, lat, lon)

                lead_time = batch['lead_time'].item()  
                forecast_time = batch['forecast_time'][0]
                # Create a multi-index for the time dimension
                time_index = pd.MultiIndex.from_tuples([(lead_time, forecast_time)], names=["lead_time", "forecast_time"])
                
                if train_index % 2 == 0:
                    ds = xr.Dataset(
                        data_vars=dict(
                            crps_temperature=(["time", "latitude", "longitude"], crps_var),
                            crps_climato=(["time", "latitude", "longitude"], crps_climato),
                        ),
                        coords=dict(
                            longitude=("longitude", test_loader.dataset.longitude), # 1D array
                            latitude=("latitude", test_loader.dataset.latitude), # 1D array
                            time=("time", time_index), # 2D array
                        )
                    )
                else:
                    ds = xr.Dataset(
                        data_vars=dict(
                            crps_wind_speed=(["time", "latitude", "longitude"], crps_var),
                            crps_climato=(["time", "latitude", "longitude"], crps_climato),
                        ),
                        coords=dict(
                            longitude=("longitude", test_loader.dataset.longitude), # 1D array
                            latitude=("latitude", test_loader.dataset.latitude), # 1D array
                            time=("time", time_index), # 2D array
                        ))

                # Reset the index to convert MultiIndex into separate variables
                ds = ds.reset_index('time')
                results.append(ds)
                #full_results.append(ds)
            train_index += 1
            final_ds = xr.concat(results, dim='time')

            # Write to NetCDF file
            results_file_path = f"{model_folder}/crps_{epoch}.nc"
            final_ds.to_netcdf(results_file_path)

    # Concatenate all results
    #full_final_ds = xr.concat(full_results, dim='time')
    # Write to NetCDF file
    #results_file_path = f"{base_dir}/crps_{epoch}.nc"
    #full_final_ds.to_netcdf(results_file_path)


if __name__ == "__main__":
    SpatialMOS_inference(28, [2018,2022],epoch=14)



