import numpy as np
import xarray as xr
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from time import time
from format_data import compute_wind_speedxr
import os

def save_train2netcdf(xarray_data, hindcast_year, time_slice, folder, selected_leads = [0,7,14,21,28,35,39]):
    path = f"{folder}/data_hindcast={hindcast_year}_forecast={time_slice[0]}_{time_slice[1]}.nc"
    if not os.path.exists(path):
        subset = xarray_data.isel(
            hindcast_year=hindcast_year,
            forecast_time=slice(time_slice[0],time_slice[1]),
            prediction_timedelta=selected_leads
            )
        subset.to_netcdf(path)

def save_test2netcdf(xarray_data, time_slice, folder, selected_leads = [0,7,14,21,28,35,39]):
    path = f"{folder}/data_forecast={time_slice[0]}_{time_slice[1]}.nc"
    if not os.path.exists(path):
        subset = xarray_data.isel(
            time=slice(time_slice[0],time_slice[1]),
            prediction_timedelta=selected_leads
            )
        subset.to_netcdf(path)

def download_train_test(train_data, test_data):
    year_range = train_data.hindcast_year.shape[0]
    forecast_times_range = train_data.forecast_time.shape[0] // 20 # by 20 years slice
    for forecast_idx in range(forecast_times_range):
        forecast_slice = (forecast_idx*20, (forecast_idx+1)*20)

        # test data
        save_test2netcdf(test_data, forecast_slice, test_folder)

        # hindcast year loop for train and obs data
        for year_idx in range(year_range):
            save_train2netcdf(train_data, year_idx, forecast_slice, train_folder)

    # Last slice
    last_slice = (forecast_times_range*20, train_data.forecast_time.shape[0])
    save_test2netcdf(test_data, last_slice, test_folder)
    for year_idx in range(year_range):
        save_train2netcdf(train_data, year_idx, last_slice, train_folder)


### compute on the fly mean and std before downloading only it, no full members 
def save_train2netcdf_mean_std(xarray_data, hindcast_year, time_slice, folder, selected_leads = [0,7,14,21,28,35,39]):
    path_mean = f"{folder}/mean/data_hindcast={hindcast_year}_forecast={time_slice[0]}_{time_slice[1]}.nc"
    path_std = f"{folder}/std/data_hindcast={hindcast_year}_forecast={time_slice[0]}_{time_slice[1]}.nc"
    if not os.path.exists(path_mean) and not os.path.exists(path_std):
        subset = xarray_data.isel(
            hindcast_year=hindcast_year,
            forecast_time=slice(time_slice[0],time_slice[1]),
            prediction_timedelta=selected_leads
            )
        compute_wind_speedxr(subset, "data")
        subset_mean = subset.mean(dim='number') # we need all means 
        # only get wind speed std
        subset_std = subset["10m_wind_speed"].sel(level=1000).std(dim='number') # we need target variable std only 
        subset_mean.to_netcdf(path_mean)
        subset_std.to_netcdf(path_std) 

def save_test2netcdf_mean_std(xarray_data, time_slice, folder, selected_leads = [0,7,14,21,28,35,39]):
    path_mean = f"{folder}/mean/data_forecast={time_slice[0]}_{time_slice[1]}.nc"
    path_std = f"{folder}/std/data_forecast={time_slice[0]}_{time_slice[1]}.nc"
    if not os.path.exists(path_mean) and not os.path.exists(path_std):
        subset = xarray_data.isel(
            time=slice(time_slice[0],time_slice[1]),
            prediction_timedelta=selected_leads
            )
        compute_wind_speedxr(subset, "data")
        subset_mean = subset.mean(dim='number') # we need all means
        # only get wind speed std
        subset_std = subset["10m_wind_speed"].sel(level=1000).std(dim='number') # we need target variable std only
        subset_mean.to_netcdf(path_mean)
        subset_std.to_netcdf(path_std) 

def download_train_test_mean_std(train_data, test_data):
    year_range = train_data.hindcast_year.shape[0]
    forecast_times_range = train_data.forecast_time.shape[0] // 20 # by 20 years slice
    for forecast_idx in range(forecast_times_range):
        forecast_slice = (forecast_idx*20, (forecast_idx+1)*20)

        # test data
        save_test2netcdf_mean_std(test_data, forecast_slice, test_folder)

        # hindcast year loop for train and obs data
        for year_idx in range(year_range):
            save_train2netcdf_mean_std(train_data, year_idx, forecast_slice, train_folder)

    # Last slice
    last_slice = (forecast_times_range*20, train_data.forecast_time.shape[0])
    save_test2netcdf_mean_std(test_data, last_slice, test_folder)
    for year_idx in range(year_range):
        save_train2netcdf_mean_std(train_data, year_idx, last_slice, train_folder)

if __name__ == "__main__":
    # destination paths
    train_folder = '/home/majanvie/scratch/data/raw/train'
    test_folder = '/home/majanvie/scratch/data/raw/test'
    obs_folder = '/home/majanvie/scratch/data/raw/obs'

    # load parameters
    param_folder = '/home/majanvie/S2S_weather/parameters/'
    with open(param_folder + "weatherbench.json") as fp:
        params = json.load(fp)

    forecast_train = xr.open_zarr(params["forecast_train_path"])
    forecast_test = xr.open_zarr(params["forecast_test_path"])
    obs = xr.open_zarr(params["obs_path"])
    obs_variables = params["obs_variables"]
    

    ### OBS DATA
    print("Extracting observation data:")
    time_start = time()
    obs = obs[obs_variables]
    t_min = min(forecast_train.valid_time.compute().values.min(), forecast_test.valid_time.compute().values.min())
    t_max = max(forecast_train.valid_time.compute().values.max(), forecast_test.valid_time.compute().values.max())
    obs = obs.sel(time=slice(t_min, t_max))
    for var in obs_variables:
        if not os.path.exists(f"{obs_folder}/{var}.nc"):
            obs_var = obs[var]
            obs_var.to_netcdf(f"{obs_folder}/{var}.nc")
    
    print(f"Extracted observation data in {time()-time_start:.2f} seconds") # 73s

    ### TRAIN AND TEST DATA: AVERAGED ENSEMBLE (no for the study, but can be useful)
    # print("Extracting train and test data:")
    # time_start = time()
    # with ProcessPoolExecutor(6) as exe:
    #     exe.submit(download_train_test(forecast_train, forecast_test))
    # print(f"Extracted train and test data in {time()-time_start:.2f} seconds")


    ### TRAIN AND TEST DATA: ENSEMBLE MEMBERS 
    # surface destination paths: change for your own
    train_folder = '/home/majanvie/scratch/data/raw/train_ensemble_surface'
    test_folder = '/home/majanvie/scratch/data/raw/test_ensemble_surface'

    forecast_train = xr.open_zarr(params["forecast_train_ens_surface"])
    forecast_test = xr.open_zarr(params["forecast_test_ens_surface"])

    print("Extracting train and test data:")
    time_start = time()
    with ProcessPoolExecutor(6) as exe:
        exe.submit(download_train_test(forecast_train, forecast_test)) # we get all members
    print(f"Extracted train and test data in {time()-time_start:.2f} seconds")


    # pressure levels destination paths: change for your own 
    train_folder = '/home/majanvie/scratch/data/raw/train'
    test_folder = '/home/majanvie/scratch/data/raw/test'

    forecast_train = xr.open_zarr(params["forecast_train_ens_levels"])
    forecast_test = xr.open_zarr(params["forecast_test_ens_levels"])

    print("Extracting train and test data:")
    time_start = time()
    with ProcessPoolExecutor(6) as exe:
        exe.submit(download_train_test_mean_std(forecast_train, forecast_test)) # directly compute and save mean and std
    print(f"Extracted train and test data in {time()-time_start:.2f} seconds")





