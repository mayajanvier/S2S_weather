import netCDF4
import numpy as np
import xarray as xr
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import norm
import os

def get_year(date: np.datetime64):
    return pd.Timestamp(date).year

def get_month(date: np.datetime64):
    return pd.Timestamp(date).month

### PandasDataset methods
def compute_wind_speed(dataframe):
    dataframe["mu_10m_wind_speed"] = np.sqrt(
        dataframe["mu_10m_u_component_of_wind"]**2 + dataframe["mu_10m_v_component_of_wind"]**2
    )
    dataframe["sigma_10m_wind_speed"] = np.sqrt(
        dataframe["sigma_10m_u_component_of_wind"]**2 + dataframe["sigma_10m_v_component_of_wind"]**2
    )
    dataframe["truth_10m_wind_speed"] = np.sqrt(
        dataframe["truth_10m_u_component_of_wind"]**2 + dataframe["truth_10m_v_component_of_wind"]**2
    )
    return dataframe


### WeatherDataset methods
def compute_wind_speedxr(dataset, name):
    """Computes wind speed in place"""
    if name =="data":
        u_name = 'u_component_of_wind'
        v_name = 'v_component_of_wind'
    elif name == "obs":
        u_name = "10m_u_component_of_wind"
        v_name = "10m_v_component_of_wind"
    dataset['10m_wind_speed'] = (dataset[u_name]**2 + dataset[v_name]**2)**0.5
    dataset['10m_wind_speed'].name = '10_wind_speed'


def adjust_date(forecast_time, hindcast_year):
    """ Manage invalid leap years delta"""
    year_delta = relativedelta(years=hindcast_year)
    new_date = forecast_time + year_delta
    return new_date
    
    
def fit_norm(data):
    # Fit a normal distribution to the data
    mu, sigma = norm.fit(data)
    return mu, sigma

def fit_norm_along_axis(arr, axis):
    # Apply the fit_norm function along the specified axis
    return np.apply_along_axis(lambda x: fit_norm(x), axis, arr)

def save_format_ensemble_data_EMOS(surface_dir, levels_dir, out_folder):
    mean_out_folder = f"{out_folder}/mean"
    std_out_folder = f"{out_folder}/std"

    for file in os.listdir(surface_dir):
        if file.endswith(".nc"):
            dataset_surface = xr.open_dataset(f"{surface_dir}/{file}")

            # mean 
            mean_surface = dataset_surface.mean(dim="number",skipna=True) # do not take into account nan values
            mean_levels = xr.open_dataset(f"{levels_dir}/mean/{file}")

            for level in mean_levels.level.values:
                for var in mean_levels.data_vars:
                    name = f"{var}_{level}"
                    mean_surface[name] = mean_levels[var].sel(level=level)
            
            mean_surface = mean_surface.drop_vars([f"specific_humidity_{level}" for level in [10,50,100]]) # all NaN
            mean_surface = mean_surface.drop_vars([f"10m_wind_speed_{level}" for level in mean_levels.level.values[:-1]])
            
            # std 
            std_surface = dataset_surface.std(dim="number",skipna=True) # do not take into account nan values
            std_levels = xr.open_dataset(f"{levels_dir}/std/{file}")

            for var in std_levels.data_vars:
                name = f"{var}"
                std_surface[name] = std_levels[var]

            mean_surface.to_netcdf(f"{mean_out_folder}/{file}")
            std_surface.to_netcdf(f"{std_out_folder}/{file}")


def format_ensemble_data_EMOS(surface_file, levels_mean_file, levels_std_file):
    dataset_surface = xr.open_dataset(surface_file)
    # mean 
    mean_surface = dataset_surface.mean(dim="number",skipna=True) # do not take into account nan values
    mean_levels = xr.open_dataset(levels_mean_file)

    for level in mean_levels.level.values:
        for var in mean_levels.data_vars:
            name = f"{var}_{level}"
            mean_surface[name] = mean_levels[var].sel(level=level)
    
    # drop var
    mean_surface = mean_surface.drop_vars([f"specific_humidity_{level}" for level in [10,50,100]]) # all NaN
    mean_surface = mean_surface.drop_vars([f"10m_wind_speed_{level}" for level in mean_levels.level.values[:-1]])
    # rename 10m_wind_speed_1000 to 10m_wind_speed
    mean_surface = mean_surface.rename_vars({"10m_wind_speed_1000": "10m_wind_speed"})
    # nan to zero
    mean_surface = mean_surface.fillna(0)

    
    # std 
    std_surface = dataset_surface.std(dim="number",skipna=True) # do not take into account nan values
    std_levels = xr.open_dataset(levels_std_file)

    for var in std_levels.data_vars:
        name = f"{var}"
        std_surface[name] = std_levels[var]

    # nan to zero
    std_surface = std_surface.fillna(0)
    return mean_surface, std_surface


if __name__== "__main__":
    raw_data_folder = "/home/majanvie/scratch/data/raw/"
    # train
    surface_dir = f"{raw_data_folder}/train_ensemble_surface"
    levels_dir = f"{raw_data_folder}/train"
    out_folder = "/home/majanvie/scratch/data/train/EMOS"
    format_ensemble_data_EMOS(surface_dir, levels_dir, out_folder)

    # test 
    surface_dir = f"{raw_data_folder}/test_ensemble_surface"
    levels_dir = f"{raw_data_folder}/test"
    out_folder = "/home/majanvie/scratch/data/test/EMOS"
    format_ensemble_data_EMOS(surface_dir, levels_dir, out_folder)
