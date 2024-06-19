import netCDF4
import numpy as np
import xarray as xr
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

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