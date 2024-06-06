import netCDF4
import numpy as np
import xarray as xr
import pandas as pd

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
def get_lead_shape(forecast, observation, ltime_index, valid_years, valid_months):
    # Select lead time
    forecast_data = forecast.isel(prediction_timedelta=ltime_index)
    # Select valid years and month
    t = forecast_data.sel(time=(
        (forecast_data.time.dt.month >= valid_months[0]) &
        (forecast_data.time.dt.month <= valid_months[-1]) &
        (forecast_data.time.dt.year >= valid_years[0]) &
        (forecast_data.time.dt.year <= valid_years[-1])))
    return forecast_data, t.time.shape[0] 



