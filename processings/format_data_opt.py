import numpy as np
import xarray as xr
import json
import pandas as pd
from tqdm import tqdm

def get_year(date: np.datetime64):
    return pd.Timestamp(date).year

def format_forecast_features(xr_dataset):
    input_data = []
    for level in xr_dataset.level.values:
        var_level = xr_dataset.sel(level=level).to_array().values.flatten()
        var_level = np.nan_to_num(var_level)  # replace nan with 0
        input_data.append(var_level)
    return np.concatenate(input_data)

def process_location_leadtime(forecast, observation, lat, lon, ltime_index, valid_years, folder):
    # Select data for specific lat/lon and lead time
    forecast_data = forecast.sel(latitude=lat, longitude=lon).isel(prediction_timedelta=ltime_index)
    ground_truth = observation.sel(latitude=lat, longitude=lon)  # time extracted after
    l_time = forecast.prediction_timedelta.values[ltime_index].astype("timedelta64[h]").astype(int)

    # Open file for writing dynamically
    file_path = f'{folder}/data_OPT_lat={lat}_lon={lon}_lead={l_time}h.json'
    with open(file_path, 'w') as f:
        f.write("[\n")  # Start of JSON array

        # Loop over forecast times: 1 over 2 is selected
        first_entry = True
        for forecast_time in forecast_data.forecast_time.values[::2]:
            #print("forecast_loop",forecast_time)
            train_data = forecast_data.sel(forecast_time=forecast_time)

            # Loop over hindcast years: keep dates only within train years
            valid_times = train_data.valid_time.compute().values
            for i, date in enumerate(valid_times):
                #print("hindcast_loop",date)
                if get_year(date) not in valid_years:
                    continue

                dict_date = {
                    "date": str(date),
                    "forecast_time": str(forecast_time),
                    "lead_time": int(l_time),
                    "lat": float(lat),
                    "lon": float(lon)
                }
                forecast_sel = train_data.isel(hindcast_year=i)
                features = format_forecast_features(forecast_sel)
                dict_date["input"] = features.tolist()  # Ensure features is a list

                # Add observation and ground truth for all variables 
                for obs_var in obs_variables:
                    #print("obs_var":obs_var)
                    fore_var = obs2forecast_var[obs_var]
                    fore_val = forecast_sel.sel(level=1000)[fore_var].values.item()
                    dict_date[f"mu_{obs_var}"] = fore_val
                    dict_date[f"sigma_{obs_var}"] = 0
                    dict_date[f"truth_{obs_var}"] = ground_truth.sel(time=date)[obs_var].values.item()

                # Write dict_date to file
                if not first_entry:
                    f.write(",\n")
                json.dump(dict_date, f)
                first_entry = False
                

        f.write("\n]")  # End of JSON array

if __name__ == "__main__":
    # paths
    train_folder = '/home/majanvie/scratch/data/train'
    test_folder = '/home/majanvie/scratch/data/test'

    param_folder = '../parameters/'
    with open(param_folder + "data.json") as fp:
        params = json.load(fp)

    forecast_train = xr.open_zarr(params["forecast_train_path"])
    forecast_test = xr.open_zarr(params["forecast_test_path"])
    obs = xr.open_zarr(params["obs_path"])
    obs_variables = params["obs_variables"]
    obs2forecast_var = params["obs2forecast_var"]
    lead_time_indices = params["lead_time_indices"]
    train_years = np.arange(params["train_years"][0], params["train_years"][1] + 1)
    test_years = np.arange(params["test_years"][0], params["test_years"][1] + 1)

    # Extract observation variables
    obs = obs[obs_variables]

    ### TRAIN DATA
    lats = forecast_train.latitude.values
    lons = forecast_train.longitude.values
    for lat in tqdm(lats, desc='Latitudes'):
        for lon in tqdm(lons, desc='Longitudes', leave=False):
            for lead_time_index in lead_time_indices:
                process_location_leadtime(forecast_train, obs, lat, lon, lead_time_index, train_years, train_folder)
            break
        break
