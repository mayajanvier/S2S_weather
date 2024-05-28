import numpy as np
import xarray as xr
import json 



#def format_obs_loc_date(obs_data, loc, date):
#    lat, lon = loc


    




if __name__ == "__main__":
    # paths
    train_folder = 'scratch/data/train'
    test_folder = 'scratch/data/test'

    param_folder = 'parameters/'
    with open(param_folder+"data.json") as fp:
        params = json.load(fp)

    forecast_train = xr.open_zarr(params["forecast_train_path"])
    forecast_test = xr.open_zarr(params["forecast_test_path"])
    obs = xr.open_zarr(params["obs_path"])
    obs_variables = params["obs_variables"]

    # build training data 
    # loop over locations and dates
    #for lat in forecast_train.lat.values:
    #    for lon in forecast_train.lon.values:
            # extract lat/lon data
    #        forecast
    #        for f_time in forecast_train.forecast_time.values:
    #            for 