import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .format_data import compute_wind_speedxr, compute_wind_speed, fit_norm_along_axis, format_ensemble_data_EMOS
import xarray as xr
import pathlib
import joblib
import re
from datetime import datetime
import os
import time


class PandasDataset(Dataset):
    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe
        self.target_column = target_column
        self.mu = dataframe[f"mu_{target_column}"].values # scalar
        self.sigma = dataframe[f"sigma_{target_column}"].values # scalar
        self.truth = dataframe[f"truth_{target_column}"].values # scalar
        self.input = dataframe["input"] # numpy arrays
        self.forecast_time = dataframe["forecast_time"]

        # standardize input
        self.scaler = StandardScaler()
        self.input = self.scaler.fit_transform(np.stack(self.input))
  
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        mu = torch.tensor(self.mu[idx], dtype=torch.float)
        sigma = torch.tensor(self.sigma[idx], dtype=torch.float)
        truth = torch.tensor(self.truth[idx], dtype=torch.float)
        input_data = torch.tensor(self.input[idx], dtype=torch.float)
        forecast_time = str(self.forecast_time.iloc[idx]) #.astype(str)
        
        return {'mu': mu, 'sigma': sigma, 'input': input_data, 'truth': truth, 'forecast_time': forecast_time}

class WeatherDataset:
    def __init__(
        self,
        data_path,
        obs_path,
        target_variable,
        lead_time_idx,
        valid_years,
        valid_months,
        subset
    ):
        self.lead_time_idx = lead_time_idx
        self.obs2forecast_var = {
            "2m_temperature": "temperature", 
            "10m_u_component_of_wind": "u_component_of_wind", 
            "10m_v_component_of_wind": "v_component_of_wind",
            "10m_wind_speed": "10m_wind_speed"}
        self.target_variable = target_variable
        self.forecast_variable = self.obs2forecast_var[self.target_variable]
        self.valid_months = np.arange(valid_months[0], valid_months[1] + 1)
        self.valid_years = np.arange(valid_years[0], valid_years[1] + 1)
        self.validation_year_begin = self.valid_years[len(self.valid_years)*9//10]
        self.subset = subset

        # Open observations
        self.obs = xr.open_mfdataset(obs_path + '/*.nc', combine='by_coords')
        self.latitude = self.obs.latitude.values
        self.longitude = self.obs.longitude.values

        # Build index of counts for files.
        data_path = pathlib.Path(data_path)
        files = sorted(list(data_path.glob("*.nc"))) 

        self.data_index = []
        for f in files:
            data = xr.open_dataset(f)
            dataset = data.isel(prediction_timedelta=self.lead_time_idx) # select lead time

            # Iterate over remaining time dimension
            if self.subset == "test":
                time_size = dataset.sizes["time"]
            else:
                time_size = dataset.sizes["forecast_time"]

            for time_idx in range(time_size):
                if self.subset == "test":
                    forecast_time = pd.to_datetime(dataset.isel(time=time_idx).time.values)
                    valid_time = pd.to_datetime(dataset.isel(time=time_idx).valid_time.values)
                else:
                    forecast_time = pd.to_datetime(dataset.isel(forecast_time=time_idx).time.values)
                    valid_time = pd.to_datetime(dataset.isel(forecast_time=time_idx).valid_time.values)
            
                if valid_time.year not in self.valid_years:
                    continue
                elif valid_time.month not in self.valid_months:   
                    continue
                self.data_index.append(
                    (f, time_idx, valid_time, forecast_time)
                )
        
        # Adapt index for train or val
        if self.subset == "train":
            self.data_index = [x for x in self.data_index
                if x[2].year < self.validation_year_begin 
            ]
        elif self.subset == "val":
            # TODO year validation
            self.data_index = [x for x in self.data_index
                if x[2].year >= self.validation_year_begin 
            ]
        elif self.subset == "test":
            pass
        else:
            raise ValueError("Unrecognized subset")
        
        # build climatology by fitting gaussians on train data, date wise, 
        # in a netcdf file, if not done already 
        if self.subset == "train":
            climato_path = f"{obs_path}/climato/{self.target_variable}_{self.valid_years[0]}_{self.valid_years[-1]}_month{self.valid_months[0]}_lead{self.lead_time_idx}.nc"
            if not pathlib.Path(climato_path).exists():
                climato = {}
                for file, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
                    truth = self.obs.sel(time=valid_time)
                    # compute wind 
                    if self.target_variable == "10m_wind_speed":
                        compute_wind_speedxr(truth,"obs")

                    mu = truth[self.target_variable].values.T
                    lead_time = self.lead_time_idx
                    date = valid_time.strftime('%m-%d') # get date without year
                    if date not in climato.keys():
                        climato[date] = {"mu": [mu]}
                    else:
                        climato[date]["mu"].append(mu) 

                # fit gaussians on mu for each date
                climato_res = []
                for date in climato.keys():
                    mu = np.stack(climato[date]["mu"])
                    mu_fit, sigma_fit = fit_norm_along_axis(mu, axis=0)
                    mu_fit = mu_fit[np.newaxis, ...]
                    sigma_fit = sigma_fit[np.newaxis, ...]
                    ds = xr.Dataset(
                        data_vars=dict(
                            mu=(["date", "latitude", "longitude"], mu_fit),
                            sigma=(["date", "latitude", "longitude"], sigma_fit),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                            date=[date] # single item
                        )
                    )
                    climato_res.append(ds)
                
                # save to csv
                final_ds = xr.concat(climato_res, dim='date')
                final_ds.to_netcdf(climato_path)

        # normalize
        self.scaler = StandardScaler()
        self.fit_scaler()

    def fit_scaler(self):
        """Get all features and fit the scaler on them."""
        all_features = []
        for file, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
            dataset = xr.open_dataset(file)
            if self.subset == "test":
                dataset = dataset.rename({'time': 'forecast_time'})
            dataset = dataset.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)

            # Extract features and reshape
            data = dataset.to_array().values # (n_vars, n_levels, n_lat, n_lon)
            data = data.reshape(-1, data.shape[2], data.shape[3]) # (n_levels*n_vars, n_lat, n_lon)
            data = data[~np.isnan(data).all(axis=(1,2))] # drop NaN of specific humidity at levels 10,50,100
            all_features.append(data)

        all_features = np.concatenate(all_features, axis=0) # Concatenate along the feature dimension
        all_features = all_features.reshape(all_features.shape[0], -1) # Flatten spatial dimensions
        self.scaler.fit(all_features)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file, forecast_idx_in_file, valid_time, forecast_time = self.data_index[idx]
        # select forecast data
        dataset = xr.open_dataset(file)
        if self.subset == "test":
            # rename time to forecast_time
            dataset = dataset.rename({'time': 'forecast_time'})
        dataset = dataset.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        truth = self.obs.sel(time=valid_time)

        # get features at right format without wind speed
        data = dataset.to_array().values # (n_vars, n_levels, n_lat, n_lon)
        data = data.reshape(-1, data.shape[2], data.shape[3]) # (n_levels*n_vars, n_lat, n_lon)
        # TODO this fails when unexpected Nan, change it 
        data = data[~np.isnan(data).all(axis=(1,2))] # drop NaN of specific humidity at levels 10,50,100
        # Normalize features
        shapes = data.shape
        data = data.reshape(data.shape[0], -1) # Flatten spatial dimensions
        data = self.scaler.transform(data) # Normalize features
        data = data.reshape(shapes[0], shapes[1], shapes[2]) # Reshape back
        features = torch.tensor(data, dtype=torch.float)

        # compute wind 
        if self.target_variable == "10m_wind_speed":
            compute_wind_speedxr(dataset,"data")
            compute_wind_speedxr(truth,"obs")

        # predicted values for the target variable at 1000 hPa
        mu = torch.tensor(dataset.sel(level=1000)[self.forecast_variable].values, dtype=torch.float) # (n_lat, n_lon)
        sigma = torch.tensor(0, dtype=torch.float)
        valid_time = str(valid_time)
        forecast_time = str(forecast_time)
        # get associated ground truth
        truth = torch.tensor(truth[self.target_variable].values.T, dtype=torch.float) # (n_lat, n_lon)
        return {
            'mu': mu, 'sigma': sigma, 'input': features, 'truth': truth,
            'valid_time': valid_time, "lead_time": self.lead_time_idx, "forecast_time": forecast_time
            }   

class WeatherEnsembleDataset:
    """Data in files"""
    def __init__(
        self,
        data_path,
        obs_path,
        target_variable,
        lead_time_idx,
        valid_years,
        valid_months,
        subset
    ):
        self.lead_time_idx = round(lead_time_idx/7)
        self.target_variable = target_variable
        self.valid_months = np.arange(valid_months[0], valid_months[1] + 1)
        self.valid_years = np.arange(valid_years[0], valid_years[1] + 1)
        self.validation_year_begin = self.valid_years[len(self.valid_years)*9//10]
        self.subset = subset

        # Open observations
        self.obs = xr.open_mfdataset(obs_path + '/*.nc', combine='by_coords')
        self.latitude = self.obs.latitude.values
        self.longitude = self.obs.longitude.values

        # Build index of counts for files
        self.data_index = []
        mean_path = f"{data_path}/mean"
        std_path = f"{data_path}/std"
        for f in os.listdir(mean_path):
            f_mean = f"{mean_path}/{f}"
            f_std = f"{std_path}/{f}"
            data_mean = xr.open_dataset(f_mean).isel(prediction_timedelta=self.lead_time_idx) # select lead time

            # Iterate over remaining time dimension
            if self.subset == "test":
                time_size = data_mean.sizes["time"]
            else:
                time_size = data_mean.sizes["forecast_time"]

            for time_idx in range(time_size):
                if self.subset == "test":
                    forecast_time = pd.to_datetime(data_mean.isel(time=time_idx).time.values)
                    valid_time = pd.to_datetime(data_mean.isel(time=time_idx).valid_time.values)
                else:
                    forecast_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).time.values)
                    valid_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).valid_time.values)
            
                if valid_time.year not in self.valid_years:
                    continue
                elif valid_time.month not in self.valid_months:   
                    continue
                self.data_index.append(
                    (f_mean, f_std, time_idx, valid_time, forecast_time)
                )
        
        # Adapt index for train or val
        if self.subset == "train":
            self.data_index = [x for x in self.data_index
                if x[3].year < self.validation_year_begin 
            ]
        elif self.subset == "val":
            self.data_index = [x for x in self.data_index
                if x[3].year >= self.validation_year_begin 
            ]
        elif self.subset == "test":
            pass
        else:
            raise ValueError("Unrecognized subset")
        
        # build climatology by fitting gaussians on train data, date wise, 
        # in a netcdf file, if not done already 
        if self.subset == "train":
            climato_path = f"{obs_path}/climato/{self.target_variable}_{self.valid_years[0]}_{self.valid_years[-1]}_month{self.valid_months[0]}_lead{self.lead_time_idx}.nc"
            if not pathlib.Path(climato_path).exists():
                climato = {}
                for _,_, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
                    truth = self.obs.sel(time=valid_time)
                    # compute wind 
                    if self.target_variable == "10m_wind_speed":
                        compute_wind_speedxr(truth,"obs")

                    mu = truth[self.target_variable].values.T
                    lead_time = self.lead_time_idx
                    date = valid_time.strftime('%m-%d') # get date without year
                    if date not in climato.keys():
                        climato[date] = {"mu": [mu]}
                    else:
                        climato[date]["mu"].append(mu) 

                # fit gaussians on mu for each date
                climato_res = []
                for date in climato.keys():
                    mu = np.stack(climato[date]["mu"])
                    mu_fit, sigma_fit = fit_norm_along_axis(mu, axis=0)
                    mu_fit = mu_fit[np.newaxis, ...]
                    sigma_fit = sigma_fit[np.newaxis, ...]
                    ds = xr.Dataset(
                        data_vars=dict(
                            mu=(["date", "latitude", "longitude"], mu_fit),
                            sigma=(["date", "latitude", "longitude"], sigma_fit),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                            date=[date] # single item
                        )
                    )
                    climato_res.append(ds)
                
                # save to csv
                final_ds = xr.concat(climato_res, dim='date')
                final_ds.to_netcdf(climato_path)

        # normalize
        self.scaler = StandardScaler()
        self.fit_scaler()
        

    def fit_scaler(self):
        """Get all features and fit the scaler on them."""
        all_features = []
        for file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
            data_mean = xr.open_dataset(file_mean)

            if self.subset == "test":
                data_mean = data_mean.rename({'time': 'forecast_time'})
            data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
            data_mean = data_mean.fillna(0) # manage NaN
            data_mean = data_mean.to_array().values # (n_vars, n_lat, n_lon)
            all_features.append(data_mean)

        all_features = np.concatenate(all_features, axis=0) # Concatenate along the feature dimension
        print(all_features.shape)
        all_features = all_features.reshape(all_features.shape[0], -1) # Flatten spatial dimensions
        print(all_features.shape)
        self.scaler.fit(all_features)

    # def fit_scaler(self):
    #     """Get all features and fit the scaler on them."""
    #     all_features = []
    #     for file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
    #         data_mean = xr.open_dataset(file_mean)

    #         if self.subset == "test":
    #             data_mean = data_mean.rename({'time': 'forecast_time'})
    #         data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
    #         data_mean = data_mean.to_array().values # (n_vars, n_lat, n_lon)
    #         all_features.append(data_mean)

    #     all_features = np.stack(all_features) # (n_files, n_vars, n_lat, n_lon)
    #     # do mean without considering NaN 
    #     self.mean = np.nanmean(all_features,axis=0) # (n_vars, n_lat, n_lon)
    #     self.std = np.nanstd(all_features,axis=0)

    #     self.mean_map = np.where(np.isnan(self.mean), 0, self.mean) # replace NaN by 0
    #     self.std_map = np.where(np.isnan(self.std), 1, self.std) # replace NaN by 1
    #     self.std_map = np.where(self.std == 0, 1, self.std) # replace 0 by 1

        #print(self.mean.shape, self.std.shape)

        # count Nan values
        #print(f"Number of NaN values in mean: {np.isnan(self.mean).sum()}")
        #print(f"Number of NaN values in std: {np.isnan(self.std).sum()}")
    
    # def fit_scaler(self):
    #     """Get all features and fit the scaler on them."""
    #     files = np.unique([x[0] for x in self.data_index])
    #     sample_count = 0 # n 
    #     self.running_mean_map = None
    #     self.running_var_map = None
    #     for file_mean in files:
    #         data = xr.open_dataset(file_mean)
    #         if self.subset == "test":
    #             data = data.rename({'time': 'forecast_time'})
    #         new_nb_samples = data["2m_temperature"].isel(longitude=0, latitude=0).values.flatten().shape[0] # delta 
    #         if sample_count == 0:
    #             # compute using xarray methods
    #             self.running_mean_map = data.mean(dim=["forecast_time", "prediction_timedelta"]).to_array().values
    #             self.running_var_map = data.var(dim=["forecast_time", "prediction_timedelta"]).to_array().values
    #             sample_count += new_nb_samples
    #         else:
    #             # compute new mean and variance
    #             new_data_mean = data.mean(dim=["forecast_time", "prediction_timedelta"]).to_array().values
    #             new_data_var = data.var(dim=["forecast_time", "prediction_timedelta"]).to_array().values # Nan to zero to avoid warning 
                
    #             # incremental mean and variance
    #             previous_mean_map = self.running_mean_map
    #             previous_var_map = self.running_var_map
    #             self.running_mean_map = (sample_count * self.running_mean_map + new_nb_samples * new_data_mean) / (sample_count + new_nb_samples)
    #             self.running_var_map = (sample_count * (previous_var_map + previous_mean_map**2) + 
    #                             new_nb_samples * (new_data_var + new_data_mean**2)) / (sample_count+ new_nb_samples) - self.running_mean_map**2          
    #             sample_count += new_nb_samples
            
    #     # manage NaN, negative values etc 
    #     self.mean_map = np.where(np.isnan(self.running_mean_map), 0, self.running_mean_map)
    #     self.running_var_map = np.where(np.isnan(self.running_var_map), 1, self.running_var_map)
    #     self.var_map = np.where(self.running_var_map <= 0, 1, self.running_var_map)
    #     self.std_map = np.sqrt(self.var_map)



    # def fit_scaler(self):
    #     """Get all features and fit the scaler on them."""
    #     files = np.unique([x[0] for x in self.data_index])
    #     sample_count = 0
    #     self.running_mean_map = None
    #     self.running_var_map = None
        
    #     for file_mean in files:
    #         data = xr.open_dataset(file_mean)

    #         # drop some variables
    #         nan_list = ["sea_ice_cover", "sea_surface_temperature", "snow_albedo", "snow_density", "soil_moisture_top_100cm", "soil_moisture_top_20cm", "soil_temperature_top_100cm","soil_temperature_top_20cm" ]
    #         data = data.drop_vars(nan_list)
            
    #         if self.subset == "test":
    #             data = data.rename({'time': 'forecast_time'})
            
    #         new_nb_samples = data["2m_temperature"].isel(longitude=0, latitude=0).values.flatten().shape[0]
            
    #         if sample_count == 0:
    #             # Compute initial mean and variance, replacing NaNs with appropriate default values
    #             self.running_mean_map = data.mean(dim=["forecast_time", "prediction_timedelta"], skipna=True).to_array().values
    #             self.running_mean_map = np.nan_to_num(self.running_mean_map, nan=0)

    #             self.running_var_map = data.var(dim=["forecast_time", "prediction_timedelta"], skipna=True).to_array().values
    #             self.running_var_map = np.nan_to_num(self.running_var_map, nan=0)  # Use 0 because variance cannot be negative or NaN

    #             sample_count += new_nb_samples
    #         else:
    #             # Compute new data mean and variance
    #             new_data_mean = data.mean(dim=["forecast_time", "prediction_timedelta"], skipna=True).to_array().values
    #             new_data_mean = np.nan_to_num(new_data_mean, nan=0) # manage Nan already because of the incremental mean calculation
                
    #             new_data_var = data.var(dim=["forecast_time", "prediction_timedelta"], skipna=True).to_array().values
    #             new_data_var = np.nan_to_num(new_data_var, nan=0)  # Use 0 because variance cannot be negative or NaN
                
    #             previous_mean_map = self.running_mean_map
    #             previous_var_map = self.running_var_map

    #             # Incremental mean and variance calculation
    #             self.running_mean_map = (sample_count * self.running_mean_map + new_nb_samples * new_data_mean) / (sample_count + new_nb_samples)
    #             self.running_var_map = (sample_count * (previous_var_map + previous_mean_map ** 2) +
    #                                     new_nb_samples * (new_data_var + new_data_mean ** 2)) / (sample_count + new_nb_samples) - self.running_mean_map ** 2
    #             sample_count += new_nb_samples

    #     # Ensure mean and variance maps are valid, replacing NaNs
    #     self.mean_map = np.nan_to_num(self.running_mean_map, nan=0)
    #     self.running_var_map = np.nan_to_num(self.running_var_map, nan=0)
    #     self.var_map = np.where(self.running_var_map <= 0, 1, self.running_var_map)
    #     self.std_map = np.sqrt(self.var_map)

    #     print("mean", self.mean_map.min(), self.mean_map.max())
    #     print("std", self.std_map.min(), self.std_map.max())

            

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time = self.data_index[idx]
        data_mean = xr.open_dataset(file_mean)
        data_std = xr.open_dataset(file_std)
        # drop some variables
        #nan_list = ["sea_ice_cover", "sea_surface_temperature", "snow_albedo", "snow_density", "soil_moisture_top_100cm", "soil_moisture_top_20cm", "soil_temperature_top_100cm","soil_temperature_top_20cm" ]
        #data_mean = data_mean.drop_vars(nan_list)
        #data_std = data_std.drop_vars(nan_list)
        
        # select forecast data
        if self.subset == "test":
            # rename time to forecast_time
            data_mean = data_mean.rename({'time': 'forecast_time'})
            data_std = data_std.rename({'time': 'forecast_time'})
        data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        data_std = data_std.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        truth = self.obs.sel(time=valid_time)

        # Manage Nan
        #data_mean = data_mean.fillna(0)
        #data_std = data_std.fillna(1)
        #print(self.scaler.mean_.shape)

        # Normalize features
        data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        #data = (data-self.mean_map)/self.std_map
        #data = np.where(np.isnan(data), 0, data)

        #print("datamin",data.min(), data.max())
        # print number of Nan values data
        #print(f"Number of NaN values in data: {np.isnan(data).sum()}") 
        # if Nan replace by 0
        #data = np.where(np.isnan(data), 0, data)

        #print(f"Number of NaN values in data: {np.isnan(data).sum()}") 
        shapes = data.shape
        data = data.reshape(data.shape[0], -1) # Flatten spatial dimensions
        data = self.scaler.transform(data) # Normalize features
        data = data.reshape(shapes[0], shapes[1], shapes[2]) # Reshape back
        data = np.where(np.isnan(data), 0, data) 
        # add land sea mask feature
        # land_sea_mask = truth["land_sea_mask"].values.T
        # data = np.concatenate([data, land_sea_mask[np.newaxis, ...]], axis=0)
        features = torch.tensor(data, dtype=torch.float) # 67x120x240

        # compute wind 
        if self.target_variable == "10m_wind_speed":
            compute_wind_speedxr(truth,"obs")

        # Manage Nan
        data_mean = data_mean.fillna(0)
        data_std = data_std.fillna(1)
        #print(self.scaler.mean_.shape)

        # mean std over ensemble predictions for the target variable
        mu = torch.tensor(data_mean[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        sigma = torch.tensor(data_std[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        valid_time = str(valid_time)
        forecast_time = str(forecast_time)
        # get associated ground truth 
        truth = torch.tensor(truth[self.target_variable].values.T, dtype=torch.float) # (n_lat, n_lon)
        return {
            'mu': mu, 'sigma': sigma, 'input': features, 'truth': truth,
            'valid_time': valid_time, "lead_time": self.lead_time_idx, "forecast_time": forecast_time
            }   

class WeatherEnsembleDatasetMM: # no detrend
    """Data in files"""
    def __init__(
        self,
        data_path,
        obs_path,
        target_variable,
        lead_time_idx,
        valid_years,
        valid_months,
        subset
    ):
        self.lead_time_idx = round(lead_time_idx/7)
        self.target_variable = target_variable
        self.valid_months = np.arange(valid_months[0], valid_months[1] + 1)
        self.valid_years = np.arange(valid_years[0], valid_years[1] + 1)
        self.validation_year_begin = self.valid_years[len(self.valid_years)*9//10]
        self.subset = subset

        # Open observations
        self.obs = xr.open_mfdataset(obs_path + '/*.nc', combine='by_coords')
        self.latitude = self.obs.latitude.values
        self.longitude = self.obs.longitude.values

        # Build index of counts for files
        self.data_index = []
        mean_path = f"{data_path}/mean"
        std_path = f"{data_path}/std"
        for f in os.listdir(mean_path):
            f_mean = f"{mean_path}/{f}"
            f_std = f"{std_path}/{f}"
            data_mean = xr.open_dataset(f_mean).isel(prediction_timedelta=self.lead_time_idx) # select lead time

            # Iterate over remaining time dimension
            if self.subset == "test":
                time_size = data_mean.sizes["time"]
            else:
                time_size = data_mean.sizes["forecast_time"]

            for time_idx in range(time_size):
                if self.subset == "test":
                    forecast_time = pd.to_datetime(data_mean.isel(time=time_idx).time.values)
                    valid_time = pd.to_datetime(data_mean.isel(time=time_idx).valid_time.values)
                else:
                    forecast_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).time.values)
                    valid_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).valid_time.values)
            
                if valid_time.year not in self.valid_years:
                    continue
                elif valid_time.month not in self.valid_months:   
                    continue
                self.data_index.append(
                    (f_mean, f_std, time_idx, valid_time, forecast_time)
                )
        
        # Adapt index for train or val
        if self.subset == "train":
            self.data_index = [x for x in self.data_index
                if x[3].year < self.validation_year_begin 
            ]
        elif self.subset == "val":
            self.data_index = [x for x in self.data_index
                if x[3].year >= self.validation_year_begin 
            ]
        elif self.subset == "test":
            pass
        else:
            raise ValueError("Unrecognized subset")
        
        # build climatology by fitting gaussians on train data, date wise, 
        # in a netcdf file, if not done already 
        if self.subset == "train":
            climato_path = f"{obs_path}/climato/{self.target_variable}_{self.valid_years[0]}_{self.valid_years[-1]}_month{self.valid_months[0]}_lead{self.lead_time_idx}.nc"
            if not pathlib.Path(climato_path).exists():
                climato = {}
                for _,_, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
                    truth = self.obs.sel(time=valid_time)
                    # compute wind 
                    if self.target_variable == "10m_wind_speed":
                        compute_wind_speedxr(truth,"obs")

                    mu = truth[self.target_variable].values.T
                    lead_time = self.lead_time_idx
                    date = valid_time.strftime('%m-%d') # get date without year
                    if date not in climato.keys():
                        climato[date] = {"mu": [mu]}
                    else:
                        climato[date]["mu"].append(mu) 

                # fit gaussians on mu for each date
                climato_res = []
                for date in climato.keys():
                    mu = np.stack(climato[date]["mu"])
                    mu_fit, sigma_fit = fit_norm_along_axis(mu, axis=0)
                    mu_fit = mu_fit[np.newaxis, ...]
                    sigma_fit = sigma_fit[np.newaxis, ...]
                    ds = xr.Dataset(
                        data_vars=dict(
                            mu=(["date", "latitude", "longitude"], mu_fit),
                            sigma=(["date", "latitude", "longitude"], sigma_fit),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                            date=[date] # single item
                        )
                    )
                    climato_res.append(ds)
                
                # save to csv
                final_ds = xr.concat(climato_res, dim='date')
                final_ds.to_netcdf(climato_path)

        # normalize
        self.scaler = StandardScaler()
        self.fit_scaler()

    def fit_scaler(self): # minmax 
        """Get all features and fit the scaler on them."""
        files = np.unique([x[0] for x in self.data_index])
        #sample_count = 0
        self.running_min_map = None
        self.running_max_map = None
        
        for file_mean in files:
            data = xr.open_dataset(file_mean).isel(prediction_timedelta=self.lead_time_idx) # by lead time as before 
            if self.subset == "test":
                data = data.rename({'time': 'forecast_time'})
            min_map = data.min(dim=["forecast_time"], skipna=True).to_array().values
            max_map = data.max(dim=["forecast_time"], skipna=True).to_array().values
            # manage NaN
            min_map = np.nan_to_num(min_map, nan=0)
            max_map = np.nan_to_num(max_map, nan=1)
            #print(min_map.shape)

            if self.running_min_map is None:
                self.running_min_map = min_map
                self.running_max_map = max_map
            else:
                self.running_min_map = np.minimum(self.running_min_map, min_map)
                self.running_max_map = np.maximum(self.running_max_map, max_map)
                   

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time = self.data_index[idx]
        data_mean = xr.open_dataset(file_mean)
        data_std = xr.open_dataset(file_std)
        
        # select forecast data
        if self.subset == "test":
            # rename time to forecast_time
            data_mean = data_mean.rename({'time': 'forecast_time'})
            data_std = data_std.rename({'time': 'forecast_time'})
        data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        data_std = data_std.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        truth = self.obs.sel(time=valid_time)


        # Normalize features using min max 
        data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        data = (data-self.running_min_map)/(self.running_max_map-self.running_min_map + 1e-3)
        data = np.where(np.isnan(data), 0, data)

        # add land sea mask feature
        land_sea_mask = truth["land_sea_mask"].values.T
        data = np.concatenate([data, land_sea_mask[np.newaxis, ...]], axis=0)
        # -1,1 
        data = 2*data - 1
        features = torch.tensor(data, dtype=torch.float) # 67x120x240

        # compute wind 
        if self.target_variable == "10m_wind_speed":
            compute_wind_speedxr(truth,"obs")

        # Manage Nan
        data_mean = data_mean.fillna(0)
        data_std = data_std.fillna(1)

        # mean std over ensemble predictions for the target variable
        # normalize priors 
        #if self.target_variable == "10m_wind_speed":
        #    mu = torch.tensor(data[65,:,:], dtype=torch.float) # (n_lat, n_lon)
        #elif self.target_variable == "2m_temperature":
        #    mu = torch.tensor(data[0,:,:], dtype=torch.float)
        mu = torch.tensor(data_mean[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        sigma = torch.tensor(data_std[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        valid_time = str(valid_time)
        forecast_time = str(forecast_time)
        # get associated ground truth 
        truth = torch.tensor(truth[self.target_variable].values.T, dtype=torch.float) # (n_lat, n_lon)
        return {
            'mu': mu, 'sigma': sigma, 'input': features, 'truth': truth,
            'valid_time': valid_time, "lead_time": self.lead_time_idx, "forecast_time": forecast_time
            }   

class WeatherEnsembleDatasetMMdetrend: # trend x,y 
    """Data in files"""
    def __init__(
        self,
        data_path,
        obs_path,
        target_variable,
        lead_time_idx,
        valid_years,
        valid_months,
        subset
    ):
        self.lead_time_idx = round(lead_time_idx/7)
        self.target_variable = target_variable
        self.valid_months = np.arange(valid_months[0], valid_months[1] + 1)
        self.valid_years = np.arange(valid_years[0], valid_years[1] + 1)
        self.validation_year_begin = self.valid_years[len(self.valid_years)*9//10]
        self.subset = subset
        self.trend_path = f"/home/majanvie/scratch/loader_detrend/trend_temp_month{valid_months[0]}_lead{lead_time_idx}.nc"
        self.trend_path_truth = f"/home/majanvie/scratch/loader_detrend/trend_truth_temp_month{valid_months[0]}_{lead_time_idx}.nc"
        self.scaler_path = f"/home/majanvie/scratch/loader_detrend/scaler_temp_month{valid_months[0]}_{lead_time_idx}.nc"
        self.index_path = f"/home/majanvie/scratch/loader_detrend/{self.subset}_index_month{valid_months[0]}_{lead_time_idx}.csv"

        # Open observations
        self.obs = xr.open_mfdataset(obs_path + '/*.nc', combine='by_coords')
        self.latitude = self.obs.latitude.values
        self.longitude = self.obs.longitude.values

        # Build index of counts for files
        if os.path.exists(self.index_path):
            self.data_index = pd.read_csv(self.index_path).values.tolist()
        else:
            self.data_index = []
            mean_path = f"{data_path}/mean"
            std_path = f"{data_path}/std"
            for f in os.listdir(mean_path):
                f_mean = f"{mean_path}/{f}"
                f_std = f"{std_path}/{f}"
                data_mean = xr.open_dataset(f_mean).isel(prediction_timedelta=self.lead_time_idx) # select lead time

                # Iterate over remaining time dimension
                if self.subset == "test":
                    time_size = data_mean.sizes["time"]
                else:
                    time_size = data_mean.sizes["forecast_time"]

                for time_idx in range(time_size):
                    if self.subset == "test":
                        forecast_time = pd.to_datetime(data_mean.isel(time=time_idx).time.values)
                        valid_time = pd.to_datetime(data_mean.isel(time=time_idx).valid_time.values)
                    else:
                        forecast_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).time.values)
                        valid_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).valid_time.values)
                
                    if valid_time.year not in self.valid_years:
                        continue
                    elif valid_time.month not in self.valid_months:   
                        continue
                    self.data_index.append(
                        (f_mean, f_std, time_idx, valid_time, forecast_time)
                    )
            
            # Adapt index for train or val
            if self.subset == "train":
                self.data_index = [x for x in self.data_index
                    if x[3].year < self.validation_year_begin 
                ]
            elif self.subset == "val":
                self.data_index = [x for x in self.data_index
                    if x[3].year >= self.validation_year_begin 
                ]
            elif self.subset == "test":
                pass
            else:
                raise ValueError("Unrecognized subset")

            #save index
            pd.DataFrame(self.data_index).to_csv(self.index_path, index=False)
        
        # build climatology by fitting gaussians on train data, date wise, 
        # in a netcdf file, if not done already 
        if self.subset == "train":
            climato_path = f"{obs_path}/climato/{self.target_variable}_{self.valid_years[0]}_{self.valid_years[-1]}_month{self.valid_months[0]}_lead{self.lead_time_idx}.nc"
            if not pathlib.Path(climato_path).exists():
                climato = {}
                for _,_, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
                    truth = self.obs.sel(time=valid_time)
                    # compute wind 
                    if self.target_variable == "10m_wind_speed":
                        compute_wind_speedxr(truth,"obs")

                    mu = truth[self.target_variable].values.T
                    lead_time = self.lead_time_idx
                    date = valid_time.strftime('%m-%d') # get date without year
                    if date not in climato.keys():
                        climato[date] = {"mu": [mu]}
                    else:
                        climato[date]["mu"].append(mu) 

                # fit gaussians on mu for each date
                climato_res = []
                for date in climato.keys():
                    mu = np.stack(climato[date]["mu"])
                    mu_fit, sigma_fit = fit_norm_along_axis(mu, axis=0)
                    mu_fit = mu_fit[np.newaxis, ...]
                    sigma_fit = sigma_fit[np.newaxis, ...]
                    ds = xr.Dataset(
                        data_vars=dict(
                            mu=(["date", "latitude", "longitude"], mu_fit),
                            sigma=(["date", "latitude", "longitude"], sigma_fit),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                            date=[date] # single item
                        )
                    )
                    climato_res.append(ds)
                
                # save to csv
                final_ds = xr.concat(climato_res, dim='date')
                final_ds.to_netcdf(climato_path)
        
        # compute train trend
        if os.path.exists(self.trend_path):
            self.trend_model = joblib.load(self.trend_path)
        else:
            self.compute_trend_temp("train")
            joblib.dump(self.trend_model, self.trend_path)

        # compute truth trend
        if os.path.exists(self.trend_path_truth):
            self.trend_model_truth = joblib.load(self.trend_path_truth)
        else:
            self.compute_trend_temp("obs")
            joblib.dump(self.trend_model_truth, self.trend_path_truth)

        # normalize
        if os.path.exists(self.scaler_path):
            self.load_scaler("train")
        else:
            self.fit_scaler("train")
            self.save_scaler("train")
        #self.fit_scaler("train")

    def fit_scaler(self, type): 
        """Get all features and fit the scaler on them."""
        if type == "train": # minmax 
            files = np.unique([x[0] for x in self.data_index])
            #sample_count = 0
            self.running_min_map = None
            self.running_max_map = None
            
            for file_mean in files:
                data = xr.open_dataset(file_mean).isel(prediction_timedelta=self.lead_time_idx) # by lead time as before 
                if self.subset == "test":
                    data = data.rename({'time': 'forecast_time'})
                # detrend temperature data
                X_data = data["2m_temperature"].valid_time.values.astype(np.int64) * 1e-9 # in seconds 
                data["2m_temperature"] = data["2m_temperature"] - self.trend_model.predict(X_data.reshape(-1,1)).reshape(data["2m_temperature"].shape)

                min_map = data.min(dim=["forecast_time"], skipna=True).to_array().values
                max_map = data.max(dim=["forecast_time"], skipna=True).to_array().values
                # manage NaN
                min_map = np.nan_to_num(min_map, nan=0)
                max_map = np.nan_to_num(max_map, nan=1)
                #print(min_map.shape)

                if self.running_min_map is None:
                    self.running_min_map = min_map
                    self.running_max_map = max_map
                else:
                    self.running_min_map = np.minimum(self.running_min_map, min_map)
                    self.running_max_map = np.maximum(self.running_max_map, max_map)
        
        # elif type == "obs": # mean/std normalization
        #     valid_times = [x[3] for x in self.data_index]
        #     truth = self.obs.sel(time=valid_times)
        #     compute_wind_speedxr(truth,"obs")
        #     if self.target_variable == "2m_temperature":
        #         # detrend temperature
        #         truth["2m_temperature"] = truth["2m_temperature"] - self.trend_model_truth.predict(truth["2m_temperature"].time.values.astype(np.int64).reshape(-1,1))
        #     truth = truth[self.target_variable]
        #     self.mean_truth = truth.mean(dim="time").to_array().values.transpose(0,2,1)
        #     self.std_truth = truth.std(dim="time").to_array().values.transpose(0,2,1)
    
    def compute_trend_temp(self, type):
        if type == 'obs':
            valid_times = [x[3] for x in self.data_index]
            truth = self.obs["2m_temperature"].sel(time=valid_times)
            times = truth.time.values
            truth = truth.values
            num_times = np.array(pd.to_datetime(times).astype(np.int64)) * 1e-9 # in seconds for stability
            model = LinearRegression()
            model.fit(num_times.reshape(-1,1), truth.reshape(truth.shape[0], -1))
            self.trend_model_truth = model

        if type == "train":
            files = np.unique([x[0] for x in self.data_index])
            X, y = None, None
            for file in files:
                data = xr.open_dataset(file)["2m_temperature"].isel(prediction_timedelta=0)
                if self.subset == "test":
                    data = data.rename({'time': 'forecast_time'})
                times = data.valid_time.values
                num_times = np.array(pd.to_datetime(times).astype(np.int64)) *1e-9 # in seconds for stability
                if X is None:
                    X = num_times
                    y = data.values.reshape(data.shape[0], -1)
                else:
                    X = np.concatenate((X, num_times))
                    y = np.concatenate((y, data.values.reshape(data.shape[0], -1)))
            model = LinearRegression()
            model.fit(X.reshape(-1,1), y)
            self.trend_model = model

    
    def load_scaler(self, type):
        """Load the scaler from a file."""
        if type == "train":
            ds = xr.open_dataset(self.scaler_path)
            self.running_min_map = ds["min"].values
            self.running_max_map = ds["max"].values
        # elif type == "obs":
        #     ds = xr.open_dataset(self.scaler_truth_path)
        #     self.mean_truth = ds["mean"].values
        #     self.std_truth = ds["std"].values

    def save_scaler(self, type):
        """Save the fitted scaler to a file."""
        if type == "train":
            ds = xr.Dataset(
                        data_vars=dict(
                            min=(["vars","latitude", "longitude"], self.running_min_map),
                            max=(["vars","latitude", "longitude"], self.running_max_map),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                            vars=("vars", np.arange(self.running_min_map.shape[0]))
                        )
                    )
            ds.to_netcdf(self.scaler_path)
        # elif type == "obs":
        #     ds = xr.Dataset(
        #                 data_vars=dict(
        #                     mean=(["vars","latitude", "longitude"], self.mean_truth),
        #                     std=(["vars","latitude", "longitude"], self.std_truth),
        #                 ),
        #                 coords=dict(
        #                     longitude=("longitude", self.longitude), # 1D array
        #                     latitude=("latitude", self.latitude), # 1D array
        #                     vars=("vars", np.arange(self.mean_truth.shape[0]))
        #                 )
        #             )
        #     ds.to_netcdf(self.scaler_truth_path)
                   

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time = self.data_index[idx]
        data_mean = xr.open_dataset(file_mean)
        data_std = xr.open_dataset(file_std)
        
        # select forecast data
        if self.subset == "test":
            # rename time to forecast_time
            data_mean = data_mean.rename({'time': 'forecast_time'})
            data_std = data_std.rename({'time': 'forecast_time'})
        data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        data_std = data_std.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        truth = self.obs.sel(time=valid_time).transpose("latitude", "longitude") # initially (lon,lat)
        land_sea_mask = truth["land_sea_mask"].values

        # compute wind 
        if self.target_variable == "10m_wind_speed":
            compute_wind_speedxr(truth,"obs")

        # detrend truth temperature data with Y TREND (test2)  
        if self.subset in ["train", "val"]:
            if self.target_variable == "2m_temperature":
                # detrend temperature
                X_truth = truth["2m_temperature"].time.values.astype(np.int64) * 1e-9
                #print("SHAPE", truth["2m_temperature"].shape)
                shape = truth["2m_temperature"].shape
                # truth trend saved in (lon,lat) instead of (lat,lon)
                truth["2m_temperature"] = truth["2m_temperature"] - self.trend_model_truth.predict(X_truth.reshape(-1,1)).reshape(shape[1],shape[0]).T
                #print("truth truth", truth["2m_temperature"].values.mean())

        
        truth = truth[self.target_variable].values
        # normalize truth data
        #truth = (truth - self.mean_truth)/self.std_truth
        truth = torch.tensor(truth, dtype=torch.float) 

        # detrend prior and feature temperature data
        X_data = data_mean["2m_temperature"].valid_time.values.astype(np.int64) * 1e-9
        data_mean["2m_temperature"] = data_mean["2m_temperature"] - self.trend_model.predict(X_data.reshape(-1,1)).reshape(data_mean["2m_temperature"].shape)
        # Normalize features using min max 
        data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        data = (data-self.running_min_map)/(self.running_max_map-self.running_min_map + 1e-3)
        data = np.where(np.isnan(data), 0, data)

        # add land sea mask feature
        data = np.concatenate([data, land_sea_mask[np.newaxis, ...]], axis=0)
        # -1,1
        #data = 2*data - 1
        features = torch.tensor(data, dtype=torch.float) # 67x121x240

        # Manage Nan
        data_mean = data_mean.fillna(0) # temperature is detrended 
        data_std = data_std.fillna(1)

        # mean std over ensemble predictions for the target variable
        mu = torch.tensor(data_mean[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        sigma = torch.tensor(data_std[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        valid_time = str(valid_time)
        forecast_time = str(forecast_time)
        # STOP MEMORY LEAK
        data_mean.close()
        data_std.close()
        
        return {
            'mu': mu, 'sigma': sigma, 'input': features, 'truth': truth,
            'valid_time': valid_time, "lead_time": self.lead_time_idx, "forecast_time": forecast_time
            }  



class WeatherEnsembleDataset2:
    """Data in files"""
    def __init__(
        self,
        data_path,
        obs_path,
        target_variable,
        lead_time_idx,
        valid_years,
        valid_months,
        subset
    ):
        self.lead_time_idx = round(lead_time_idx/7)
        self.target_variable = target_variable
        self.valid_months = np.arange(valid_months[0], valid_months[1] + 1)
        self.valid_years = np.arange(valid_years[0], valid_years[1] + 1)
        self.validation_year_begin = self.valid_years[len(self.valid_years)*9//10]
        self.subset = subset

        # Open observations
        self.obs = xr.open_mfdataset(obs_path + '/*.nc', combine='by_coords')
        self.latitude = self.obs.latitude.values
        self.longitude = self.obs.longitude.values

        # Build index of counts for files
        self.data_index = []
        mean_path = f"{data_path}/mean"
        std_path = f"{data_path}/std"
        for f in os.listdir(mean_path):
            f_mean = f"{mean_path}/{f}"
            f_std = f"{std_path}/{f}"
            data_mean = xr.open_dataset(f_mean).isel(prediction_timedelta=self.lead_time_idx) # select lead time

            # Iterate over remaining time dimension
            if self.subset == "test":
                time_size = data_mean.sizes["time"]
            else:
                time_size = data_mean.sizes["forecast_time"]

            for time_idx in range(time_size):
                if self.subset == "test":
                    forecast_time = pd.to_datetime(data_mean.isel(time=time_idx).time.values)
                    valid_time = pd.to_datetime(data_mean.isel(time=time_idx).valid_time.values)
                else:
                    forecast_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).time.values)
                    valid_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).valid_time.values)
            
                if valid_time.year not in self.valid_years:
                    continue
                elif valid_time.month not in self.valid_months:   
                    continue
                self.data_index.append(
                    (f_mean, f_std, time_idx, valid_time, forecast_time)
                )
        
        # Adapt index for train or val
        if self.subset == "train":
            self.data_index = [x for x in self.data_index
                if x[3].year < self.validation_year_begin 
            ]
        elif self.subset == "val":
            self.data_index = [x for x in self.data_index
                if x[3].year >= self.validation_year_begin 
            ]
        elif self.subset == "test":
            pass
        else:
            raise ValueError("Unrecognized subset")
        
        # build climatology by fitting gaussians on train data, date wise, 
        # in a netcdf file, if not done already 
        if self.subset == "train":
            climato_path = f"{obs_path}/climato/{self.target_variable}_{self.valid_years[0]}_{self.valid_years[-1]}_month{self.valid_months[0]}_lead{self.lead_time_idx}.nc"
            if not pathlib.Path(climato_path).exists():
                climato = {}
                for _,_, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
                    truth = self.obs.sel(time=valid_time)
                    # compute wind 
                    if self.target_variable == "10m_wind_speed":
                        compute_wind_speedxr(truth,"obs")

                    mu = truth[self.target_variable].values.T
                    lead_time = self.lead_time_idx
                    date = valid_time.strftime('%m-%d') # get date without year
                    if date not in climato.keys():
                        climato[date] = {"mu": [mu]}
                    else:
                        climato[date]["mu"].append(mu) 

                # fit gaussians on mu for each date
                climato_res = []
                for date in climato.keys():
                    mu = np.stack(climato[date]["mu"])
                    mu_fit, sigma_fit = fit_norm_along_axis(mu, axis=0)
                    mu_fit = mu_fit[np.newaxis, ...]
                    sigma_fit = sigma_fit[np.newaxis, ...]
                    ds = xr.Dataset(
                        data_vars=dict(
                            mu=(["date", "latitude", "longitude"], mu_fit),
                            sigma=(["date", "latitude", "longitude"], sigma_fit),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                            date=[date] # single item
                        )
                    )
                    climato_res.append(ds)
                
                # save to csv
                final_ds = xr.concat(climato_res, dim='date')
                final_ds.to_netcdf(climato_path)

        # normalize
        #self.scaler = StandardScaler()
        self.fit_scaler()
        

    def fit_scaler(self):
        """Get all features and fit the scaler on them."""
        all_features = []
        all_features_t = []
        for file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
            data_mean = xr.open_dataset(file_mean)
            if self.subset == "test":
                data_mean = data_mean.rename({'time': 'forecast_time'})
            data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
            data_mean = data_mean.to_array().values # (n_vars, n_lat, n_lon)
            all_features.append(data_mean)

            # truth data normalisation
            truth = self.obs.sel(time=valid_time)
            compute_wind_speedxr(truth,"obs")
            all_features_t.append(truth[self.target_variable].values.T)

        all_features = np.stack(all_features) # (n_files, n_vars, n_lat, n_lon)
        # do mean without considering NaN 
        self.mean = np.nanmean(all_features,axis=0) # (n_vars, n_lat, n_lon)
        self.std = np.nanstd(all_features,axis=0)
        self.mean = np.where(np.isnan(self.mean), 0, self.mean) # replace NaN by 0
        self.std = np.where(self.std == 0, 1, self.std) # replace 0 by 1 for future standardisation

        all_features_t = np.stack(all_features_t) # (n_files, n_vars, n_lat, n_lon)
        # do mean without considering NaN 
        self.mean_t = np.nanmean(all_features_t,axis=0) # (n_vars, n_lat, n_lon)
        self.std_t = np.nanstd(all_features_t,axis=0)
        self.mean_t = np.where(np.isnan(self.mean_t), 0, self.mean_t) # replace NaN by 0
        self.std_t = np.where(self.std_t == 0, 1, self.std_t) # replace 0 by 1 for future standardisation

    def fit_scaler(self):
        """Get all features and fit the scaler on them."""
        all_features = []
        all_features_t = []
        for file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
            data_mean = xr.open_dataset(file_mean)

            if self.subset == "test":
                data_mean = data_mean.rename({'time': 'forecast_time'})
            data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
            data_mean = data_mean.fillna(0) # manage NaN
            data_mean = data_mean.to_array().values # (n_vars, n_lat, n_lon)
            all_features.append(data_mean)

            # truth data normalisation
            truth = self.obs.sel(time=valid_time)
            compute_wind_speedxr(truth,"obs")
            all_features_t.append(truth[self.target_variable].values.T)

        #all_features = np.concatenate(all_features, axis=0) # Concatenate along the feature dimension
        all_features = np.stack(all_features) # (n_files, n_vars, n_lat, n_lon)
        self.mean = np.mean(all_features,axis=0) # (n_vars, n_lat, n_lon)
        self.std = np.std(all_features,axis=0) # (n_vars, n_lat, n_lon)

        all_features_t = np.stack(all_features_t) # (n_files, n_lat, n_lon)
        self.mean_t = np.mean(all_features_t,axis=0) # (n_lat, n_lon)
        self.std_t = np.std(all_features_t,axis=0) # (n_lat, n_lon)


    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time = self.data_index[idx]
        data_mean = xr.open_dataset(file_mean)
        data_std = xr.open_dataset(file_std)
        # select forecast data
        if self.subset == "test":
            # rename time to forecast_time
            data_mean = data_mean.rename({'time': 'forecast_time'})
            data_std = data_std.rename({'time': 'forecast_time'})
        data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        data_std = data_std.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        truth = self.obs.sel(time=valid_time)

        # Manage Nan
        data_mean = data_mean.fillna(0)
        data_std = data_std.fillna(0)
        #print(self.scaler.mean_.shape)

        # Normalize features
        data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        data = (data-self.mean)/self.std
        # shapes = data.shape
        # data = data.reshape(data.shape[0], -1) # Flatten spatial dimensions
        # data = self.scaler.transform(data) # Normalize features
        # data = data.reshape(shapes[0], shapes[1], shapes[2]) # Reshape back
        features = torch.tensor(data, dtype=torch.float)

        # compute wind 
        if self.target_variable == "10m_wind_speed":
            compute_wind_speedxr(truth,"obs")

        # mean std over ensemble predictions for the target variable
        mu = torch.tensor(data_mean[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        sigma = torch.tensor(data_std[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        valid_time = str(valid_time)
        forecast_time = str(forecast_time)
        # get associated ground truth
        truth = truth[self.target_variable].values.T
        truth = (truth - self.mean_t)/self.std_t
        truth = torch.tensor(truth, dtype=torch.float) # (n_lat, n_lon)
        return {
            'mu': mu, 'sigma': sigma, 'input': features, 'truth': truth,
            'valid_time': valid_time, "lead_time": self.lead_time_idx, "forecast_time": forecast_time
            }   

class WeatherYearEnsembleDataset:
    """Data in files"""
    def __init__(
        self,
        data_path,
        obs_path,
        valid_years,
        subset
    ):
        #self.lead_time_idx = round(lead_time_idx/7)
        #self.target_variable = target_variable
        self.idx2lead = {0:0, 1:7, 2:14, 3:21, 4:28, 5:35, 6:39}
        self.data_path = data_path
        self.obs_path = obs_path
        self.valid_years = np.arange(valid_years[0], valid_years[1] + 1)
        self.validation_year_begin = self.valid_years[len(self.valid_years)*9//10]
        self.subset = subset
        self.index_path = f"/home/majanvie/scratch/loader/{subset}_index.csv"
        #self.scaler_path = f"/home/majanvie/scratch/loader/scaler.nc"
        #self.scaler_truth_path = f"/home/majanvie/scratch/loader/scaler_truth.nc"
        self.trend_path = f"/home/majanvie/scratch/loader/trend_temp.nc"
        self.trend_truth_path = f"/home/majanvie/scratch/loader/trend_temp_truth.nc"

        # Open observations
        self.obs = xr.open_mfdataset(obs_path + '/*.nc', combine='by_coords')
        self.latitude = self.obs.latitude.values[21:]
        self.longitude = self.obs.longitude.values
        self.lat_norm = (self.latitude - self.latitude.min())/(self.latitude.max() - self.latitude.min())
        self.lon_norm = (self.longitude - self.longitude.min())/(self.longitude.max() - self.longitude.min())

        # build index of counts for files
        if os.path.exists(self.index_path):
            print("loading index")
            self.data_index = pd.read_csv(self.index_path).values.tolist()
        else:
            print("building index")
            self.build_index()
            self.save_index()
        
        # not give original forecast 
        # TODO build scaler trend for them 
        self.data_index = [x for x in self.data_index
                if x[5] > 0
            ]

        print("len index", len(self.data_index))

        # build climato
        self.build_climato()

        # compute train trend
        if os.path.exists(self.trend_path):
            self.trend_model = joblib.load(self.trend_path) 
        else:
            self.compute_trend_temp("train")
            self.save_trend_temp("train")

        # compute truth trends
        if os.path.exists(self.trend_truth_path):
            self.trend_model_truth = joblib.load(self.trend_truth_path)
        else:
            self.compute_trend_temp("obs")
            self.save_trend_temp("obs")

        # normalize training data
        # if os.path.exists(self.scaler_path):
        #     print("loading scaler")
        #     self.load_scaler("train")   
        # else:
        #     print("building scaler")
        #     self.fit_scaler("train")
        #     self.save_scaler("train")

        # normalize truth data 
        # self.scaler_truth = StandardScaler()
        # if os.path.exists(self.scaler_truth_path):
        #     self.load_scaler("obs")
        #     print("loading truth scaler")
        # else:
        #     self.fit_scaler("obs")
        #     self.save_scaler("obs")

    def build_index(self):
        # Build index of counts for files
        self.data_index = []
        mean_path = f"{self.data_path}/mean"
        std_path = f"{self.data_path}/std"
        for f in os.listdir(mean_path):
            f_mean = f"{mean_path}/{f}"
            f_std = f"{std_path}/{f}"
            for lead_time_idx in range(7):
                data_mean = xr.open_dataset(f_mean).isel(prediction_timedelta=lead_time_idx) # select lead time
                # Iterate over remaining time dimension
                if self.subset == "test":
                    time_size = data_mean.sizes["time"]
                else:
                    time_size = data_mean.sizes["forecast_time"]

                for time_idx in range(time_size):
                    if self.subset == "test":
                        forecast_time = pd.to_datetime(data_mean.isel(time=time_idx).time.values)
                        valid_time = pd.to_datetime(data_mean.isel(time=time_idx).valid_time.values)
                    else:
                        forecast_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).time.values)
                        valid_time = pd.to_datetime(data_mean.isel(forecast_time=time_idx).valid_time.values)
                
                    if valid_time.year not in self.valid_years:
                        continue
                    self.data_index.append(
                        (f_mean, f_std, time_idx, valid_time, forecast_time, lead_time_idx)
                    )
        
        # Adapt index for train or val and save 
        if self.subset == "train":
            self.data_index = [x for x in self.data_index
                if x[3].year < self.validation_year_begin 
            ]
        elif self.subset == "val":
            self.data_index = [x for x in self.data_index
                if x[3].year >= self.validation_year_begin 
            ]
        elif self.subset == "test":
            pass
        else:
            raise ValueError("Unrecognized subset")

    def save_index(self):
        # save to csv
        pd.DataFrame(self.data_index).to_csv(self.index_path, index=False)
    
    def build_climato(self):
        # build climatology by fitting gaussians on train data, date wise, 
        # in a netcdf file, if not done already 
        if self.subset == "train":
            climato_path = f"{self.obs_path}/climato/{self.valid_years[0]}_{self.valid_years[-1]}.nc"
            if not pathlib.Path(climato_path).exists():
                valid_times = [x[3] for x in self.data_index]
                truth = self.obs.sel(time=valid_times)
                compute_wind_speedxr(truth,"obs")
                mu = truth.groupby("time.dayofyear").mean(dim=["time"])
                sigma = truth.groupby("time.dayofyear").std(dim=["time"])
                # rename to avoid conflict 
                ds_mu = mu.rename({var: f"{var}_mean" for var in mu.data_vars})
                ds_sigma = sigma.rename({var: f"{var}_std" for var in sigma.data_vars})
                # Merge the datasets
                combined_ds = xr.merge([ds_mu, ds_sigma])
                # save to file
                combined_ds.to_netcdf(climato_path)

    def fit_scaler(self, type):
        """Get all features and fit the scaler on them."""
        if type == "train":
            files = np.unique([x[0] for x in self.data_index])
            #sample_count = 0
            self.running_min_map = None
            self.running_max_map = None

            for file_mean in files:
                data = xr.open_dataset(file_mean) # all lead times 
                if self.subset == "test":
                    data = data.rename({'time': 'forecast_time'})
                # detrend temperature data
                X_data = data["2m_temperature"].valid_time.values.astype(np.int64) * 1e-9 # in seconds 
                data["2m_temperature"] = data["2m_temperature"] - self.trend_model.predict(X_data.reshape(-1,1)).reshape(data["2m_temperature"].shape)

                min_map = data.min(dim=["forecast_time", "prediction_timedelta"], skipna=True).to_array().values
                max_map = data.max(dim=["forecast_time", "prediction_timedelta"], skipna=True).to_array().values
                # manage NaN
                min_map = np.nan_to_num(min_map, nan=0)
                max_map = np.nan_to_num(max_map, nan=1)
                #print(min_map.shape)

                if self.running_min_map is None:
                    self.running_min_map = min_map
                    self.running_max_map = max_map
                else:
                    self.running_min_map = np.minimum(self.running_min_map, min_map)
                    self.running_max_map = np.maximum(self.running_max_map, max_map)
        
        # elif type == "obs": # mean/std normalization
        #     valid_times = [x[3] for x in self.data_index]
        #     truth = self.obs.sel(time=valid_times)
        #     compute_wind_speedxr(truth,"obs")
        #     if self.target_variable == "2m_temperature":
        #         # detrend temperature
        #         truth["2m_temperature"] = truth["2m_temperature"] - self.trend_model_truth.predict(truth["2m_temperature"].time.values.astype(np.int64).reshape(-1,1))
        #     truth = truth[self.target_variable]
        #     self.mean_truth = truth.mean(dim="time").to_array().values.transpose(0,2,1)
        #     self.std_truth = truth.std(dim="time").to_array().values.transpose(0,2,1)


    def compute_trend_temp(self, type):
        if type == 'obs':
            valid_times = [x[3] for x in self.data_index]
            truth = self.obs["2m_temperature"].sel(time=valid_times)
            times = truth.time.values
            truth = truth.values
            num_times = np.array(pd.to_datetime(times).astype(np.int64)) * 1e-9 # in seconds for stability
            model = LinearRegression()
            model.fit(num_times.reshape(-1,1), truth.reshape(truth.shape[0], -1))
            self.trend_model_truth = model

        elif type == "train":
            files = np.unique([x[0] for x in self.data_index])
            X, y = None, None
            for file in files:
                data = xr.open_dataset(file)["2m_temperature"].isel(prediction_timedelta=0)
                if self.subset == "test":
                    data = data.rename({'time': 'forecast_time'})
                times = data.valid_time.values
                num_times = np.array(pd.to_datetime(times).astype(np.int64)) *1e-9 # in seconds for stability
                if X is None:
                    X = num_times
                    y = data.values.reshape(data.shape[0], -1)
                else:
                    X = np.concatenate((X, num_times))
                    y = np.concatenate((y, data.values.reshape(data.shape[0], -1)))
            model = LinearRegression()
            model.fit(X.reshape(-1,1), y)
            self.trend_model = model


    def save_trend_temp(self, type):
        if type == "obs":
            joblib.dump(self.trend_model_truth, self.trend_truth_path)
        elif type == "train":
            joblib.dump(self.trend_model, self.trend_path)
    

    def save_scaler(self, type):
        """Save the fitted scaler to a file."""
        if type == "train":
            ds = xr.Dataset(
                        data_vars=dict(
                            min=(["vars","latitude", "longitude"], self.running_min_map),
                            max=(["vars","latitude", "longitude"], self.running_max_map),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                            vars=("vars", np.arange(self.running_min_map.shape[0]))
                        )
                    )
            ds.to_netcdf(self.scaler_path)
        elif type == "obs":
            ds = xr.Dataset(
                        data_vars=dict(
                            mean=(["vars","latitude", "longitude"], self.mean_truth),
                            std=(["vars","latitude", "longitude"], self.std_truth),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                            vars=("vars", np.arange(self.mean_truth.shape[0]))
                        )
                    )
            ds.to_netcdf(self.scaler_truth_path)
    
    def load_scaler(self, scaler_path, type):
        """Load the scaler from a file."""
        if type == "train":
            ds = xr.open_dataset(scaler_path)
            self.running_min_map = ds["min"].values
            self.running_max_map = ds["max"].values
        # elif type == "obs":
        #     ds = xr.open_dataset(self.scaler_truth_path)
        #     self.mean_truth = ds["mean"].values
        #     self.std_truth = ds["std"].values

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time, lead_time_idx = self.data_index[idx]
        data_mean = xr.open_dataset(file_mean)
        data_std = xr.open_dataset(file_std)
        valid_time = pd.to_datetime(valid_time)
        forecast_time = pd.to_datetime(forecast_time)
        valid_month = valid_time.month

        # load scaler 
        scaler_path = f"/home/majanvie/scratch/loader_detrend/scaler_temp_month{valid_month}_{self.idx2lead[lead_time_idx]}.nc"
        self.load_scaler(scaler_path, "train")
        # select forecast data
        if self.subset == "test":
            # rename time to forecast_time
            data_mean = data_mean.rename({'time': 'forecast_time'})
            data_std = data_std.rename({'time': 'forecast_time'})
        data_mean = data_mean.isel(prediction_timedelta=lead_time_idx, forecast_time=forecast_idx_in_file)
        data_std = data_std.isel(prediction_timedelta=lead_time_idx, forecast_time=forecast_idx_in_file)
        data_std = data_std.fillna(1)
        truth = self.obs.sel(time=valid_time).transpose("latitude", "longitude") # initially (lon,lat)
        land_sea_mask = truth["land_sea_mask"].values[21:,:] # remove Antartica

        # compute wind
        compute_wind_speedxr(truth,"obs")

        if self.subset in ["train", "val"]:
            # detrend truth temperature data
            X_truth = truth["2m_temperature"].time.values.astype(np.int64) * 1e-9
            shape = truth["2m_temperature"].shape
            truth["2m_temperature"] = truth["2m_temperature"] - self.trend_model_truth.predict(X_truth.reshape(-1,1)).reshape(shape[1],shape[0]).T 
        truth = truth[["2m_temperature", "10m_wind_speed"]].to_array().values # (2, n_lat, n_lon)
        truth = torch.tensor(truth[:,21:,:], dtype=torch.float) # 2x100x240, crop Antartica

        # detrend temperature data
        X_data = data_mean["2m_temperature"].valid_time.values.astype(np.int64) * 1e-9
        data_mean["2m_temperature"] = data_mean["2m_temperature"] - self.trend_model.predict(X_data.reshape(-1,1)).reshape(data_mean["2m_temperature"].shape)
        # normalize data 
        data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        data = (data-self.running_min_map)/(self.running_max_map-self.running_min_map + 1e-3)
        data = np.where(np.isnan(data), 0, data)

        # add std data
        val_std = data_std[["2m_temperature", "10m_wind_speed"]].to_array().values
        data = np.concatenate([data[:,21:,:], val_std[:,21:,:]], axis=0)

        # add land sea mask feature
        data = np.concatenate([data, land_sea_mask[np.newaxis, ...]], axis=0)  
        # add lead time and day of year and lat lon as features
        shapes = data.shape
        lat = np.ones((shapes[1], shapes[2])) * self.lat_norm[:, np.newaxis]  # Broadcasting along axis 1 (lat)
        lon = np.ones((shapes[1], shapes[2])) * self.lon_norm[np.newaxis, :]  # Broadcasting along axis 2 (lon)
        lead_time = np.ones((shapes[1], shapes[2])) * (lead_time_idx/7)
        year = valid_time.year
        doy = valid_time.timetuple().tm_yday
        day_of_year = valid_time.timetuple().tm_yday / 366
        sin_day_of_year = np.ones((shapes[1], shapes[2])) * np.sin(2*np.pi*day_of_year)
        cos_day_of_year = np.ones((shapes[1], shapes[2])) * np.cos(2*np.pi*day_of_year)
        data = np.concatenate([data,
            lead_time[np.newaxis, ...],
            sin_day_of_year[np.newaxis, ...],
            cos_day_of_year[np.newaxis, ...],
            lat[np.newaxis, ...],
            lon[np.newaxis, ...]], axis=0)
        features = torch.tensor(data, dtype=torch.float) #74x100x240

        valid_time = str(valid_time)
        forecast_time = str(forecast_time)
        # STOP MEMORY LEAK 
        data_mean.close() 
        data_std.close()
        return {
            'input': features, 'truth': truth, "day_of_year": doy,
            'valid_time': valid_time, "lead_time": lead_time_idx, "forecast_time": forecast_time
            }   



if __name__== "__main__":
    ### PandasDataset 
    # data_folder = "../scratch/data/train/"
    # train_data = pd.read_json(data_folder+"PPE_OPT_lat=-90.0_lon=0.0_lead=24h.json")
    # train_data = compute_wind_speed(train_data)
    # print(train_data.shape)

    # # separate train and validation randomly
    # train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True)

    # train_data = PandasDataset(train_data, "10m_wind_speed")
    # train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    # val_data = PandasDataset(val_data, target_column="10m_wind_speed")
    # val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    # for batch in train_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(" ")

    #     print("VALUES")
    #     print(batch['mu'], batch['sigma'], batch['input'], batch['truth'], batch['forecast_time'])
    #     break

    # for batch in val_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(" ")

    #     print("VALUES")
    #     print(batch['mu'], batch['sigma'], batch['input'], batch['truth'], batch['forecast_time'])
    #     break

    ### WeatherDataset
    # data_folder = "/home/majanvie/scratch/data/raw"
    # train_folder = f"{data_folder}/train"
    # test_folder = f"{data_folder}/test"
    # obs_folder = f"{data_folder}/obs"

    # test_dataset = WeatherDataset(
    #     data_path=test_folder,
    #     obs_path=obs_folder,
    #     target_variable="2m_temperature",
    #     lead_time_idx=28,
    #     valid_years=[2018,2022],
    #     valid_months=[1,3],
    #     subset="test")
    
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(test_loader.dataset.data_index))
    # for batch in test_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")
    #     break

    # for variable in ["10m_wind_speed"]:
    #     print("Variable",variable)
    #     for month in range(4,5):
    #         print("Month", month)
    #         dtime = time.time()
    #         train_dataset = WeatherDataset(
    #             data_path=train_folder,
    #             obs_path=obs_folder,
    #             target_variable=variable,
    #             lead_time_idx=14,
    #             valid_years=[1996,2017],
    #             valid_months=[month,month],
    #             subset="train")
    #         print("dataset time", time.time()-dtime)
    #                 #break
    #             #break
    # print("done climato")

            # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            # print("Nb of training examples:",len(train_loader.dataset.data_index))
            # for batch in train_loader:
            #     print("SHAPES")
            #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
            #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
            #     print(" ")
            #     break
    # dtime = time.time()
    # val_dataset = WeatherDataset(
    #     data_path=train_folder,
    #     obs_path=obs_folder,
    #     target_variable="10m_wind_speed",
    #     lead_time_idx=28,
    #     valid_years=[1996,2017],
    #     valid_months=[1,1],
    #     subset="val")
    # print("dataset time", time.time()-dtime)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(val_loader.dataset.data_index))
    # dtime = time.time()
    # for batch in val_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(" ")
    #     break
    # print(time.time()-dtime)

    # print(val_dataset.data_index[0])
    # print(train_dataset.data_index[0])


    ### WeatherEnsembleDataset
    # data_folder = "/home/majanvie/scratch/data" 
    # train_folder = f"{data_folder}/train/EMOS"
    # test_folder = f"{data_folder}/test/EMOS"
    # obs_folder = "/home/majanvie/scratch/data/raw/obs"

    # dtime = time.time()

    # train_dataset = WeatherEnsembleDataset(
    #     data_path=train_folder,
    #     obs_path=obs_folder,
    #     target_variable="10m_wind_speed",
    #     lead_time_idx=14,
    #     valid_years=[1996,1997],
    #     valid_months=[1,1],
    #     subset="train")

    # print("dataset time", time.time()-dtime)
    

    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(train_loader.dataset.data_index))
    # dtime = time.time()
    # for batch in train_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")
    #     print(batch["input"])
    #     print(batch["input"][0][0][10][10].mean())
    #     print(batch["input"].min(), batch["input"].max(), batch["input"].mean())
    #     break
    # print("getitem",time.time()-dtime)

    # test_dataset = WeatherEnsembleDataset(
    #     data_path=test_folder,
    #     obs_path=obs_folder,
    #     target_variable="10m_wind_speed",
    #     lead_time_idx=14,
    #     valid_years=[2018,2022],
    #     valid_months=[1,1],
    #     subset="test")

    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(test_loader.dataset.data_index))
    # for batch in test_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")
    #     print(batch["input"][0,:,0,0].mean(), batch["input"][0,0,:,:].mean())
    #     break


    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(train_loader.dataset.data_index))
    # dtime = time.time()
    # for batch in train_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")
    #     break
    # print("getitem",time.time()-dtime)
    
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(train_loader.dataset.data_index))
    # dtime = time.time()
    # for batch in train_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")
    #     break
    # print("getitem",time.time()-dtime)

    # val_dataset = WeatherEnsembleDataset(
    #     data_path=train_folder,
    #     obs_path=obs_folder,
    #     target_variable="10m_wind_speed",
    #     lead_time_idx=14,
    #     valid_years=[1996,2017],
    #     valid_months=[1,1],
    #     subset="val")

    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(val_loader.dataset.data_index))
    # for batch in val_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")
    #     break

    ### WeatherYearEnsembleDataset
    # data_folder = "/home/majanvie/scratch/data"
    # train_folder = f"{data_folder}/train/EMOS"
    # test_folder = f"{data_folder}/test/EMOS"
    # obs_folder = "/home/majanvie/scratch/data/raw/obs"

    # dtime = time.time()
    # train_dataset = WeatherYearEnsembleDataset(
    #     data_path=train_folder,
    #     obs_path=obs_folder,
    #     valid_years=[1996,2017],
    #     subset="train")
    # print("dataset time", time.time()-dtime)
    
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(train_loader.dataset.data_index))
    # i = 0
    # for batch in train_loader:
    #     print("SHAPES")
    #     print(batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")

    #     print("VALUES")
    #     print(batch["input"].min(), batch["input"].max(), batch["input"].mean())
    #     print(batch["truth"])
    #     if i == 3:
    #         break
    #     i +=1 

    # test 
    # dtime = time.time()
    # test_dataset = WeatherYearEnsembleDataset(
    #     data_path=test_folder,
    #     obs_path=obs_folder,
    #     valid_years=[2018,2022],
    #     subset="test")
    # print("dataset time", time.time()-dtime)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(test_loader.dataset.data_index))
    # for batch in test_loader:
    #     print("SHAPES")
    #     print(batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")
    #     print(batch["input"])
    #     print(batch["input"].min(), batch["input"].max(), batch["input"].mean())
    #     print(batch["truth"].min(), batch["truth"].max(), batch["truth"].mean())
    #     break

    # val
    # dtime = time.time()
    # val_dataset = WeatherYearEnsembleDataset(
    #     data_path=train_folder,
    #     obs_path=obs_folder,
    #     valid_years=[1996,2017],
    #     subset="val")
    # print("dataset time", time.time()-dtime)
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(val_loader.dataset.data_index))
    # for batch in val_loader:
    #     print("SHAPES")
    #     print(batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")
    #     break

    ## DETREND 
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    
    train_dataset = WeatherEnsembleDatasetMMdetrend(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable="2m_temperature",
        lead_time_idx=14,
        valid_years=[1996,2017],
        valid_months=[1,1],
        subset="train")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for batch in train_loader:
        print("SHAPES")
        print(batch['input'].shape, batch['truth'].shape, batch["mu"].shape, batch["sigma"].shape)
        break
