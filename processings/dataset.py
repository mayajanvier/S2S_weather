import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from format_data import compute_wind_speedxr, compute_wind_speed, fit_norm_along_axis, format_ensemble_data_EMOS
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


class WeatherEnsDataset:
    """Data on the fly"""
    def __init__(
        self,
        surface_path,
        levels_path,
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
        for f in os.listdir(surface_path):
            f_surface = f"{surface_path}/{f}"
            f_mean = f"{levels_path}/mean/{f}"
            f_std = f"{levels_path}/std/{f}"
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
                    (f_surface, f_mean, f_std, time_idx, valid_time, forecast_time)
                )
        
        # Adapt index for train or val
        if self.subset == "train":
            self.data_index = [x for x in self.data_index
                if x[4].year < self.validation_year_begin 
            ]
        elif self.subset == "val":
            self.data_index = [x for x in self.data_index
                if x[4].year >= self.validation_year_begin 
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
                for _,_,_, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
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
        for file_surface, file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
            data_mean, data_std = format_ensemble_data_EMOS(file_surface, file_mean, file_std)
            if self.subset == "test":
                data_mean = data_mean.rename({'time': 'forecast_time'})
            data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
            data_mean = data_mean.to_array().values # (n_vars, n_levels, n_lat, n_lon)
            all_features.append(data_mean)

        all_features = np.concatenate(all_features, axis=0) # Concatenate along the feature dimension
        all_features = all_features.reshape(all_features.shape[0], -1) # Flatten spatial dimensions
        self.scaler.fit(all_features)

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_surface, file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time = self.data_index[idx]
        data_mean, data_std = format_ensemble_data_EMOS(file_surface, file_mean, file_std)
        # select forecast data
        if self.subset == "test":
            # rename time to forecast_time
            data_mean = data_mean.rename({'time': 'forecast_time'})
            data_std = data_std.rename({'time': 'forecast_time'})
        data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        data_std = data_std.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
        truth = self.obs.sel(time=valid_time)

        # Normalize features
        data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        shapes = data.shape
        data = data.reshape(data.shape[0], -1) # Flatten spatial dimensions
        data = self.scaler.transform(data) # Normalize features
        data = data.reshape(shapes[0], shapes[1], shapes[2]) # Reshape back
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
        #self.scaler = StandardScaler()
        self.fit_scaler()
        

    # def fit_scaler(self):
    #     """Get all features and fit the scaler on them."""
    #     all_features = []
    #     for file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
    #         data_mean = xr.open_dataset(file_mean)

    #         if self.subset == "test":
    #             data_mean = data_mean.rename({'time': 'forecast_time'})
    #         data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
    #         data_mean = data_mean.fillna(0) # manage NaN
    #         data_mean = data_mean.to_array().values # (n_vars, n_lat, n_lon)
    #         all_features.append(data_mean)

    #     all_features = np.concatenate(all_features, axis=0) # Concatenate along the feature dimension
    #     print(all_features.shape)
    #     all_features = all_features.reshape(all_features.shape[0], -1) # Flatten spatial dimensions
    #     print(all_features.shape)
    #     self.scaler.fit(all_features)

    def fit_scaler(self):
        """Get all features and fit the scaler on them."""
        all_features = []
        for file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time in self.data_index:
            data_mean = xr.open_dataset(file_mean)

            if self.subset == "test":
                data_mean = data_mean.rename({'time': 'forecast_time'})
            data_mean = data_mean.isel(prediction_timedelta=self.lead_time_idx, forecast_time=forecast_idx_in_file)
            data_mean = data_mean.to_array().values # (n_vars, n_lat, n_lon)
            all_features.append(data_mean)

        all_features = np.stack(all_features) # (n_files, n_vars, n_lat, n_lon)
        # do mean without considering NaN 
        self.mean_map = np.nanmean(all_features,axis=0) # (n_vars, n_lat, n_lon)
        self.std_map = np.nanstd(all_features,axis=0)

        self.mean_map = np.where(np.isnan(self.mean), 0, self.mean) # replace NaN by 0
        self.std_map = np.where(np.isnan(self.std), 1, self.std) # replace NaN by 1
        self.std_map = np.where(self.std == 0, 1, self.std) # replace 0 by 1

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
        #data_mean = data_mean.fillna(0)
        #data_std = data_std.fillna(0)
        #print(self.scaler.mean_.shape)

        # Normalize features
        data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        print("datamin",data.min(), data.max())
        data = (data-self.mean_map)/self.std_map
        data = np.where(np.isnan(data), 0, data)
        # print number of Nan values data
        #print(f"Number of NaN values in data: {np.isnan(data).sum()}") 
        # if Nan replace by 0
        #data = np.where(np.isnan(data), 0, data)

        #print(f"Number of NaN values in data: {np.isnan(data).sum()}") 
        # shapes = data.shape
        # data = data.reshape(data.shape[0], -1) # Flatten spatial dimensions
        # data = self.scaler.transform(data) # Normalize features
        # data = data.reshape(shapes[0], shapes[1], shapes[2]) # Reshape back
        # data = np.where(np.isnan(data), 0, data) 
        # add land sea mask feature
        # land_sea_mask = truth["land_sea_mask"].values.T
        # data = np.concatenate([data, land_sea_mask[np.newaxis, ...]], axis=0)
        features = torch.tensor(data, dtype=torch.float) # 67x120x240

        # compute wind 
        if self.target_variable == "10m_wind_speed":
            compute_wind_speedxr(truth,"obs")

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
        self.data_path = data_path
        self.obs_path = obs_path
        self.valid_years = np.arange(valid_years[0], valid_years[1] + 1)
        self.validation_year_begin = self.valid_years[len(self.valid_years)*9//10]
        self.subset = subset
        self.index_path = f"/home/majanvie/scratch/loader/{subset}_index.csv"
        self.scaler_path = f"/home/majanvie/scratch/loader/{subset}_scaler.joblib"
        self.scaler_truth_path = f"/home/majanvie/scratch/loader/{subset}_scaler_truth.joblib"
        self.trend_path = f"/home/majanvie/scratch/loader/{subset}_trend_temp.nc"
        self.trend_truth_path = f"/home/majanvie/scratch/loader/{subset}_trend_temp_truth.nc"

        # Open observations
        self.obs = xr.open_mfdataset(obs_path + '/*.nc', combine='by_coords')
        self.latitude = self.obs.latitude.values
        self.longitude = self.obs.longitude.values

        # build index of counts for files
        if os.path.exists(self.index_path):
            print("loading index")
            self.data_index = pd.read_csv(self.index_path).values.tolist()
        else:
            print("building index")
            self.build_index()
            self.save_index()

        print("len index", len(self.data_index))
        print(time.time()-dtime)

        # compute train trends
        if os.path.exists(self.trend_path):
            #data = xr.open_dataset(self.trend_path)
            self.trend_model = joblib.load(self.trend_path) 
        else:
            self.compute_trend_temp("train")
            self.save_trend("train")

        # compute truth trends
        if os.path.exists(self.trend_truth_path):
            #data = xr.open_dataset(self.trend_truth_path)
            self.trend_model_truth = joblib.load(self.trend_truth_path)
        else:
            self.compute_trend_temp("obs")
            self.save_trend("obs")

        # normalize training data
        if os.path.exists(self.scaler_path):
            self.load_scaler("train")
        else:
            self.fit_scaler("train")
            self.save_scaler("train")

        # normalize truth data 
        self.scaler_truth = StandardScaler()
        if os.path.exists(self.scaler_truth_path):
            self.load_scaler("obs")
            print("loading")
        else:
            self.fit_scaler("obs")
            self.save_scaler("obs")

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
            sample_count = 0 
            self.running_mean_map = None
            self.running_var_map = None
            for file_mean in files:
                data = xr.open_dataset(file_mean)
                if self.subset == "test":
                    data = data.rename({'time': 'forecast_time'})
                new_nb_samples = data["2m_temperature"].isel(longitude=0, latitude=0).values.flatten().shape[0] # delta 
                # detrend temperature data
                data["2m_temperature"] = data["2m_temperature"] - self.trend_model.predict(data["2m_temperature"].valid_time.values.astype(np.int64).reshape(-1,1))
                if sample_count == 0:
                    # compute using xarray methods
                    self.running_mean_map = data.mean(dim=["forecast_time", "prediction_timedelta"]).to_array().values
                    self.running_var_map = data.var(dim=["forecast_time", "prediction_timedelta"]).to_array().values
                    sample_count += new_nb_samples
                else:
                    # compute new mean and variance
                    new_data_mean = data.mean(dim=["forecast_time", "prediction_timedelta"]).to_array().values
                    new_data_var = data.var(dim=["forecast_time", "prediction_timedelta"]).to_array().values # Nan to zero to avoid warning 
                    
                    # incremental mean and variance
                    previous_mean_map = self.running_mean_map
                    previous_var_map = self.running_var_map
                    self.running_mean_map = (sample_count * self.running_mean_map + new_nb_samples * new_data_mean) / (sample_count + new_nb_samples)
                    self.running_var_map = (sample_count * (previous_var_map + previous_mean_map**2) + 
                                    new_nb_samples * (new_data_var + new_data_mean**2)) / (sample_count+ new_nb_samples) - self.running_mean_map**2          
                    sample_count += new_nb_samples
                
            # manage NaN, negative values etc 
            self.mean_map = np.where(np.isnan(self.running_mean_map), 0, self.running_mean_map)
            self.running_var_map = np.where(np.isnan(self.running_var_map), 1, self.running_var_map)
            self.var_map = np.where(self.running_var_map <= 0, 1, self.running_var_map)
            self.std_map = np.sqrt(self.var_map)

        elif type == "obs":
            valid_times = [x[3] for x in self.data_index]
            truth = self.obs.sel(time=valid_times)
            compute_wind_speedxr(truth,"obs")
            truth = truth[["2m_temperature", "10m_wind_speed"]]
            # detrend temperature data
            truth["2m_temperature"] = truth["2m_temperature"] - self.trend_model_truth.predict(truth["2m_temperature"].time.values.astype(np.int64).reshape(-1,1))
            self.mean_truth = truth.mean(dim="time").to_array().values.transpose(0,2,1)
            self.std_truth = truth.std(dim="time").to_array().values.transpose(0,2,1)


    def compute_trend_temp(self, type):
        if type == 'obs':
            valid_times = [x[3] for x in self.data_index]
            truth = self.obs["2m_temperature"].sel(time=valid_times).values
            times = truth.time.values
            num_times = np.array(pd.to_datetime(times).astype(np.int64))*1e-9 # in seconds for stability
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
            joblib.dump(self.trend_truth_model, self.trend_truth_path)
        elif type == "train":
            joblib.dump(self.trend_model, self.trend_path)

        
                
    # def fit_scaler(self, type):
    #     """Get all features and fit the scaler on them"""
    #     # Initialize lists to store partial means and variances
    #     feature_count = 0
    #     partial_means = None
    #     partial_variances = None
    #     i = 0

    #     for file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time, lead_time_idx in self.data_index:
    #         if type == "train":
    #             data_mean = xr.open_dataset(file_mean)
    #             if self.subset == "test":
    #                 data_mean = data_mean.rename({'time': 'forecast_time'})

    #             data_mean = data_mean.isel(prediction_timedelta=lead_time_idx, forecast_time=forecast_idx_in_file)
    #             #data_mean = data_mean.fillna(0)  # manage NaN
    #             data_mean = data_mean.to_array().values  # (n_vars, n_lat, n_lon)

    #         elif type == "obs":
    #             data_mean = self.obs.sel(time=valid_time)
    #             compute_wind_speedxr(data_mean,"obs") # compute wind 
    #             data_mean = data_mean[["2m_temperature", "10m_wind_speed"]].to_array().values.transpose(0,2,1) # (2, n_lat, n_lon)

    #         # Flatten spatial dimensions
    #         #data_mean = data_mean.reshape(data_mean.shape[0], -1)
    #         print("data_mean",data_mean.shape)

    #         # Incrementally compute mean and variance
    #         feature_count += data_mean.shape[0]
    #         if partial_means is None:
    #             partial_means = [data_mean]#np.mean(data_mean, axis=0)
    #             partial_variances = np.var(data_mean, axis=0)
    #         else:
    #             new_mean = np.mean(data_mean, axis=0)
    #             new_variance = np.var(data_mean, axis=0)
                
    #             # Update means and variances
    #             partial_means = (partial_means * (feature_count - data_mean.shape[0]) + new_mean * data_mean.shape[0]) / feature_count
    #             partial_variances = ((partial_variances * (feature_count - data_mean.shape[0]) + new_variance * data_mean.shape[0])
    #                                  / feature_count + 
    #                                  (partial_means - new_mean)**2 * (feature_count - data_mean.shape[0]) * data_mean.shape[0] / feature_count**2)
    #         i+=1
    #         if i ==10:
    #             break

    #     # Set scaler parameters
    #     if type == "train":
    #         self.scaler.mean_ = partial_means
    #         self.scaler.var_ = partial_variances
    #         self.scaler.scale_ = np.sqrt(partial_variances)
    #         self.scaler.n_samples_seen_ = feature_count
    #     elif type =="obs":
    #         self.scaler_truth.mean_ = partial_means
    #         self.scaler_truth.var_ = partial_variances
    #         self.scaler_truth.scale_ = np.sqrt(partial_variances)
    #         self.scaler_truth.n_samples_seen_ = feature_count
    

    def save_scaler(self, type):
        """Save the fitted scaler to a file."""
        if type == "train":
            ds = xr.Dataset(
                        data_vars=dict(
                            mean=(["latitude", "longitude"], self.mean_map),
                            std=(["latitude", "longitude"], self.std_map),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                        )
                    )
            ds.to_netcdf(self.scaler_path)
        elif type == "obs":
            ds = xr.Dataset(
                        data_vars=dict(
                            mean=(["latitude", "longitude"], self.mean_truth),
                            std=(["latitude", "longitude"], self.std_truth),
                        ),
                        coords=dict(
                            longitude=("longitude", self.longitude), # 1D array
                            latitude=("latitude", self.latitude), # 1D array
                        )
                    )
            ds.to_netcdf(self.scaler_truth_path)
    
    def load_scaler(self, type):
        """Load the scaler from a file."""
        if type == "train":
            ds = xr.open_dataset(self.scaler_path)
            self.mean_map = ds["mean"].values
            self.std_map = ds["std"].values
        elif type == "obs":
            ds = xr.open_dataset(self.scaler_truth_path)
            self.mean_truth = ds["mean"].values
            self.std_truth = ds["std"].values

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_mean, file_std, forecast_idx_in_file, valid_time, forecast_time, lead_time_idx = self.data_index[idx]
        data_mean = xr.open_dataset(file_mean)
        data_std = xr.open_dataset(file_std)
        valid_time = pd.to_datetime(valid_time)
        forecast_time = pd.to_datetime(forecast_time)
        # select forecast data
        if self.subset == "test":
            # rename time to forecast_time
            data_mean = data_mean.rename({'time': 'forecast_time'})
            data_std = data_std.rename({'time': 'forecast_time'})
        data_mean = data_mean.isel(prediction_timedelta=lead_time_idx, forecast_time=forecast_idx_in_file)
        data_std = data_std.isel(prediction_timedelta=lead_time_idx, forecast_time=forecast_idx_in_file)
        truth = self.obs.sel(time=valid_time)

        # Manage Nan
        #data_mean = data_mean.fillna(0)
        #data_std = data_std.fillna(0)

        # Normalize ground truth 
        # compute_wind_speedxr(truth,"obs") # compute wind 
        # truth = truth[["2m_temperature", "10m_wind_speed"]].to_array().values
        # truth = torch.tensor(truth.transpose(0,2,1), dtype=torch.float) # (2, n_lat, n_lon)
        # tshapes = truth.shape
        # truth = truth.reshape(truth.shape[0], -1) # Flatten spatial dimensions
        # truth = self.scaler_truth.transform(truth) # Normalize truth
        # truth = truth.reshape(tshapes[0], tshapes[1], tshapes[2]) # Reshape back
        # print(truth.shape)
        # print(truth)

        compute_wind_speedxr(truth,"obs")
        # detrend temperature data
        truth["2m_temperature"] = truth["2m_temperature"] - self.trend_model_truth.predict(truth["2m_temperature"].time.values.astype(np.int64).reshape(-1,1))
        truth = truth[["2m_temperature", "10m_wind_speed"]].to_array().values.transpose(0,2,1) # (2, n_lat, n_lon)
        truth = (truth - self.mean_truth)/self.std_truth
        truth = torch.tensor(truth[:,1:,:], dtype=torch.float) # 2x120x240


        # Normalize features
        # data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        # shapes = data.shape
        # print(shapes)
        # print(self.scaler.mean_.shape, self.scaler.scale_.shape)
        # data = data.reshape(data.shape[0], -1) # Flatten spatial dimensions
        # data = self.scaler.transform(data) # Normalize features
        # data = data.reshape(shapes[0], shapes[1], shapes[2]) # Reshape back

        # detrend temperature data
        data_mean["2m_temperature"] = data_mean["2m_temperature"] - self.trend_model.predict(data_mean["2m_temperature"].valid_time.values.astype(np.int64).reshape(-1,1))
        # normalize data 
        data = data_mean.to_array().values # (n_vars, n_lat, n_lon)
        shapes = data.shape
        data = (data-self.mean_map)/self.std_map
        data = np.where(np.isnan(data), 0, data)
        # add land sea mask feature
        land_sea_mask = truth["land_sea_mask"].values.T
        data = np.concatenate([data, land_sea_mask[np.newaxis, ...]], axis=0)
        
        # add lead time and day of year as features
        lead_time = np.ones((shapes[1], shapes[2])) * (lead_time_idx/7)
        year = valid_time.year
        day_of_year = valid_time.timetuple().tm_yday / 366
        sin_day_of_year = np.ones((shapes[1], shapes[2])) * np.sin(2*np.pi*day_of_year)
        cos_day_of_year = np.ones((shapes[1], shapes[2])) * np.cos(2*np.pi*day_of_year)
        data = np.concatenate([data,
            lead_time[np.newaxis, ...],
            sin_day_of_year[np.newaxis, ...],
            cos_day_of_year[np.newaxis, ...]], axis=0)
        features = torch.tensor(data[:,1:,:], dtype=torch.float) #70x120x240

        # mean std over ensemble predictions for the target variable
        #mu = torch.tensor(data_mean[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        #sigma = torch.tensor(data_std[self.target_variable].values, dtype=torch.float) # (n_lat, n_lon)
        valid_time = str(valid_time)
        forecast_time = str(forecast_time)
        return {
            'input': features, 'truth': truth, "day_of_year": valid_time.timetuple().tm_yday,
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
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    test_folder = f"{data_folder}/test/EMOS"
    obs_folder = "/home/majanvie/scratch/data/raw/obs"

    dtime = time.time()

    train_dataset = WeatherEnsembleDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable="10m_wind_speed",
        lead_time_idx=14,
        valid_years=[1996,1997],
        valid_months=[1,1],
        subset="train")

    print("dataset time", time.time()-dtime)
    

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("Nb of training examples:",len(train_loader.dataset.data_index))
    dtime = time.time()
    for batch in train_loader:
        print("SHAPES")
        print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
        print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
        print(" ")
        print(batch["input"])
        print(batch["input"][0][0].mean())
        print(batch["input"].min(), batch["input"].max(), batch["input"].mean())
        break
    print("getitem",time.time()-dtime)

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
    # for batch in train_loader:
    #     print("SHAPES")
    #     print(batch['input'].shape, batch['truth'].shape)
    #     print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
    #     print(" ")

    #     print("VALUES")
    #     print(batch["input"][0][0].mean())
    #     print(batch["truth"])
    #     break

    # # test 
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
    #     break

    # # val
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
