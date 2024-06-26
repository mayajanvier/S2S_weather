import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .format_data import compute_wind_speedxr, compute_wind_speed, fit_norm_along_axis, format_ensemble_data_EMOS
import xarray as xr
import pathlib
import re
from datetime import datetime


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
        sigma = torch.tensor(data_std[self.target_variable], dtype=torch.float) # (n_lat, n_lon)
        valid_time = str(valid_time)
        forecast_time = str(forecast_time)
        # get associated ground truth
        truth = torch.tensor(truth[self.target_variable].values.T, dtype=torch.float) # (n_lat, n_lon)
        return {
            'mu': mu, 'sigma': sigma, 'input': features, 'truth': truth,
            'valid_time': valid_time, "lead_time": self.lead_time_idx, "forecast_time": forecast_time
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

    ### WeatherDatasets
    data_folder = "/home/majanvie/scratch/data/raw"
    train_folder = f"{data_folder}/train"
    test_folder = f"{data_folder}/test"
    obs_folder = f"{data_folder}/obs"

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
    #     for month in range(1,13):
    #         print("Month", month)
    variable = "2m_temperature"
    month=1
    train_dataset = WeatherDataset(
        data_path=train_folder,
        obs_path=obs_folder,
        target_variable=variable,
        lead_time_idx=28,
        valid_years=[1996,2017],
        valid_months=[month,month],
        subset="train")
            #break
        #break

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print("Nb of training examples:",len(train_loader.dataset.data_index))
    for batch in train_loader:
        print("SHAPES")
        print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
        print(batch['forecast_time'], batch['valid_time'], batch['lead_time'])
        print(" ")
        break

    # val_dataset = WeatherDataset(
    # data_path=train_folder,
    # obs_path=obs_folder,
    # target_variable="10m_wind_speed",
    # lead_time_idx=28,
    # valid_years=[1996,2017],
    # valid_months=[1,1],
    # subset="val")
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    # print("Nb of training examples:",len(val_loader.dataset.data_index))
    # for batch in val_loader:
    #     print("SHAPES")
    #     print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
    #     print(" ")
    #     break

    # print(val_dataset.data_index[0])
    # print(train_dataset.data_index[0])


