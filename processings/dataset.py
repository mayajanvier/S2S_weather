import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from format_data import get_lead_shape, get_year, get_month, compute_wind_speed
import xarray as xr
import pathlib

VALIDATION_DAY_BEGIN = 1 # a voir avec David

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

class WeatherTestDataset:
    def __init__(
        self,
        data_path,
        obs_path,
        target_variable,
        lead_time_idx,
        valid_years,
        valid_months
    ):
        self.lead_time_idx = lead_time_idx
        self.obs2forecast_var = {
            "2m_temperature": "temperature", 
            "10m_u_component_of_wind": "u_component_of_wind", 
            "10m_v_component_of_wind": "v_component_of_wind"}
        self.target_variable = target_variable
        self.forecast_variable = self.obs2forecast_var[self.target_variable]
        self.valid_months = np.arange(valid_months[0], valid_months[1] + 1)
        self.valid_years = np.arange(valid_years[0], valid_years[1] + 1)
        print(self.valid_years, self.valid_months)

        # Open observations
        self.obs = xr.open_mfdataset(obs_path + '/*.nc', combine='by_coords')

        # Build index of counts for test files.
        data_path = pathlib.Path(data_path)
        files = sorted(list(data_path.glob("*.nc"))) 

        #logger.info("Building index for test dataset...") # who is logger 

        self.data_index = []
        # only one "loop" for test files
        for f in files:
            data = xr.open_dataset(f)
            dataset, size = get_lead_shape(data, self.obs, self.lead_time_idx, self.valid_years, self.valid_months)
            if size == 0: # no data within time ranges and month
                continue

            # Iterate over each time and prediction_timedelta
            for time_idx in range(dataset.sizes["time"]):
                valid_time = dataset.time.values[time_idx] # lead time already selected
                # keep only within valid years and month
                if get_year(valid_time) not in self.valid_years:
                    continue
                elif get_month(valid_time) not in self.valid_months:   
                    continue
                self.data_index.append(
                    (f, time_idx, pd.to_datetime(valid_time))
                )

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file, forecast_idx_in_file, valid_time = self.data_index[idx]
        # select forecast data
        dataset = xr.open_dataset(file).isel(prediction_timedelta=self.lead_time_idx, time=forecast_idx_in_file)
        # compute wind 
        if self.target_variable == "10m_wind_speed":
            ds['10m_wind_speed'] = (ds['u_component_of_wind']**2 + ds['v_component_of_wind']**2)**0.5
            ds['10m_wind_speed'].name = '10_wind_speed'

        # predicted values for the target variable at 1000 hPa
        mu = torch.tensor(dataset.sel(level=1000)[self.forecast_variable].values, dtype=torch.float) # (n_lat, n_lon)
        sigma = torch.tensor(0, dtype=torch.float)
        # get features at right format
        data = dataset.to_array().values #  (n_vars, n_levels, n_lat, n_lon)
        data = data.reshape(-1, data.shape[2], data.shape[3]) # (n_levels*n_vars, n_lat, n_lon)
        features = torch.tensor(data, dtype=torch.float)
        valid_time = str(valid_time)
        # get associated ground truth
        truth = torch.tensor(self.obs.sel(time=valid_time)[self.target_variable].values.T, dtype=torch.float) # (n_lat, n_lon)
        return {'mu': mu, 'sigma': sigma, 'input': features, 'truth': truth, 'valid_time': valid_time}






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
    test_folder = f"{data_folder}/test"
    obs_folder = f"{data_folder}/obs"
    test_dataset = WeatherTestDataset(
        data_path=test_folder,
        obs_path=obs_folder,
        target_variable="2m_temperature",
        lead_time_idx=28,
        valid_years=[2018,2022],
        valid_months=[1,3])
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print("Nb of training examples:",len(test_loader.dataset.data_index))
    for batch in test_loader:
        print("SHAPES")
        print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
        print(" ")
        break
