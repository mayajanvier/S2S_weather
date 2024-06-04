import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

if __name__== "__main__":
    data_folder = "../scratch/data/train/"
    train_data = pd.read_json(data_folder+"PPE_OPT_lat=-90.0_lon=0.0_lead=24h.json")
    train_data = compute_wind_speed(train_data)
    print(train_data.shape)

    # separate train and validation randomly
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True)

    train_data = PandasDataset(train_data, "10m_wind_speed")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

    val_data = PandasDataset(val_data, target_column="10m_wind_speed")
    val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

    for batch in train_loader:
        print("SHAPES")
        print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
        print(" ")

        print("VALUES")
        print(batch['mu'], batch['sigma'], batch['input'], batch['truth'], batch['forecast_time'])
        break

    for batch in val_loader:
        print("SHAPES")
        print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
        print(" ")

        print("VALUES")
        print(batch['mu'], batch['sigma'], batch['input'], batch['truth'], batch['forecast_time'])
        break