import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler

class PandasDataset(Dataset):
    def __init__(self, dataframe, target_column):
        self.dataframe = dataframe
        self.target_column = target_column
        self.mu = dataframe[f"mu_{target_column}"].values # scalar
        self.sigma = dataframe[f"sigma_{target_column}"].values # scalar
        self.truth = dataframe["truth"].values # scalar
        self.input = dataframe["input"] # numpy arrays

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
        
        return {'mu': mu, 'sigma': sigma, 'input': input_data, 'truth': truth}


if __name__== "__main__":
    data_folder = "../scratch/"
    train_data = pd.read_json(data_folder+'data_2m_temperature.json')
    feature_dim = len(train_data["input"][0]) #.shape[0] 
    print(feature_dim)
    train_data = PandasDataset(train_data, "2m_temperature")
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    for batch in train_loader:
        print("SHAPES")
        print(batch['mu'].shape, batch['sigma'].shape, batch['input'].shape, batch['truth'].shape)
        print(" ")

        print("VALUES")
        print(batch['mu'], batch['sigma'], batch['input'], batch['truth'])
        break