import pandas as pd 
import numpy as np
import xarray as xr
from metrics import crps_normal
import torch
import json
from processings.dataset import WeatherYearEnsembleDataset, WeatherEnsembleDatasetMMdetrend, WeatherYearEnsembleDatasetNorm
from torch.utils.data import DataLoader
from model import SpatialEMOS, DRUnet, DRUnetPrior, DRUnetPriorVar, DRUnetVar
from torch.distributions import Normal
from processings.format_data import compute_wind_speedxr

### MODEL INFERENCES ###

### EMOS 
def specialSpatialEMOS_inference_detrend(lead_time, valid_years, train_years, name, train_index, epoch, var): # detrend MM 
    data_folder = "/home/majanvie/scratch/data" 
    train_folder = f"{data_folder}/train/EMOS"
    test_folder = f"{data_folder}/test/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    climato_folder = f"{obs_folder}/climato"

    base_dir = f"training_results/spatial_month_ensemble_detrend_temp/lead{lead_time}"
    full_results = []   
    train_index = 1
    for month in range(1,13):
        print(f"Month {month}")
        for variable in ["2m_temperature"]: #, "10m_wind_speed"]:
            if variable == "2m_temperature":
                epoch = 9
            elif variable == "10m_wind_speed":
                epoch = 14
            print(f"Variable {variable}")
            model_folder = f"{base_dir}/training_{train_index}_spatial_month{month}_{variable}_lead={lead_time}"
            climato_path = f"{climato_folder}/{variable}_{train_years[0]}_{train_years[-1]}_month{month}_lead{lead_time}.nc"
            climato = xr.open_dataset(climato_path)

            test_dataset = WeatherEnsembleDatasetMMdetrend(
                data_path=test_folder,
                obs_path=obs_folder,
                target_variable=variable,
                lead_time_idx=lead_time,
                valid_years=valid_years,
                valid_months=[month,month],
                subset="test")

            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            # load model weights
            model = SpatialEMOS(67, 121, 240, 4)
            model.load_state_dict(torch.load(
                f"{model_folder}/model_{epoch}.pth"
                ))
            model.eval()

            # Compute performance metrics
            results = []
            for batch in test_loader:
                mu = batch['mu']
                sigma = batch['sigma']
                X = batch['input']
                y = batch['truth']

                valid_time = batch['valid_time']
                valid_date = pd.to_datetime(valid_time).strftime('%m-%d')
                mu_clim = torch.tensor(climato["mu"].sel(date=valid_date).values[0])
                sigma_clim = torch.tensor(climato["sigma"].sel(date=valid_date).values[0])

                out_distrib = model(mu, sigma, X, y)
                climato_distrib = Normal(mu_clim, sigma_clim)


                # detrend temperature
                # in test dataset, y truth is already trended
                if variable == "2m_temperature":
                    Xt = pd.to_datetime(valid_time).astype(np.int64)[0] * 1e-9
                    shape = y.shape
                    trend = test_dataset.trend_model_truth.predict(Xt.reshape(-1,1)).reshape(shape[2],shape[1]).T
                    out_distrib.loc += torch.tensor(trend, dtype=out_distrib.loc.dtype)

                print("pred", out_distrib.loc.mean())
                print("diff", (y-out_distrib.loc).mean())
                crps_var = crps_normal(out_distrib,y).detach().numpy()  # shape (1, lat, lon)
                crps_climato = crps_normal(climato_distrib,y).detach().numpy() # shape (1, lat, lon)
  
                forecast_time = batch['forecast_time'][0]
                # Create a multi-index for the time dimension
                time_index = pd.MultiIndex.from_tuples([(lead_time, forecast_time)], names=["lead_time", "forecast_time"])
                
                ds = xr.Dataset(
                    data_vars=dict(
                        crps_temperature=(["time", "latitude", "longitude"], crps_var),
                        crps_temperature_climato=(["time", "latitude", "longitude"], crps_climato),
                    ),
                    coords=dict(
                        longitude=("longitude", test_loader.dataset.longitude), # 1D array
                        latitude=("latitude", test_loader.dataset.latitude), # 1D array
                        time=("time", time_index), # 2D array
                    )
                )

                # Reset the index to convert MultiIndex into separate variables
                ds = ds.reset_index('time')
                results.append(ds)
                #full_results.append(ds)
            train_index += 1
            final_ds = xr.concat(results, dim='time')

            # Write to NetCDF file
            results_file_path = f"{model_folder}/crps_{epoch}.nc"
            final_ds.to_netcdf(results_file_path)

### DRUNets
# DRUnet both [Month Lead Agg]
def DRUnet_inference(valid_years, train_years, train_index, epoch, device):
    data_folder = "/home/majanvie/scratch/data" 
    test_folder = f"{data_folder}/test/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    climato_folder = f"{obs_folder}/climato"

    base_dir = f"training_results/DRUnet"

    model_folder = f"{base_dir}/training_{train_index}_DRUnet"
    climato_path = f"{climato_folder}/{train_years[0]}_{train_years[-1]}.nc"
    climato = xr.open_dataset(climato_path)
    # directement recuperer la ground truth dans le fichier 
    era = xr.open_mfdataset(f"{obs_folder}/*.nc", combine='by_coords')
    compute_wind_speedxr(era, "obs")

    test_dataset = WeatherYearEnsembleDataset(
        data_path=test_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="test")

    print("test dataset", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=16)
    lat_beg = test_dataset.latitude_beg

    # load model weights
    model = DRUnet(74,4).to(device)
    model.load_state_dict(torch.load(
        f"{model_folder}/model_{epoch}.pth"
        ))
    model.eval()

    # Compute performance metrics
    results = []
    i = 0
    for batch in test_loader:
        X = batch['input'].to(device)
        y = batch['truth'].to(device)
        lead_time = batch['lead_time'].item()

        # climatology
        valid_time = batch['valid_time']
        day_of_year = batch["day_of_year"]
        mu_clim_temp = torch.tensor(climato["2m_temperature_mean"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 
        sigma_clim_temp = torch.tensor(climato["2m_temperature_std"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 
        mu_clim_wind = torch.tensor(climato["10m_wind_speed_mean"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:]
        sigma_clim_wind = torch.tensor(climato["10m_wind_speed_std"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 

        # model inference
        maps, out_distrib_temp, out_distrib_wind = model(X)
        truth = era.sel(time=valid_time) 
        #print(truth["2m_temperature"].values.transpose(0,2,1).shape)
        y_temp = torch.Tensor(truth["2m_temperature"].values.transpose(0,2,1))[:,lat_beg:,:].to(device)
        y_wind = torch.Tensor(truth["10m_wind_speed"].values.transpose(0,2,1))[:,lat_beg:,:].to(device)


        # add trend temperature (trend y)
        Xt = pd.to_datetime(valid_time).astype(np.int64)[0] * 1e-9
        trend = test_dataset.trend_model_truth.predict(Xt.reshape(-1,1)).reshape(1, 240, 121).transpose(0,2,1)[:,lat_beg:,:]
        out_distrib_temp.loc += torch.tensor(trend, dtype=out_distrib_temp.loc.dtype).to(device)

        climato_distrib_temp = Normal(mu_clim_temp.to(device), sigma_clim_temp.to(device))
        climato_distrib_wind = Normal(mu_clim_wind.to(device), sigma_clim_wind.to(device))

        crps_temp = crps_normal(out_distrib_temp,y_temp).cpu().detach().numpy()  # shape (1, lat, lon)
        crps_wind = crps_normal(out_distrib_wind,y_wind).cpu().detach().numpy()
        crps_climato_temp = crps_normal(climato_distrib_temp,y_temp).cpu().detach().numpy() # shape (1, lat, lon)
        crps_climato_wind = crps_normal(climato_distrib_wind,y_wind).cpu().detach().numpy()


        forecast_time = batch['forecast_time'][0]
        # Create a multi-index for the time dimension
        time_index = pd.MultiIndex.from_tuples([(lead_time, forecast_time)], names=["lead_time", "forecast_time"])

        ds = xr.Dataset(
            data_vars=dict(
                crps_temperature=(["time", "latitude", "longitude"], crps_temp),
                crps_wind_speed=(["time", "latitude", "longitude"], crps_wind),
                crps_temperature_climato=(["time", "latitude", "longitude"], crps_climato_temp),
                crps_wind_speed_climato=(["time", "latitude", "longitude"], crps_climato_wind),
            ),
            coords=dict(
                longitude=("longitude", test_loader.dataset.longitude), # 1D array
                latitude=("latitude", test_loader.dataset.latitude), # 1D array
                time=("time", time_index), # 2D array
            )
        )
        
        if i%100 == 0:
            print(i, crps_temp.mean(), crps_wind.mean(), crps_climato_temp.mean(), crps_climato_wind.mean())
        i+=1
        
        # Reset the index to convert MultiIndex into separate variables
        ds = ds.reset_index('time')
        results.append(ds)

    final_ds = xr.concat(results, dim='time')

    # Write to NetCDF file
    results_file_path = f"{model_folder}/crps_{epoch}.nc"
    final_ds.to_netcdf(results_file_path)

# DRUNet single [General Agg]
def DRUnet_inference_Norm_Var(valid_years, train_years, train_index, epoch, device, variable=None):
    data_folder = "/home/majanvie/scratch/data" 
    test_folder = f"{data_folder}/test/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    climato_folder = f"{obs_folder}/climato"

    base_dir = f"training_results/DRUnet"

    if variable == None:
        model_folder = f"{base_dir}/training_{train_index}_DRUnet_norm"
    else:
        model_folder = f"{base_dir}/training_{train_index}_DRUnet_norm_{variable}"
    climato_path = f"{climato_folder}/{train_years[0]}_{train_years[-1]}.nc"
    climato = xr.open_dataset(climato_path)
    # directement recuperer la ground truth dans le fichier 
    era = xr.open_mfdataset(f"{obs_folder}/*.nc", combine='by_coords')
    compute_wind_speedxr(era, "obs")

    test_dataset = WeatherYearEnsembleDatasetNorm(
        data_path=test_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="test")

    print("test dataset", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=16)
    lat_beg = test_dataset.latitude_beg

    # load model weights
    if variable == None:
        model = DRUnet(74,4).to(device)
    else:
        # TODO 77,2 
        model = DRUnetVar(77,4,variable).to(device)
    model.load_state_dict(torch.load(
        f"{model_folder}/model_{epoch}.pth"
        ))
    model.eval()

    # Compute performance metrics
    results = []
    i = 0
    for batch in test_loader:
        X = batch['input'].to(device)
        y = batch['truth'].to(device)
        #mu = batch['mu'].to(device)
        #sigma = batch['sigma'].to(device)
        lead_time = batch['lead_time'].item()

        # climatology
        valid_time = batch['valid_time']
        day_of_year = batch["day_of_year"]
        #valid_date = pd.to_datetime(valid_time).strftime('%m-%d')
        # mu_clim_temp = torch.tensor(climato["2m_temperature_mean"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 
        # sigma_clim_temp = torch.tensor(climato["2m_temperature_std"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 
        # mu_clim_wind = torch.tensor(climato["10m_wind_speed_mean"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:]
        # sigma_clim_wind = torch.tensor(climato["10m_wind_speed_std"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 
        mu_clim = torch.tensor(climato[variable+"_mean"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:]
        sigma_clim = torch.tensor(climato[variable+"_std"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:]

        # model inference
        maps, out_distrib = model(X)

        #y_temp = y[:,0,:,:]
        #y_wind = y[:,1,:,:]
        truth = era.sel(time=valid_time) 
        #print(truth["2m_temperature"].values.transpose(0,2,1).shape)
        #y_temp = torch.Tensor(truth["2m_temperature"].values.transpose(0,2,1))[:,lat_beg:,:].to(device)
        #y_wind = torch.Tensor(truth["10m_wind_speed"].values.transpose(0,2,1))[:,lat_beg:,:].to(device)
        y = torch.Tensor(truth[variable].values.transpose(0,2,1))[:,lat_beg:,:].to(device)


        # add trend temperature (trend y)
        if variable == "2m_temperature":
            Xt = pd.to_datetime(valid_time).astype(np.int64)[0] * 1e-9
            trend = test_dataset.trend_model_truth.predict(Xt.reshape(-1,1)).reshape(1, 240, 121).transpose(0,2,1)[:,lat_beg:,:]
            out_distrib_temp.loc += torch.tensor(trend, dtype=out_distrib_temp.loc.dtype).to(device)

        # climato_distrib_temp = Normal(mu_clim_temp.to(device), sigma_clim_temp.to(device))
        # climato_distrib_wind = Normal(mu_clim_wind.to(device), sigma_clim_wind.to(device))
        climato_distrib = Normal(mu_clim.to(device), sigma_clim.to(device))

        # crps_temp = crps_normal(out_distrib_temp,y_temp).cpu().detach().numpy()  # shape (1, lat, lon)
        # crps_wind = crps_normal(out_distrib_wind,y_wind).cpu().detach().numpy()
        crps = crps_normal(out_distrib,y).cpu().detach().numpy()
        # crps_climato_temp = crps_normal(climato_distrib_temp,y_temp).cpu().detach().numpy() # shape (1, lat, lon)
        # crps_climato_wind = crps_normal(climato_distrib_wind,y_wind).cpu().detach().numpy()
        crps_climato = crps_normal(climato_distrib,y).cpu().detach().numpy()


        forecast_time = batch['forecast_time'][0]
        # Create a multi-index for the time dimension
        time_index = pd.MultiIndex.from_tuples([(lead_time, forecast_time)], names=["lead_time", "forecast_time"])

        # ds = xr.Dataset(
        #     data_vars=dict(
        #         crps_temperature=(["time", "latitude", "longitude"], crps_temp),
        #         crps_wind_speed=(["time", "latitude", "longitude"], crps_wind),
        #         crps_temperature_climato=(["time", "latitude", "longitude"], crps_climato_temp),
        #         crps_wind_speed_climato=(["time", "latitude", "longitude"], crps_climato_wind),
        #     ),
        #     coords=dict(
        #         longitude=("longitude", test_loader.dataset.longitude), # 1D array
        #         latitude=("latitude", test_loader.dataset.latitude), # 1D array
        #         time=("time", time_index), # 2D array
        #         #lead_time=("lead_time", [lead_time]),
        #         #forecast_time=("forecast_time", [forecast_time])
        #     )
        # )
        if variable == "2m_temperature":
            ds = xr.Dataset(
                data_vars=dict(
                    crps_temperature=(["time", "latitude", "longitude"], crps),
                    crps_temperature_climato=(["time", "latitude", "longitude"], crps_climato),
                ),
                coords=dict(
                    longitude=("longitude", test_loader.dataset.longitude), # 1D array
                    latitude=("latitude", test_loader.dataset.latitude), # 1D array
                    time=("time", time_index), # 2D array

            ))
        elif variable == "10m_wind_speed":
            ds = xr.Dataset(
                data_vars=dict(
                    crps_wind_speed=(["time", "latitude", "longitude"], crps),
                    crps_wind_speed_climato=(["time", "latitude", "longitude"], crps_climato),
                ),
                coords=dict(
                    longitude=("longitude", test_loader.dataset.longitude), # 1D array
                    latitude=("latitude", test_loader.dataset.latitude), # 1D array
                    time=("time", time_index), # 2D array

            ))

        
        if i%100 == 0:
            print(i, crps.mean(), crps_climato.mean())
        i+=1
        
        # Reset the index to convert MultiIndex into separate variables
        ds = ds.reset_index('time')
        results.append(ds)

    final_ds = xr.concat(results, dim='time')

    # Write to NetCDF file
    results_file_path = f"{model_folder}/crps_{epoch}.nc"
    final_ds.to_netcdf(results_file_path)

# DRUnet+prior both [Month Lead Agg]
def DRUnet_inference_Prior( valid_years, train_years, train_index, epoch, device, variable=None):
    data_folder = "/home/majanvie/scratch/data" 
    test_folder = f"{data_folder}/test/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    climato_folder = f"{obs_folder}/climato"

    base_dir = f"training_results/DRUnet"
    if variable is None:
        model_folder = f"{base_dir}/training_{train_index}_DRUnet_prior"
    else:
        model_folder = f"{base_dir}/training_{train_index}_DRUnet_prior_{variable}"
    climato_path = f"{climato_folder}/{train_years[0]}_{train_years[-1]}.nc"
    climato = xr.open_dataset(climato_path)
    # directement recuperer la ground truth dans le fichier 
    era = xr.open_mfdataset(f"{obs_folder}/*.nc", combine='by_coords')
    compute_wind_speedxr(era, "obs")

    test_dataset = WeatherYearEnsembleDataset(
        data_path=test_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="test")

    print("test dataset", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=16)
    lat_beg = test_dataset.latitude_beg

    # load model weights
    model = DRUnetPrior(74,4).to(device)
    model.load_state_dict(torch.load(
        f"{model_folder}/model_{epoch}.pth"
        ))
    model.eval()

    # Compute performance metrics
    results = []
    i = 0
    for batch in test_loader:
        X = batch['input'].to(device)
        y = batch['truth'].to(device)
        mu = batch['mu'].to(device)
        sigma = batch['sigma'].to(device)
        lead_time = batch['lead_time'].item()

        # climatology
        valid_time = batch['valid_time']
        day_of_year = batch["day_of_year"]
        mu_clim_temp = torch.tensor(climato["2m_temperature_mean"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 
        sigma_clim_temp = torch.tensor(climato["2m_temperature_std"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 
        mu_clim_wind = torch.tensor(climato["10m_wind_speed_mean"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:]
        sigma_clim_wind = torch.tensor(climato["10m_wind_speed_std"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 

        # model inference
        maps, out_distrib_temp, out_distrib_wind = model(X, mu, sigma)

        #y_temp = y[:,0,:,:]
        #y_wind = y[:,1,:,:]
        truth = era.sel(time=valid_time) 
        #print(truth["2m_temperature"].values.transpose(0,2,1).shape)
        y_temp = torch.Tensor(truth["2m_temperature"].values.transpose(0,2,1))[:,lat_beg:,:].to(device)
        y_wind = torch.Tensor(truth["10m_wind_speed"].values.transpose(0,2,1))[:,lat_beg:,:].to(device)


        # add trend temperature (trend y)
        Xt = pd.to_datetime(valid_time).astype(np.int64)[0] * 1e-9
        trend = test_dataset.trend_model_truth.predict(Xt.reshape(-1,1)).reshape(1, 240, 121).transpose(0,2,1)[:,lat_beg:,:]
        out_distrib_temp.loc += torch.tensor(trend, dtype=out_distrib_temp.loc.dtype).to(device)

        climato_distrib_temp = Normal(mu_clim_temp.to(device), sigma_clim_temp.to(device))
        climato_distrib_wind = Normal(mu_clim_wind.to(device), sigma_clim_wind.to(device))

        crps_temp = crps_normal(out_distrib_temp,y_temp).cpu().detach().numpy()  # shape (1, lat, lon)
        crps_wind = crps_normal(out_distrib_wind,y_wind).cpu().detach().numpy()
        crps_climato_temp = crps_normal(climato_distrib_temp,y_temp).cpu().detach().numpy() # shape (1, lat, lon)
        crps_climato_wind = crps_normal(climato_distrib_wind,y_wind).cpu().detach().numpy()


        forecast_time = batch['forecast_time'][0]
        # Create a multi-index for the time dimension
        time_index = pd.MultiIndex.from_tuples([(lead_time, forecast_time)], names=["lead_time", "forecast_time"])

        ds = xr.Dataset(
            data_vars=dict(
                crps_temperature=(["time", "latitude", "longitude"], crps_temp),
                crps_wind_speed=(["time", "latitude", "longitude"], crps_wind),
                crps_temperature_climato=(["time", "latitude", "longitude"], crps_climato_temp),
                crps_wind_speed_climato=(["time", "latitude", "longitude"], crps_climato_wind),
            ),
            coords=dict(
                longitude=("longitude", test_loader.dataset.longitude), # 1D array
                latitude=("latitude", test_loader.dataset.latitude), # 1D array
                time=("time", time_index), # 2D array
            )
        )
        
        if i%100 == 0:
            print(i, crps_temp.mean(), crps_wind.mean(), crps_climato_temp.mean(), crps_climato_wind.mean())
        i+=1
        
        # Reset the index to convert MultiIndex into separate variables
        ds = ds.reset_index('time')
        results.append(ds)

    final_ds = xr.concat(results, dim='time')

    # Write to NetCDF file
    results_file_path = f"{model_folder}/crps_{epoch}.nc"
    final_ds.to_netcdf(results_file_path)

# DRUnet+prior single [Month Lead Agg]
def DRUnet_inference_Prior_Var(valid_years, train_years, train_index, epoch, device, variable=None):
    data_folder = "/home/majanvie/scratch/data" 
    test_folder = f"{data_folder}/test/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    climato_folder = f"{obs_folder}/climato"

    base_dir = f"training_results/DRUnet"
    if variable is None:
        model_folder = f"{base_dir}/training_{train_index}_DRUnet_prior"
    else:
        model_folder = f"{base_dir}/training_{train_index}_DRUnet_prior_{variable}"
    climato_path = f"{climato_folder}/{train_years[0]}_{train_years[-1]}.nc"
    climato = xr.open_dataset(climato_path)
    # directement recuperer la ground truth dans le fichier 
    era = xr.open_mfdataset(f"{obs_folder}/*.nc", combine='by_coords')
    compute_wind_speedxr(era, "obs")

    test_dataset = WeatherYearEnsembleDataset(
        data_path=test_folder,
        obs_path=obs_folder,
        valid_years=valid_years,
        subset="test")

    print("test dataset", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=16)
    lat_beg = test_dataset.latitude_beg

    # load model weights
    model = DRUnetPriorVar(77,2,variable).to(device)
    model.load_state_dict(torch.load(
        f"{model_folder}/model_{epoch}.pth"
        ))
    model.eval()

    # Compute performance metrics
    results = []
    i = 0
    for batch in test_loader:
        X = batch['input'].to(device)
        if variable ==  "2m_temperature":
            y = batch['truth'].to(device)[:,0,:,:]
            mu = batch['mu'].to(device)[:,0,:,:]
            sigma = batch['sigma'].to(device)[:,0,:,:]
        elif variable == "10m_wind_speed":
            y = batch['truth'].to(device)[:,1,:,:]
            mu = batch['mu'].to(device)[:,1,:,:]
            sigma = batch['sigma'].to(device)[:,1,:,:]
        lead_time = batch['lead_time'].item()

        # climatology
        valid_time = batch['valid_time']
        day_of_year = batch["day_of_year"]
        mu_clim = torch.tensor(climato[f"{variable}_mean"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 
        sigma_clim = torch.tensor(climato[f"{variable}_std"].sel(dayofyear=day_of_year).values[0]).T[lat_beg:,:] 

        # model inference
        maps, out_distrib = model(X, mu, sigma)
        truth = era.sel(time=valid_time) 
        y = torch.Tensor(truth[f"{variable}"].values.transpose(0,2,1))[:,lat_beg:,:].to(device)

        # add trend temperature (trend y)
        Xt = pd.to_datetime(valid_time).astype(np.int64)[0] * 1e-9
        trend = test_dataset.trend_model_truth.predict(Xt.reshape(-1,1)).reshape(1, 240, 121).transpose(0,2,1)[:,lat_beg:,:]
        if variable == "2m_temperature":
            out_distrib.loc += torch.tensor(trend, dtype=out_distrib.loc.dtype).to(device)

        climato_distrib = Normal(mu_clim.to(device), sigma_clim.to(device))

        crps = crps_normal(out_distrib,y).cpu().detach().numpy()  # shape (1, lat, lon)
        crps_climato = crps_normal(climato_distrib,y).cpu().detach().numpy() # shape (1, lat, lon)


        forecast_time = batch['forecast_time'][0]
        # Create a multi-index for the time dimension
        time_index = pd.MultiIndex.from_tuples([(lead_time, forecast_time)], names=["lead_time", "forecast_time"])

        if variable == "2m_temperature":
            ds = xr.Dataset(
                data_vars=dict(
                    crps_temperature=(["time", "latitude", "longitude"], crps),
                    crps_temperature_climato=(["time", "latitude", "longitude"], crps_climato),
                ),
                coords=dict(
                    longitude=("longitude", test_loader.dataset.longitude), # 1D array
                    latitude=("latitude", test_loader.dataset.latitude), # 1D array
                    time=("time", time_index), # 2D array
                )
            )
        elif variable == "10m_wind_speed":
            ds = xr.Dataset(
                data_vars=dict(
                    crps_wind_speed=(["time", "latitude", "longitude"], crps),
                    crps_wind_speed_climato=(["time", "latitude", "longitude"], crps_climato),
                ),
                coords=dict(
                    longitude=("longitude", test_loader.dataset.longitude), # 1D array
                    latitude=("latitude", test_loader.dataset.latitude), # 1D array
                    time=("time", time_index), # 2D array
                )
            )
        
        if i%100 == 0:
            print(i, crps.mean(), crps_climato.mean(), lead_time)
        i+=1
        
        # Reset the index to convert MultiIndex into separate variables
        ds = ds.reset_index('time')
        results.append(ds)

    final_ds = xr.concat(results, dim='time')

    # Write to NetCDF file
    results_file_path = f"{model_folder}/crps_{epoch}.nc"
    final_ds.to_netcdf(results_file_path)

### RAW AND CLIMATOLOGY INFERENCES 
def ClimatoModel_inference(lead_time, valid_years, train_years):
    data_folder = "/home/majanvie/scratch/data/raw"
    test_folder = f"{data_folder}/test"
    obs_folder = f"{data_folder}/obs"
    climato_folder = f"{obs_folder}/climato"

    base_dir = f"{climato_folder}/mean_train"
    full_results = []   
    for month in range(1,13):
        print(f"Month {month}")
        for variable in ["2m_temperature", "10m_wind_speed"]:
            print(f"Variable {variable}")
            model_path = f"{base_dir}/{variable}_{train_years[0]}_{train_years[-1]}_month{month}_lead{lead_time}.nc"
            climato_path = f"{climato_folder}/{variable}_{train_years[0]}_{train_years[-1]}_month{month}_lead{lead_time}.nc"
            climato = xr.open_dataset(climato_path)
            gaussian = xr.open_dataset(f"{model_path}")

            test_dataset = WeatherDataset(
                data_path=test_folder,
                obs_path=obs_folder,
                target_variable=variable,
                lead_time_idx=lead_time,
                valid_years=valid_years,
                valid_months=[month,month],
                subset="test")

            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

            # Compute performance metrics
            results = []
            for batch in test_loader:
                y = batch['truth']
                valid_time = batch['valid_time']
                valid_date = pd.to_datetime(valid_time).strftime('%m-%d')

                mu = torch.tensor(gaussian["mu"].sel(date=valid_date).values[0])
                sigma = torch.tensor(gaussian["sigma"].sel(date=valid_date).values[0])

                mu_clim = torch.tensor(climato["mu"].sel(date=valid_date).values[0])
                sigma_clim = torch.tensor(climato["sigma"].sel(date=valid_date).values[0])

                out_distrib = Normal(mu, sigma)
                climato_distrib = Normal(mu_clim, sigma_clim)

                crps_var = crps_normal(out_distrib,y).detach().numpy()  # shape (1, lat, lon)
                crps_climato = crps_normal(climato_distrib,y).detach().numpy() # shape (1, lat, lon)

                lead_time = batch['lead_time'].item()  
                forecast_time = batch['forecast_time'][0]
                # Create a multi-index for the time dimension
                time_index = pd.MultiIndex.from_tuples([(lead_time, forecast_time)], names=["lead_time", "forecast_time"])
                
                if variable == "2m_temperature":
                    ds = xr.Dataset(
                        data_vars=dict(
                            crps_temperature=(["time", "latitude", "longitude"], crps_var),
                            crps_temperature_climato=(["time", "latitude", "longitude"], crps_climato),
                        ),
                        coords=dict(
                            longitude=("longitude", test_loader.dataset.longitude), # 1D array
                            latitude=("latitude", test_loader.dataset.latitude), # 1D array
                            time=("time", time_index), # 2D array
                        )
                    )
                elif variable == "10m_wind_speed":
                    ds = xr.Dataset(
                        data_vars=dict(
                            crps_wind_speed=(["time", "latitude", "longitude"], crps_var),
                            crps_wind_speed_climato=(["time", "latitude", "longitude"], crps_climato),
                        ),
                        coords=dict(
                            longitude=("longitude", test_loader.dataset.longitude), # 1D array
                            latitude=("latitude", test_loader.dataset.latitude), # 1D array
                            time=("time", time_index), # 2D array
                        ))

                # Reset the index to convert MultiIndex into separate variables
                ds = ds.reset_index('time')
                results.append(ds)
            final_ds = xr.concat(results, dim='time')

            # Write to NetCDF file
            results_file_path = f"{base_dir}/crps_month{month}_{variable}.nc"
            final_ds.to_netcdf(results_file_path)

def RawIFS_inference(lead_time, valid_years, train_years):
    data_folder = "/home/majanvie/scratch/data" 
    test_folder = f"{data_folder}/test/EMOS"
    obs_folder = f"{data_folder}/raw/obs"
    climato_folder = f"{obs_folder}/climato"

    base_dir = f"{test_folder}/crps/lead{lead_time}"
    full_results = []   
    for month in range(1,13):
        print(f"Month {month}")
        for variable in ["2m_temperature", "10m_wind_speed"]:
            print(f"Variable {variable}")
            climato_path = f"{climato_folder}/{variable}_{train_years[0]}_{train_years[-1]}_month{month}_lead{lead_time}.nc"
            climato = xr.open_dataset(climato_path)

            test_dataset = WeatherEnsembleDataset(
                data_path=test_folder,
                obs_path=obs_folder,
                target_variable=variable,
                lead_time_idx=lead_time,
                valid_years=valid_years,
                valid_months=[month,month],
                subset="test")

            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

            # Compute performance metrics
            results = []
            for batch in test_loader:
                mu = batch['mu']
                sigma = batch['sigma']
                y = batch['truth']
                valid_time = batch['valid_time']
                valid_date = pd.to_datetime(valid_time).strftime('%m-%d')

                mu_clim = torch.tensor(climato["mu"].sel(date=valid_date).values[0])
                sigma_clim = torch.tensor(climato["sigma"].sel(date=valid_date).values[0])

                out_distrib = Normal(mu, sigma)
                climato_distrib = Normal(mu_clim, sigma_clim)

                crps_var = crps_normal(out_distrib,y).detach().numpy()  # shape (1, lat, lon)
                crps_climato = crps_normal(climato_distrib,y).detach().numpy() # shape (1, lat, lon)

                #lead_time = batch['lead_time'].item()  
                forecast_time = batch['forecast_time'][0]
                # Create a multi-index for the time dimension
                time_index = pd.MultiIndex.from_tuples([(lead_time, forecast_time)], names=["lead_time", "forecast_time"])
                
                if variable == "2m_temperature":
                    ds = xr.Dataset(
                        data_vars=dict(
                            crps_temperature=(["time", "latitude", "longitude"], crps_var),
                            crps_temperature_climato=(["time", "latitude", "longitude"], crps_climato),
                        ),
                        coords=dict(
                            longitude=("longitude", test_loader.dataset.longitude), # 1D array
                            latitude=("latitude", test_loader.dataset.latitude), # 1D array
                            time=("time", time_index), # 2D array
                        )
                    )
                elif variable == "10m_wind_speed":
                    ds = xr.Dataset(
                        data_vars=dict(
                            crps_wind_speed=(["time", "latitude", "longitude"], crps_var),
                            crps_wind_speed_climato=(["time", "latitude", "longitude"], crps_climato),
                        ),
                        coords=dict(
                            longitude=("longitude", test_loader.dataset.longitude), # 1D array
                            latitude=("latitude", test_loader.dataset.latitude), # 1D array
                            time=("time", time_index), # 2D array
                        ))

                # Reset the index to convert MultiIndex into separate variables
                ds = ds.reset_index('time')
                results.append(ds)
            final_ds = xr.concat(results, dim='time')

            # Write to NetCDF file
            results_file_path = f"{base_dir}/crps_month{month}_{variable}.nc"
            final_ds.to_netcdf(results_file_path)


if __name__ == "__main__":
    valid_years = [2018,2022]
    train_years = [1996,2017]
    
    #for lead_time in [39]:
        #ClimatoModel_inference(lead_time, [2018,2022],train_years=[1996,2017])
        #RawIFS_inference(lead_time, [2018,2022], [1996,2017])
        #SpatialEMOS_inference(lead_time, [2018,2022], [1996,2017])
        #specialSpatialEMOS_inference(lead_time, [2018,2022], [1996,2017], "log_nostd_MM_lead", 73, 19, "2m_temperature") # emos 3
        #specialSpatialEMOS_inference_detrend(lead_time, [2018,2022], [1996,2017], "log_MM_detrend", 77 , 9, "2m_temperature") # detrend MM 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)    
    DRUnet_inference(valid_years, train_years, train_index=6, epoch=9, device=device)
    DRUnet_inference(valid_years, train_years, train_index=6, epoch=5, device=device)

    # DRUnet+prior single
    parser = argparse.ArgumentParser(description="Run spatial month experiment.")
    parser.add_argument('-var', '--var', type=str, required=True, help="Lead index to use in the experiment")
    parser.add_argument('-id', '--id', type=int, required=True, help="Experiment id")
    args = parser.parse_args()
    variable = args.var
    train_index = args.id
    valid_years = [2018,2022]
    train_years = [1996,2017]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)    
    DRUnet_inference_Prior_Var(valid_years, train_years, train_index=train_index, epoch=10, device=device, variable=variable)







