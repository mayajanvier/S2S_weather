# S2S_weather: Improving Subseasonal-to-Seasonal Weather Forecasting by Leveraging Statistical Post-Processing
End of studies interniship at INRIA Paris in ARCHES team, supervised by Claire Monteleoni and David Landry (May-November 2024)

Subseasonal-to-Seasonal (S2S) weather forecasts are global or local weather predictions (classically 2meter temperature and total precipitation) beyond 2 weeks but less than 3 months ahead. These types of forecasts are essential for climate change adaptation, with an increasing number of extreme meteorological events (floods, cold-waves) threatening infrastructures and human health. However, classical Numerical Weather Predictions (NWP) atmospheric models show low skill at these ranges, as they are too far from the initial conditions while being sensitive to slowly evolving coupled components with the ocean.

We investigated statistical post-processing methods, including deep learning, to improve the IFS forecasts from the European Centre for Medium-Range Weather Forecasts), for 2m temperature and 10m wind speed, at 7, 14, 21, 28, 35 and 39 days ahead. We built reusable pipelines for [WeatherBench2](https://github.com/google-research/weatherbench2) (Rasp et al. 2024) data, from downloading to pre-processing. We also discussed and implemented normalization strategies and metrics fit to our problem and weather post-processing in general. You can read our findings in the report. 

![alt text](https://github.com/mayajanvier/S2S_weather/blob/main/crpss_temp_14.png)
Figure: CRPSS of 2m temperature at lead 14 days for different methods: top row: raw model and EMOS (Gneiting et al. 2005), bottom row: variants of DRUNet (Pic et al. 2024)

# Repository organisation 
The `parameter` folder contains the WeatherBench2 paths to the files we used in this study.

The `processings` folder contains codes to download (`download_weatherbench2.py`) and pre-process this data (`dataset.py`). For DRUnet, the [Month Lead Agg] is performed by the WeatherYearEnsembleDataset, while the [General Agg] by the WeatherYearEnsembleDataset. 

You can run the different experiments of this project by running the functions in `main.py`, and the models are defined in `model.py`: 
- post-processing models: SpatialEMOS, DRUNetPrior (DRUnet+prior both), DRUnetPriorVar (DRUnet+prior single)
- forecasting models: DRUnet (DRUnet both), DRUnetVar (DRUnet single)

