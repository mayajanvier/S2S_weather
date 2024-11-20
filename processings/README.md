# Downloading and formatting WeatherBench2 data
This folder contains the codes to prepare the data before running the experiments. Follow these steps: 

1. **Downloading:** Create the folders needed to store this raw data: `obs`, `train_surface_dir`, `test_surface_dir`, `train_levels_dir` and `test_levels_dir`. Update your paths to it using the variables `obs_folder`, `train_folder` and `test_folder` in the code. Run `download_weatherbench2.py`. 
The script first downloads observations (ERA5) into the `obs_folder`. Then, we download the surface variables files with their members into separated train and test surface folders. Finally, it directly computes and saves the aggregated mean and std for the pressure levels variables into separated train and test levels folders.

2. **Formatting raw data:** Create your own `out_folder`, with `mean` and `std` sub-directories for both train and test data. Indicate the correct `surface_dir` and `levels_dir` containing the raw data from step 1. Run `format_data.py`. This script merges surface and pressure levels data into single netcdf files, for train and test data. 

3. **Preprocessing:** In order to run the experiments, we need several files: *data indices* containing paths and information about our samples, *trend models* for temperature, *scalers* for data normalisation, and *climatology*. Running `dataset.py` main code will create all of these files. Create the needed folders to store them and setup your own paths in the main code and in the different methods of the dataset classes before running the code. 

You should be setup to go to the `main.py` script and start running experiments. 