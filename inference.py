import pandas as pd 
from metrics import compute_crps_normal
import torch
import json
from processings.dataset import PandasDataset, compute_wind_speed
from torch.utils.data import DataLoader
from model import MOS


param_folder = "parameters/"
# Load parameters
with open(param_folder + "test.json") as fp: # test 
    test_params = json.load(fp)

model_folder = test_params["model_folder"]
with open(model_folder + "params.json") as fp: # train
    train_params = json.load(fp)
out_dim = train_params["out_dim"]

# Load and format test data 
test = pd.read_json(test_params["test_file_path"])
test = compute_wind_speed(test)
var = train_params["target_variable"] # one model per variable 
test_data = PandasDataset(test, var)
test_data = DataLoader(test_data, batch_size=1, shuffle=True)
    
# Load model weights
feature_dim = len(test["input"][0])
epoch = test_params["model_epoch"]
model = MOS(feature_dim,out_dim)
model.load_state_dict(torch.load(model_folder + f"model_{epoch}.pth"))
model.eval()

# Compute performance metrics
results_file_path = model_folder + f"results_model_{epoch}.json"
with open(results_file_path, 'w') as f:
    f.write("[\n")  # Start of JSON array
    first_entry = True

    crps_list = []
    for batch in test_data:
        # TODO: add metrics 
        crps_var = compute_crps_normal(model, batch).detach().numpy()[0]
        dict_date = {
            "forecast_time": batch["forecast_time"],
            "crps": float(crps_var)
            }
        crps_list.append(crps_var)
        if not first_entry:
            f.write(",\n")
        json.dump(dict_date, f)
        first_entry = False

    # Compute mean CRPS
    dict_mean = {
        "forecast_time": "mean",
        "crps": float(sum(crps_list)/len(crps_list))
        }
    if not first_entry:
        f.write(",\n")
    json.dump(dict_mean, f)

    # End of JSON array
    f.write("\n]")  





