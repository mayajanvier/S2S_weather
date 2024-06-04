import pandas as pd
from model import MOS
from processings.dataset import PandasDataset, compute_wind_speed
from train import train, create_training_folder
from torch.utils.data import DataLoader
from metrics import crps_normal
import json
from sklearn.model_selection import train_test_split


def main():
    # get training parameters
    param_folder = 'parameters/' 
    with open(param_folder+"train.json") as fp:
        params = json.load(fp)
    
    name_experiment = params["name_experiment"]
    target_column = params["target_variable"]
    batch_size = params["batch_size"]
    nb_epoch = params["nb_epoch"]
    lr = params["lr"]
    out_dim = params["out_dim"]
    save_every = params["save_every"]

    # create training folder
    folder = create_training_folder(name_experiment)

    # load data
    data_folder = "../scratch/data/train/"
    # TODO: change this to the correct path
    data_path = data_folder+f'PPE_OPT_lat=-90.0_lon=0.0_lead=24h.json'
    train_data  = pd.read_json(data_path)
    train_data = compute_wind_speed(train_data)
    feature_dim = len(train_data["input"][0])

    # separate train and validation randomly
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, shuffle=True)
    train_indices, val_indices = train_data.index.tolist(), val_data.index.tolist()

    # build dataloaders
    train_data   = PandasDataset(train_data, target_column=target_column)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data = PandasDataset(val_data, target_column=target_column)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # model setup and training
    criterion = crps_normal
    model = MOS(feature_dim, out_dim)

    # save parametrization of training
    params["data_path"] = data_path
    params["feature_dim"] = feature_dim
    params["metric"] = "CRPS_Normal" # TODO les noms faire liste lien comme a dreem, pareil pour model 
    params["train_indices"] = train_indices
    params["val_indices"] = val_indices
    with open(folder+"/params.json", "w") as fp:
        json.dump(params, fp)

    # training
    train(
        train_loader,
        val_loader,
        model=model,
        nb_epoch=nb_epoch,
        lr=lr,
        criterion=criterion,
        result_folder=folder,
        save_every=save_every
        )


if __name__ == "__main__":
    main()