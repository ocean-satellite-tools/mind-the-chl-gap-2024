import numpy as np
import xarray as xr
import os
import sys

sys.path.append(os.getcwd())

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_utils import *
from src.models import DeepONet
from src.pinn_train_utils import *

import wandb


def get_water_mask(zarr_ds):
    st_ = zarr_ds["sst"].values[0, :, :]
    water_mask = np.zeros_like(st_)
    water_mask[np.isnan(st_)] = 1.0
    return torch.tensor(water_mask)


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    water_mask,
    sample_water_only=False,
    num_epochs=100,
    scheduler=None,
    lambda_physics=0.1,
    D=0.1,
    num_sensors=100,
    show_every=50,
    plot_every=False,
):
    model.train()
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss, phys_loss = train_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            water_mask,
            sample_water_only=sample_water_only,
            lambda_phys=lambda_physics,
            D=D,
            num_sensors=num_sensors,
            epoch=epoch,
        )
        val_loss, val_phys = validate(
            model,
            criterion,
            val_loader,
            device,
            water_mask,
            sample_water_only=sample_water_only,
            lambda_phys=lambda_physics,
            D=D,
            num_sensors=num_sensors,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_loss_mn = np.mean(train_loss)
        if (epoch + 1) % show_every == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_mn:.4f}, Val Loss: {val_loss:.4f}, Val phys. Loss: {val_phys:.4f}"
            )

        wandb.log({"loss": val_loss})

        if scheduler is not None:
            scheduler.step(val_loss)
    return train_losses, val_losses


def get_act_fun_from_config(act_fun):
    f = act_fun
    if f == "swish":
        return nn.SiLU
    elif f == "gelu":
        return nn.GELU
    elif f == "tanh":
        return nn.Tanh
    elif f == "elu":
        return nn.ELU
    elif f == "relu":
        return nn.ReLU


def train_model():
    run = wandb.init()
    config = wandb.config
    if config.onet_output_fun == "relu":
        onet_output = F.relu
    elif config.onet_output_fun == "sigmoid":
        onet_output = F.sigmoid
    else:
        onet_output = lambda x: x
    torch.random.manual_seed(0)
    model = DeepONet(
        9,
        2,
        64,
        num_basis_functions=config.num_basis_functions,
        trunk_act=get_act_fun_from_config(config.trunk_act),
        branch_linear_act=get_act_fun_from_config(config.branch_linear_act),
        onet_output_fun=onet_output,
        image_size=(176, 240),
    ).to(device)
    # model.apply(init_network)
    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    # scheduler = ReduceLROnPlateau(
    #     opt, mode="min", factor=0.5, patience=config.patience, verbose=True
    # )
    torch.cuda.empty_cache()
    num_sensors = config.num_sensors
    n_epochs = 50
    train_losses, test_losses = train(
        model,
        opt,
        criterion,
        train_loader,
        test_loader,
        device,
        water_mask,
        sample_water_only=True,
        num_epochs=n_epochs,
        lambda_physics=config.physics_weight,
        scheduler=None,
        num_sensors=num_sensors,
        show_every=10,
        D=config.D,
    )
    return train_losses, test_losses, model


if __name__ == "__main__":

    data_dir = "data/"

    time_start = "2017-01-01"
    time_end = "2022-12-31"

    zarr_ds = load_data(data_dir, time_start=time_start, time_end=time_end)

    batch_size = 64
    train_loader, test_loader = get_data_loaders(zarr_ds, 0.8, batch_size=batch_size)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    water_mask = get_water_mask(zarr_ds)

    wandb.login()

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "trunk_act": {"values": ["swish", "tanh", "elu"]},
            "branch_linear_act": {"values": ["swish", "tanh", "relu"]},
            "onet_output_fun": {"values": ["sigmoid", "relu", "none"]},
            "D": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
            "physics_weight": {"values": [0.1, 0.2, 0.05]},
            "num_basis_functions": {"values": [50, 100, 200]},
            "num_sensors": {"values": [1000, 1500]},
            "learning_rate": {"values": [1e-4, 5e-4, 5e-5]},
        },
    }

    sweep = wandb.sweep(sweep_config, project="CHL_deeponet_hyperparam_optim")
    wandb.agent(sweep, function=train_model, count=100)
