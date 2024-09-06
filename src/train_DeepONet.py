import numpy as np
import xarray as xr
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import sys

sys.path.append(os.getcwd())

from src.data_utils import *
from src.models import DeepONet
from src.pinn_train_utils import *

import matplotlib.pyplot as plt


def get_water_mask(zarr_ds):
    st_ = zarr_ds["sst"].values[0, :, :]
    water_mask = np.zeros_like(st_)
    water_mask[np.isnan(st_)] = 1.0
    return torch.tensor(water_mask)


def sample_grid(model, x):
    torch.cuda.empty_cache()
    batch_size = x.shape[0]
    height, width = (176, 240)
    dev = x.device
    grid_coords = get_grid_coords(batch_size, height, width).to(dev)

    output = torch.zeros((batch_size, height, width)).to(dev)
    chunk_size = 2000  # Adjust based on your GPU memory
    for i in range(0, height * width, chunk_size):
        chunk_coords = grid_coords[:, i : i + chunk_size, :]
        with torch.no_grad():
            chunk_output = model(x, chunk_coords)
        output.view(batch_size, -1)[:, i : i + chunk_size] = chunk_output
    torch.cuda.empty_cache()
    return output.cpu().numpy()


def plot_grid_output(
    model, x, y, water_mask, batch_ind=2, savefig=False, savename=None
):
    wm = (1.0 - water_mask).repeat(batch_size, 1, 1)
    output = sample_grid(model, x)
    masked_output = wm * output
    fig, axs = plt.subplots(1, 3, figsize=(12, 10))
    axs[0].imshow(y.cpu()[batch_ind, :, :], clim=(0, 1))
    axs[0].set_title("Original image")
    axs[1].imshow(masked_output[batch_ind, :, :], clim=(0, 1))
    axs[1].set_title("Model output sampled at grid (masked)")
    axs[2].imshow(output[batch_ind, :, :], clim=(0, 1))
    axs[2].set_title("Model Output sampled at grid (unnmasked)")

    if savefig:
        plt.savefig(f"plots/{savename}.png")
    plt.show()


if __name__ == "__main__":

    data_dir = "data/"
    time_start = "2017-01-01"
    time_end = "2022-12-31"

    zarr_ds = load_data(data_dir, time_start=time_start, time_end=time_end)

    batch_size = 100
    train_loader, test_loader = get_data_loaders(zarr_ds, 0.8, batch_size=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    water_mask = get_water_mask(zarr_ds)
    torch.random.manual_seed(0)

    my_x, my_y = next(iter(test_loader))
    my_x = my_x.to(device)

    for lambda_phys in [0.0, 0.1, 0.25]:
        if lambda_phys == 0.0:
            D_values = [1.0]
        else:
            D_values = [1.0, 0.1, 0.01, 0.001, 0.0001]
        for D in D_values:
            print(f"Training with D = {D}, lambda_phys = {lambda_phys}")
            torch.random.manual_seed(0)
            model = DeepONet(9, 2, 64, 150, (176, 240)).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=2e-4)
            criterion = nn.MSELoss()
            scheduler = ReduceLROnPlateau(
                opt, mode="min", factor=0.6, patience=8, verbose=True
            )

            torch.cuda.empty_cache()
            num_sensors = 1500
            n_epochs = 20
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
                lambda_physics=lambda_phys,
                scheduler=scheduler,
                num_sensors=num_sensors,
                show_every=20,
                D=D,
            )
            plot_grid_output(
                model,
                my_x,
                my_y,
                water_mask,
                batch_ind=10,
                savefig=True,
                savename=f"DeepONet_v0_D_{D}_phys_l_{lambda_phys}",
            )
