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
from src.vae_model import *
from tqdm import tqdm

if __name__ == "__main__":

    data_dir = "data/"
    batch_size = 32
    kl_lambda = 0.5

    chl_data = torch.load(data_dir + "chl_dataset_2016_2022.pt")
    train_loader, test_loader = get_data_loaders(chl_data, 0.8, batch_size=batch_size)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    log_dir = "runs/vae"  # for tensorboard
    model = ConvVAE(input_channels=10, latent_dim=40).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, verbose=True
    )

    num_epochs = 2_000
    train_rec_losses, train_kl_losses, val_losses = train(
        model,
        opt,
        train_loader,
        test_loader,
        device,
        kl_lambda=kl_lambda,
        num_epochs=num_epochs,
        show_every=25,
        plot_every=50,
        scheduler=scheduler,
        checkpoint_dir="saved_models/checkpoints/vae/",
        checkpoint_patience=25,
        log_dir=log_dir,
    )
