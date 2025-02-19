import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class XarrayDataset(Dataset):
    "Doesn't quite work yet. This is for generating a dataset directly from a .zarr file."

    def __init__(self, file_path, chunks="auto"):
        self.ds = xr.open_zarr(file_path, chunks=chunks)
        self.channels = list(self.ds.data_vars)
        self.num_samples = self.ds.dims["time"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = []
        for channel in self.channels:
            # Load a single time step for all channels
            channel_data = self.ds[channel].isel(time=idx).values
            data.append(channel_data)

        # Stack channels along a new dimension
        stacked_data = np.stack(data, axis=0)

        # Convert to torch tensor
        return torch.from_numpy(stacked_data).float()


def load_data(data_dir, time_start="2018-01-01", time_end="2022-12-31"):
    zarr_ds = xr.open_zarr(store=data_dir + "IO.zarr", consolidated=True)
    zarr_ds = zarr_ds.sel(lat=slice(32, -11.75), lon=slice(42, 101.75))
    zarr_ds = zarr_ds.sortby("time")
    zarr_ds = zarr_ds.sel(time=slice(time_start, time_end))
    return zarr_ds


def custom_normalize(data):
    non_zero_mask = data != 0
    normalized_data = np.zeros_like(data, dtype=np.float32)
    if np.any(non_zero_mask):
        non_zero_data = data[non_zero_mask]
        min_val = np.min(non_zero_data)
        max_val = np.max(non_zero_data)
        # Avoid division by zero
        if max_val > min_val:
            normalized_data[non_zero_mask] = (non_zero_data - min_val) / (
                max_val - min_val
            )
        else:
            normalized_data[non_zero_mask] = 1  # If all non-zero values are the same
    return normalized_data


def stack_data(zarr_ds, variables, normalize=True, drop_nans=True):
    stacked_data = []
    for var in variables:
        x = zarr_ds[var].values
        if var in ["CHL_cmes-level3", "CHL_cmes-gapfree"]:
            x = np.log(x)
        if drop_nans:
            x = np.nan_to_num(x, nan=0.0)

        if normalize:
            x = custom_normalize(x)

        stacked_data.append(x)
    return stacked_data


def gen_dataset(zarr_ds):
    variables = [
        "CHL_cmes-level3",
        "air_temp",
        "sst",
        "adt",
        "curr_dir",
        "u_wind",
        "v_wind",
        "u_curr",
        "v_curr",
    ]
    stacked_data = stack_data(zarr_ds, variables, drop_nans=True)
    xs = np.transpose(np.stack(stacked_data, axis=3), (0, 3, 1, 2))
    xs = custom_normalize(xs)

    target = np.log(zarr_ds["CHL_cmes-gapfree"].values)
    target = np.nan_to_num(target, nan=0.0)
    target = custom_normalize(target)

    return CustomDataset(xs, target)


def get_data_loaders(data: CustomDataset, frac_train, batch_size=32):
    n_train = int(np.ceil(frac_train * len(data)))
    n_test = len(data) - n_train
    train_set, test_set = random_split(data, [n_train, n_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def data_loaders_from_zarr(zarr_ds, frac_train, batch_size=32):
    data = gen_dataset(zarr_ds)
    train_loader, test_loader = get_data_loaders(
        data, frac_train, batch_size=batch_size
    )

    return train_loader, test_loader
