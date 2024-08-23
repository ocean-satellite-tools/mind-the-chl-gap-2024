import numpy as np
import xarray as xr


def load_preprocess_data():
    # Load and preprocess the data
    zarr_path = "~/shared-public/mind_the_chl_gap/U-Net_with_CHL_pred.zarr"
    zarr_ds = xr.open_zarr(zarr_path)["gapfree_pred"]

    # Load gappy data (level 3)
    level3_path = "~/shared-public/mind_the_chl_gap/IO.zarr"
    level3_ds = xr.open_zarr(level3_path).sel(
        lat=slice(zarr_ds.lat.values.max(), zarr_ds.lat.values.min()),
        lon=slice(zarr_ds.lon.values.min(), zarr_ds.lon.values.max()),
    )
    # Extract latitude and longitude extents to determine height and width
    lat = level3_ds.lat.values
    lon = level3_ds.lon.values

    # Set a time slice for the datasets
    time_slice = slice("2022-01-01", "2022-12-31")  # Adjust as needed
    gappy_data = level3_ds.sel(time=time_slice)

    return gappy_data, lat, lon


def stack_data(gappy_data, flatten=True):
    # Variables to include in the branch net
    variables = ["CHL_cmes-level3", "sst", "u_wind", "v_wind", "air_temp", "ug_curr"]

    # Prepare the data by stacking variables for each time slice
    stacked_data = np.stack([gappy_data[var].values for var in variables], axis=1)
    stacked_data = np.transpose(
        stacked_data, (0, 2, 3, 1)
    )  # Shape: (train_size, height, width, num_variables)
    if flatten:
        timesteps, w, h, channels = stacked_data.shape
        stacked_data = stacked_data.reshape(timesteps, w * h, channels)
    return stacked_data


def split_train_test(data, frac_train=0.8):
    # Split data into training and testing sets
    dataset_size = data.shape[0]
    train_size = int(frac_train * dataset_size)

    # Training and testing data
    train_ims = data[:train_size].astype(np.float32)
    test_ims = data[train_size:].astype(np.float32)

    return train_ims, test_ims
