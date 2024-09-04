import torch
import torch.nn.functional as F
import numpy as np


def sample_image_at_points(image, coords):
    # Ensure coords are in the range [-1, 1]
    coords = torch.clamp(coords, -1.0, 1.0)
    # Reshape coords for grid_sample
    coords = coords.unsqueeze(1)
    # Reverse the order of coordinates from (y, x) to (x, y) if necessary
    coords = coords[..., [1, 0]]
    # Use grid_sample for bilinear interpolation
    sampled = F.grid_sample(
        image.unsqueeze(1),
        coords,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    # Reshape output
    return sampled.squeeze(2).permute(0, 2, 1).squeeze()


def get_uniform_coords(batch_size, num_sensors):
    coords = torch.rand(batch_size, num_sensors, 2) * 2 - 1
    return coords


def sample_water_sensors(y, num_sensors, water_mask, oversample_factor=10):
    device = y.device
    batch_size, h, w = y.shape
    water_mask = water_mask.to(device).unsqueeze(0)

    coords = torch.zeros((batch_size, num_sensors, 2), device=device)
    samples = torch.zeros((batch_size, num_sensors), device=device)

    # Generate oversampled coordinates
    candidates = get_uniform_coords(batch_size, num_sensors * oversample_factor).to(
        device
    )

    # Sample water mask at candidate coordinates
    mask_values = sample_image_at_points(
        water_mask.repeat(batch_size, 1, 1), candidates
    )

    # Find valid coordinates (where mask value < 0.5)
    valid_mask = mask_values < 0.5

    for b in range(batch_size):
        valid_coords = candidates[b, valid_mask[b]]
        if valid_coords.shape[0] >= num_sensors:
            # If we have enough valid coordinates, randomly select num_sensors
            perm = torch.randperm(valid_coords.shape[0])
            coords[b] = valid_coords[perm[:num_sensors]]
        else:
            # If we don't have enough, use all valid ones and resample for the rest
            coords[b, : valid_coords.shape[0]] = valid_coords
            remaining = num_sensors - valid_coords.shape[0]
            while remaining > 0:
                new_candidates = get_uniform_coords(
                    1, remaining * oversample_factor
                ).to(device)
                new_mask_values = sample_image_at_points(water_mask, new_candidates)
                new_valid_mask = new_mask_values < 0.5
                new_valid_coords = new_candidates[0, new_valid_mask[0]]
                if new_valid_coords.shape[0] > 0:
                    to_add = min(remaining, new_valid_coords.shape[0])
                    coords[b, -remaining : -remaining + to_add] = new_valid_coords[
                        :to_add
                    ]
                    remaining -= to_add

    # Sample the image at the valid coordinates
    for b in range(batch_size):
        samples[b] = sample_image_at_points(
            y[b].unsqueeze(0), coords[b].unsqueeze(0)
        ).squeeze()

    return coords, samples


def get_grid_coords(batch_size, height, width):
    # Generate the grid coordinates
    x_coords = torch.linspace(-1, 1, height)
    y_coords = torch.linspace(-1, 1, width)

    # Create the meshgrid, with coordinates in the order of (x, y)
    meshgrid = torch.meshgrid(x_coords, y_coords, indexing="ij")

    # Stack and reshape to get the coordinates in the format (2, height * width)
    coords = torch.stack(meshgrid, dim=0).reshape(2, -1)

    # Repeat the coordinates for the batch
    coords = coords.repeat(batch_size, 1, 1)

    # Permute to get the final shape (batch_size, height * width, 2)
    coords = coords.permute(0, 2, 1)

    return coords


def sample_sensors(y, num_sensors):
    device = y.device
    batch_size, w, h = y.shape
    coords = get_uniform_coords(batch_size, num_sensors).to(device)
    return coords, sample_image_at_points(y, coords)


def sample_velocity_at_sensors(u, v, coords):
    u_sample = sample_image_at_points(u, coords)
    v_sample = sample_image_at_points(v, coords)

    return u_sample, v_sample
