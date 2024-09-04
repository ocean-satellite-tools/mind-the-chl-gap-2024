def sample_sensors(target, num_sensors, seed=None):
    """
    Sample random locations from the target image to act as sensors.

    :param target: Target chlorophyll values tensor (batch_size, height, width)
    :param num_sensors: Number of sensor locations to sample
    :param seed: Random seed for reproducibility
    :return: Sampled sensor locations and their corresponding values
    """
    if seed is not None:
        torch.manual_seed(seed)

    batch_size, height, width = target.shape

    # Sample random locations
    y_coords = torch.randint(0, height, (batch_size, num_sensors))
    x_coords = torch.randint(0, width, (batch_size, num_sensors))

    # Get values at sampled locations
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_sensors)
    sensor_values = target[batch_indices, y_coords, x_coords]

    # Normalize coordinates to [-1, 1] range and stack them
    normalized_coords = torch.stack(
        [
            2 * x_coords.float() / (width - 1) - 1,
            2 * y_coords.float() / (height - 1) - 1,
        ],
        dim=-1,
    )

    # Return normalized coordinates with shape (batch_size, num_sensors, 2)
    # and sensor values with shape (batch_size, num_sensors)
    return normalized_coords, sensor_values


def sample_velocity_at_sensors(u, v, normalized_coords):
    """
    Sample the velocity fields u and v at the sensor locations.

    :param u: Velocity field in the x direction (batch_size, height, width)
    :param v: Velocity field in the y direction (batch_size, height, width)
    :param normalized_coords: Normalized sensor coordinates (batch_size, num_sensors, 2)
    :return: Sampled u and v values at the sensor locations
    """
    batch_size, height, width = u.shape
    num_sensors = normalized_coords.shape[1]

    # Convert normalized coordinates back to image coordinates
    x_coords = ((normalized_coords[:, :, 0] + 1) * (width - 1) / 2).long()
    y_coords = ((normalized_coords[:, :, 1] + 1) * (height - 1) / 2).long()

    # Ensure indices are within bounds
    x_coords = torch.clamp(x_coords, 0, width - 1)
    y_coords = torch.clamp(y_coords, 0, height - 1)

    # Sample u and v at the sensor locations
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_sensors)
    u_samples = u[batch_indices, y_coords, x_coords]
    v_samples = v[batch_indices, y_coords, x_coords]

    return u_samples, v_samples
