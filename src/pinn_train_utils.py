import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


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


def advection_diffusion(yhat, y_coords, u, v, D):
    # Compute gradients
    # y_coords.requires_grad_(True)
    grads = torch.autograd.grad(
        yhat,
        y_coords,
        grad_outputs=torch.ones_like(yhat),
        create_graph=True,
    )[0]

    # Compute second derivatives (you might need to adjust this based on your specific needs)
    dC_dq = torch.autograd.grad(
        grads,
        y_coords,
        grad_outputs=torch.ones_like(grads),
        create_graph=True,
    )[0]

    residual = u * grads[:, :, 0] + v * grads[:, :, 1] - D * dC_dq.sum(axis=2)
    return residual


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


def train_epoch(
    model,
    optimizer,
    criterion,
    train_loader,
    device,
    lambda_phys=0.1,
    D=0.1,
    num_sensors=100,
    epoch=0,
):
    model.train()
    rec_losses, phys_losses = [], []
    pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y_coords, y_targets = sample_sensors(y, num_sensors)
        y_coords = y_coords.to(device)
        y_coords.requires_grad = True
        y_targets = y_targets.to(device)
        optimizer.zero_grad()
        outputs = model(x, y_coords)
        rec_loss = criterion(outputs, y_targets)

        u, v = sample_velocity_at_sensors(x[:, 7, :, :], x[:, 8, :, :], y_coords)
        residual = advection_diffusion(outputs, y_coords, u, v, D)
        loss_physics = residual.pow(2).mean()

        loss = rec_loss + lambda_phys * loss_physics
        loss.backward()
        optimizer.step()
        rec_losses.append(rec_loss.item())
        phys_losses.append(loss_physics.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "rec. loss": f"{rec_loss.item():.4f}",
                "phys. loss": f"{loss_physics.item():.4f}",
            }
        )
    # Close the progress bar
    pbar.close()
    # Return average loss for the epoch
    return rec_losses, phys_losses


def validate(
    model, criterion, val_loader, device, lambda_phys=0.1, D=0.1, num_sensors=100
):
    model.eval()
    total_loss, physics_loss = 0.0, 0.0

    # with torch.no_grad():
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y_coords, y_targets = sample_sensors(y, num_sensors)
        y_coords = y_coords.to(device)
        y_coords.requires_grad = True
        y_targets = y_targets.to(device)
        outputs = model(x, y_coords)
        rec_loss = criterion(outputs, y_targets)

        u, v = sample_velocity_at_sensors(x[:, 7, :, :], x[:, 8, :, :], y_coords)
        residual = advection_diffusion(outputs, y_coords, u, v, D)
        loss_physics = residual.pow(2).mean()

        loss = rec_loss + lambda_phys * loss_physics
        total_loss += loss.item()
        physics_loss += loss_physics.item()
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "rec. loss": f"{rec_loss.item():.4f}",
                "phys. loss": f"{loss_physics.item():.4f}",
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pbar.close()

    return total_loss / len(val_loader), physics_loss / len(val_loader)


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    scheduler=None,
    lambda_physics=0.1,
    D=0.1,
    num_sensors=100,
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
            lambda_phys=lambda_physics,
            D=D,
            num_sensors=num_sensors,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_loss_mn = np.mean(train_loss)
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_mn:.4f}, Val Loss: {val_loss:.4f}, Val phys. Loss: {val_phys:.4f}"
            )

        if scheduler is not None:
            scheduler.step()
    return train_losses, val_losses
