import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.image_sampling_utils import *


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


def train_epoch(
    model,
    optimizer,
    criterion,
    train_loader,
    device,
    water_mask,
    sample_water_only=False,
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
        if sample_water_only:
            y_coords, y_targets = sample_water_sensors(y, num_sensors, water_mask)
        else:
            y_coords, y_targets = sample_sensors(y, num_sensors)

        y_coords = y_coords.to(device)
        y_coords.requires_grad = True
        u, v = sample_velocity_at_sensors(x[:, 7, :, :], x[:, 8, :, :], y_coords)

        y_targets = y_targets.to(device)
        optimizer.zero_grad()
        outputs = model(x, y_coords)
        rec_loss = criterion(outputs, y_targets)

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
    model,
    criterion,
    val_loader,
    device,
    water_mask,
    sample_water_only=False,
    lambda_phys=0.1,
    D=0.1,
    num_sensors=100,
):
    model.eval()
    total_loss, physics_loss = 0.0, 0.0

    # with torch.no_grad():
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for x, y in pbar:
        x = x.to(device)
        if sample_water_only:
            y_coords, y_targets = sample_water_sensors(y, num_sensors, water_mask)
        else:
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
    my_x, my_y = next(iter(val_loader))
    my_x = my_x.to(device)
    my_y = my_y.to(device)
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
            if plot_every:
                output = sample_grid(model, my_x)
                plot_grid_output(output, my_x, my_y, water_mask, batch_ind=0)

        if scheduler is not None:
            scheduler.step(val_loss)
    return train_losses, val_losses


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
    output, x, y, water_mask, batch_ind=2, savefig=False, savename=None
):
    batch_size = x.shape[0]
    wm = (1.0 - water_mask).repeat(batch_size, 1, 1)
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
