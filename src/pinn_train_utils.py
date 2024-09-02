import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def advection_diffusion(c, u, v, D, dx, dy):
    # Simplified 2D advection-diffusion equation
    # dc/dt = -u * dc/dx - v * dc/dy + D * (d^2c/dx^2 + d^2c/dy^2)
    dc_dx = torch.gradient(c, dim=1)[0] / (2 * dx)
    dc_dy = torch.gradient(c, dim=2)[0] / (2 * dy)
    d2c_dx2 = torch.gradient(dc_dx, dim=1)[0] / (dx**2)
    d2c_dy2 = torch.gradient(dc_dy, dim=2)[0] / (dy**2)

    return -u * dc_dx - v * dc_dy + D * (d2c_dx2 + d2c_dy2)


def train_epoch(model, optimizer, criterion, train_loader, device, lambda_phys=0.1):
    model.train()
    rec_losses, phys_losses = [], []
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        x, y = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(x).squeeze()
        rec_loss = criterion(outputs, y)
        pred_dc_dt = advection_diffusion(
            outputs.squeeze(), x[:, 7, :, :], x[:, 8, :, :], D, dx, dy
        )
        loss_physics = F.mse_loss(pred_dc_dt, phys_target)
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


def validate(model, criterion, val_loader, device, lambda_phys=0.1):
    model.eval()
    total_loss, physics_loss = 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for inputs, targets in pbar:
            x, y = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            rec_loss = criterion(outputs, y)
            pred_dc_dt = advection_diffusion(
                outputs.squeeze(), x[:, 7, :, :], x[:, 8, :, :], D, dx, dy
            )
            loss_physics = F.mse_loss(pred_dc_dt, phys_target)
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
        )
        val_loss, val_phys = validate(
            model, criterion, val_loader, device, lambda_phys=lambda_physics
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_loss_mn = np.mean(train_loss)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_mn:.4f}, Val Loss: {val_loss:.4f}, Val phys. Loss: {val_phys:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)
    return train_losses, val_losses
