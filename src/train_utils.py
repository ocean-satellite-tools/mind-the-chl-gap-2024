import torch
import numpy as np
from tqdm import tqdm


def train_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = []
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    # Close the progress bar
    pbar.close()
    # Return average loss for the epoch
    return train_loss


def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        pbar.close()

    return total_loss / len(val_loader)


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    scheduler=None,
):
    model.train()
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        val_loss = validate(model, criterion, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_loss_mn = np.mean(train_loss)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss_mn:.4f}, Val Loss: {val_loss:.4f}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)
    return train_losses, val_losses
