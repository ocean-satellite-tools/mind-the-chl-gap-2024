import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            self._conv_block(input_channels, 32, kernel_size=3),
            self._conv_block(32, 64),
            self._conv_block(64, 128, kernel_size=3),
            self._conv_block(128, 256),
            nn.Flatten(),
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 176, 240)
            dummy_output = self.conv_layers(dummy_input)
            self.flatten_size = dummy_output.shape[1]

        # self.fc_mu = nn.Sequential(
        #     nn.Linear(self.flatten_size, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, latent_dim),
        #     nn.ReLU(),
        # )
        # self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),  # Add layer normalization
            nn.ReLU(),
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),  # Add layer normalization
            nn.ReLU(),
        )

    def _conv_block(self, in_channels, out_channels, kernel_size=4):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            # nn.LayerNorm([out_channels]),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm([256]),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(256, 256 * 11 * 15),
            nn.LayerNorm([256 * 11 * 15]),  # LayerNorm for stability in decoding
        )
        self.conv_layers = nn.Sequential(
            self._deconv_block(256, 128),
            self._deconv_block(128, 64),
            self._deconv_block(64, 32),
            self._deconv_block(32, output_channels, final_layer=True),
        )

    def _deconv_block(self, in_channels, out_channels, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),  # Adjust as per data normalization
            )
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.LayerNorm([out_channels]),
            nn.LeakyReLU(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 11, 15)
        return self.conv_layers(x)


class ConvVAE(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    bsz = x.shape[0]
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kld = (
        -0.5
        * torch.sum(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / bsz).sum()
    )
    return recon_loss, kld


def train_epoch(model, opt, train_loader, device, kl_lambda=0.5, epoch=0, writer=None):
    model.train()
    pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False)
    rec_losses, kl_losses = [], []
    for batch_idx, (x, y) in enumerate(pbar):
        opt.zero_grad()
        x = x.to(device)
        y = y.to(device)
        xy = torch.cat((x, y.unsqueeze(1)), dim=1)
        xhat, mu, logvar = model(xy)
        rec_loss, kl_loss = vae_loss(xhat, xy, mu, logvar)
        total_loss = rec_loss + kl_lambda * kl_loss
        total_loss.backward()
        opt.step()

        pbar.set_postfix(
            {
                "Loss": f"{total_loss.item():.4f}",
                "rec. loss": f"{rec_loss.item():.4f}",
                "KL loss": f"{kl_loss.item():.4f}",
            }
        )
        rec_losses.append(rec_loss.item())
        kl_losses.append(kl_loss.item())

        if writer is not None:
            writer.add_scalar(
                "Train/Reconstruction_Loss",
                rec_loss.item(),
                epoch * len(train_loader) + batch_idx,
            )
            writer.add_scalar(
                "Train/KL_Loss", kl_loss.item(), epoch * len(train_loader) + batch_idx
            )
            writer.add_scalar(
                "Train/Total_Loss",
                total_loss.item(),
                epoch * len(train_loader) + batch_idx,
            )

    pbar.close()
    return rec_losses, kl_losses


def validate(model, test_loader, device, kl_lambda=0.5, epoch=0, writer=None):
    model.eval()
    total_loss, total_rec_loss, total_kl_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            xy = torch.cat((x, y.unsqueeze(1)), dim=1)
            xhat, mu, logvar = model(xy)
            rec_loss, kl_loss = vae_loss(xhat, xy, mu, logvar)
            loss = rec_loss + kl_lambda * kl_loss

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()

    avg_loss = total_loss / len(test_loader)
    avg_rec_loss = total_rec_loss / len(test_loader)
    avg_kl_loss = total_kl_loss / len(test_loader)

    if writer is not None:
        writer.add_scalar("Validation/Total_Loss", avg_loss, epoch)
        writer.add_scalar("Validation/Reconstruction_Loss", avg_rec_loss, epoch)
        writer.add_scalar("Validation/KL_Loss", avg_kl_loss, epoch)

    return avg_loss, avg_rec_loss, avg_kl_loss


def save_checkpoint(state, file_dir):
    torch.save(state, file_dir + "best_vae.pt")


def train(
    model,
    opt,
    train_loader,
    test_loader,
    device,
    kl_lambda=0.5,
    num_epochs=100,
    show_every=50,
    plot_every=100,
    scheduler=None,
    checkpoint_dir="saved_models",
    checkpoint_patience=20,
    log_dir=None,
):
    model.train()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    my_x, my_y = next(iter(test_loader))
    my_xy = torch.cat((my_x, my_y.unsqueeze(1)), dim=1)
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
        fig_x = plot_channels(my_xy)
        writer.add_figure("Data", fig_x, 0)
    else:
        writer = None

    train_rec_losses, train_kl_losses, val_losses = [], [], []
    for epoch in range(num_epochs):
        rec_losses, kl_losses = train_epoch(
            model,
            opt,
            train_loader,
            device,
            kl_lambda=kl_lambda,
            epoch=epoch,
            writer=writer,
        )
        val_loss, val_rec_loss, val_kl_loss = validate(
            model,
            test_loader,
            device,
            kl_lambda=kl_lambda,
            epoch=epoch,
            writer=writer,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_dir,
            )
            print(f"Saving current best model with val loss = {val_loss}")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % show_every == 0:
            rec_loss_mn = np.mean(rec_losses)
            kl_loss_mn = np.mean(kl_losses)
            val_loss_mn = np.mean(val_losses)
            print(
                f"Epoch {epoch+1}/{num_epochs}: Train loss: {rec_loss_mn:.4f} | {kl_loss_mn:.4f}, Val loss: {val_loss_mn:.4f}"
            )
        if plot_every is not False and (epoch + 1) % plot_every == 0:
            xhat = get_output(model, my_x, my_y, device=device)
            fig_r = plot_channels(xhat)

            if log_dir is not None:
                writer.add_figure("Reconstructions", fig_r, epoch)

        if scheduler is not None:
            scheduler.step(val_loss)

        if epochs_no_improve >= checkpoint_patience:
            print(
                f"Validation loss at epoch {epoch} hasn't improved in {checkpoint_patience} epochs. Resetting to best model from epoch {best_epoch}."
            )
            # Load the best checkpoint
            checkpoint = torch.load(checkpoint_dir + "best_vae.pt")
            model.load_state_dict(checkpoint["state_dict"])
            opt.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            epochs_no_improve = 0  # Reset patience counter

        train_rec_losses.append(rec_losses)
        train_kl_losses.append(kl_losses)
        val_losses.append(val_loss)

    if log_dir is not None:
        writer.close()

    return train_rec_losses, train_kl_losses, val_losses


def get_output(model, x, y, device):
    torch.cuda.empty_cache()
    x = x.to(device)
    y = y.to(device)
    xy = torch.cat((x, y.unsqueeze(1)), dim=1)
    with torch.no_grad():
        xhat, mu, logvar = model(xy)
    return xhat.detach().cpu().numpy()


def plot_channels(x, batch_ind=0, clim=True):
    fig, axs = plt.subplots(3, 3)
    for i, ax in enumerate(axs.ravel()):
        if clim:
            ax.imshow(x[batch_ind, i, :, :], clim=(0, 1))
        else:
            ax.imshow(x[batch_ind, i, :, :])
        ax.set_xticks([])
        ax.set_yticks([])

    # plt.show()
    # return fig
