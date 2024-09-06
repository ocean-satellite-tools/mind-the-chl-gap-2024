import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            self._conv_block(input_channels, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            nn.Flatten(),
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 176, 240)
            dummy_output = self.conv_layers(dummy_input)
            self.flatten_size = dummy_output.shape[1]

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
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
        self.fc = nn.Linear(latent_dim, 256 * 11 * 15)
        self.conv_layers = nn.Sequential(
            self._deconv_block(256, 128),
            self._deconv_block(128, 64),
            self._deconv_block(64, 32),
            self._deconv_block(32, output_channels, final_layer=True),
        )

    def _deconv_block(self, in_channels, out_channels, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.ReLU(),  # Adjust if your data is not normalized to [-1, 1]
            )
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
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
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld


def train_epoch(model, opt, train_loader, device, kl_lambda=0.5, epoch=0):
    model.train()
    pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False)
    rec_losses, kl_losses = [], []
    for x, y in pbar:
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
    pbar.close()
    return rec_losses, kl_losses


def validate(model, test_loader, device, kl_lambda=0.5):
    model.eval()
    L = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            xy = torch.cat((x, y.unsqueeze(1)), dim=1)
            xhat, mu, logvar = model(xy)
            rec_loss, kl_loss = vae_loss(xhat, xy, mu, logvar)
            total_loss = rec_loss + kl_lambda * kl_loss
            L += total_loss.item()

    return L / len(test_loader)


def train(
    model,
    opt,
    train_loader,
    test_loader,
    device,
    kl_lambda=0.5,
    num_epochs=100,
    show_every=50,
    scheduler=None,
):
    model.train()
    train_rec_losses, train_kl_losses, val_losses = [], [], []
    for epoch in range(num_epochs):
        rec_losses, kl_losses = train_epoch(
            model, opt, train_loader, device, kl_lambda=kl_lambda, epoch=epoch
        )
        val_loss = validate(model, test_loader, device, kl_lambda=kl_lambda)

        train_rec_losses.append(rec_losses)
        train_kl_losses.append(kl_losses)
        val_losses.append(val_loss)

        if (epoch + 1) % show_every == 0:
            rec_loss_mn = np.mean(rec_losses)
            kl_loss_mn = np.mean(kl_losses)
            val_loss_mn = np.mean(val_losses)
            print(
                f"Epoch {epoch+1}/{num_epochs}: Train loss: {rec_loss_mn:.4f} | {kl_loss_mn:.4f}, Val loss: {val_loss_mn:.4f}"
            )

        if scheduler is not None:
            scheduler.step(val_loss)

    return train_rec_losses, train_kl_losses, val_losses
