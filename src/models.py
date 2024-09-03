import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1, output_fun=F.sigmoid):
        super(UNet, self).__init__()

        # Encoder (downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # Bridge
        self.bridge = DoubleConv(512, 1024)

        # Decoder (upsampling)
        self.dec4 = DoubleConv(1024 + 512, 512)
        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)

        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.output_fun = output_fun

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bridge
        bridge = self.bridge(self.pool(enc4))

        # Decoder
        dec4 = self.dec4(torch.cat([self.upsample(bridge), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))

        return self.output_fun(self.final_conv(dec1))


def init_weights_siren(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        fan_in = m.weight.data.size()[1]
        w_std = 1 / fan_in
        nn.init.uniform_(m.weight, -w_std, w_std)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -torch.pi, torch.pi)


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = Sin(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CHL_CNN(nn.Module):
    def __init__(self):
        super(CHL_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            Sin(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            Sin(inplace=True),
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 88 x 120
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            Sin(inplace=True),
            ResNetBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            Sin(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 44 x 60
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            Sin(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Sin(inplace=True),
        )

        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # Output: 128 x 88 x 120
            Sin(inplace=True),
            ResNetBlock(128, 128),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # Output: 64 x 176 x 240
            Sin(inplace=True),
        )

        self.final_conv = nn.Conv2d(
            64, 1, kernel_size=3, padding=1
        )  # Output: 1 x 176 x 240

    def forward(self, x):
        x = self.features(x)
        x = self.upsampler(x)
        x = self.final_conv(x)
        return x


## ===== DeepONet


class ConvBranchNet(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_basis_functions, image_size):
        super(ConvBranchNet, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_basis_functions = num_basis_functions
        self.image_size = image_size

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            self._conv_block(input_channels, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
        )
        # Calculate the size of the flattened features after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *image_size)
            conv_output = self.conv_layers(dummy_input)
            self.flat_features = conv_output.view(1, -1).size(1)

        # Fully connected layers for basis functions
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_basis_functions),
        )

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # Convolutional feature extraction
        conv_features = self.conv_layers(x)
        flat_features = conv_features.view(x.size(0), -1)
        basis_functions = self.fc_layers(flat_features)
        return basis_functions


class TrunkNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TrunkNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DeepONet(nn.Module):
    def __init__(
        self,
        input_channels,
        trunk_input_dim,
        hidden_dim,
        num_basis_functions,
        image_size,
    ):
        super(DeepONet, self).__init__()
        self.branch = ConvBranchNet(
            input_channels, hidden_dim, num_basis_functions, image_size
        )
        self.trunk = TrunkNet(trunk_input_dim, num_basis_functions)

    def forward(self, u, y):
        b = self.branch(u)  # Shape: (batch_size, num_basis_functions)
        t = self.trunk(y)  # Shape: (batch_size, num_sensors, num_basis_functions)
        return torch.sum(b.unsqueeze(1) * t, dim=-1)  # Shape: (batch_size, num_sensors)


# class DeepONet(nn.Module):
#     def __init__(
#         self, input_channels, trunk_input_dim, hidden_dim, output_dim, image_size
#     ):
#         super(DeepONet, self).__init__()
#         self.branch = ConvBranchNet(input_channels, hidden_dim, output_dim, image_size)
#         self.trunk = TrunkNet(trunk_input_dim, output_dim)

#     def forward(self, u, y):
#         b = self.branch(u)
#         t = self.trunk(y)
#         s = torch.sum(torch.sum(b * t.unsqueeze(2), dim=-1), dim=-1)
#         return F.sigmoid(s)
