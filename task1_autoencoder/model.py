"""
Convolutional Autoencoders for 128x128 jet images.

ConvAutoencoder (default):
  Encoder: 128 → 64 → 32 → 16 → 8 (strided conv)
  Bottleneck: FC → latent_dim → FC
  Decoder: 8 → 16 → 32 → 64 → 128 (upsample + conv)

ShallowAutoencoder (fewer downsamples for finer spatial detail):
  Encoder: 128 → 64 → 32 → 16 (3x strided conv)
  Bottleneck: 1x1 conv channel compression (no FC)
  Decoder: 16 → 32 → 64 → 128 (upsample + conv)
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):

    def __init__(self, in_channels=3, latent_dim=128, channels=None, output_act="relu"):
        super().__init__()
        ch = channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, ch[0]),
            self._conv_block(ch[0], ch[1]),
            self._conv_block(ch[1], ch[2]),
            self._conv_block(ch[2], ch[3]),
        )

        self.flatten_size = ch[3] * 8 * 8
        self.fc_encode = nn.Sequential(
            nn.Linear(self.flatten_size, latent_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_size),
            nn.LeakyReLU(0.2),
        )

        # Output activation
        if output_act == "sigmoid":
            out_act = nn.Sigmoid()
        elif output_act == "softplus":
            out_act = nn.Softplus()
        else:
            out_act = nn.ReLU()

        # Decoder (upsample + conv to avoid checkerboard artifacts)
        self._decoder_channels = ch
        self.decoder = nn.Sequential(
            self._upsample_block(ch[3], ch[2]),
            self._upsample_block(ch[2], ch[1]),
            self._upsample_block(ch[1], ch[0]),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch[0], in_channels, kernel_size=3, padding=1),
            out_act,
        )

    @staticmethod
    def _conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    @staticmethod
    def _upsample_block(in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_encode(h)

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), self._decoder_channels[-1], 8, 8)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SegNetAutoencoder(nn.Module):
    """SegNet-style AE: MaxPool(indices) encoder + MaxUnpool decoder, FC bottleneck.

    3x downsample (128->16), max pooling indices preserve spatial positions
    of dominant features for precise reconstruction of sparse signals.
    """

    def __init__(self, in_channels=3, latent_dim=512, channels=None, output_act="relu"):
        super().__init__()
        ch = channels or [32, 64, 128]
        self.latent_dim = latent_dim
        self.output_act = output_act

        # Encoder conv blocks (no stride, pooling is separate)
        self.enc1 = self._conv_block(in_channels, ch[0])
        self.enc2 = self._conv_block(ch[0], ch[1])
        self.enc3 = self._conv_block(ch[1], ch[2])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # FC bottleneck
        self.flatten_size = ch[2] * 16 * 16
        self.fc_encode = nn.Sequential(
            nn.Linear(self.flatten_size, latent_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_size),
            nn.LeakyReLU(0.2),
        )

        # Decoder
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3 = self._conv_block(ch[2], ch[1])
        self.dec2 = self._conv_block(ch[1], ch[0])
        self.dec1_conv = nn.Conv2d(ch[0], in_channels, kernel_size=3, padding=1)

        self._decoder_channels = ch

    @staticmethod
    def _conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    def encode(self, x):
        x = self.enc1(x)
        x, self.idx1 = self.pool(x)  # 128->64
        x = self.enc2(x)
        x, self.idx2 = self.pool(x)  # 64->32
        x = self.enc3(x)
        x, self.idx3 = self.pool(x)  # 32->16
        x = x.view(x.size(0), -1)
        return self.fc_encode(x)

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), self._decoder_channels[2], 16, 16)
        x = self.unpool(x, self.idx3)  # 16->32
        x = self.dec3(x)
        x = self.unpool(x, self.idx2)  # 32->64
        x = self.dec2(x)
        x = self.unpool(x, self.idx1)  # 64->128
        x = self.dec1_conv(x)
        if self.output_act == "sigmoid":
            x = torch.sigmoid(x)
        else:
            x = torch.relu(x)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ShallowAutoencoder(nn.Module):
    """3x downsampling (128->16), 1x1 conv bottleneck, preserves spatial structure."""

    def __init__(self, in_channels=3, bottleneck_ch=4, channels=None):
        super().__init__()
        ch = channels or [32, 64, 128]
        self.bottleneck_ch = bottleneck_ch

        # Encoder: 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, ch[0]),
            self._conv_block(ch[0], ch[1]),
            self._conv_block(ch[1], ch[2]),
        )

        # Bottleneck: 1x1 conv to compress channels
        self.compress = nn.Sequential(
            nn.Conv2d(ch[2], bottleneck_ch, kernel_size=1),
            nn.LeakyReLU(0.2),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck_ch, ch[2], kernel_size=1),
            nn.LeakyReLU(0.2),
        )

        # Decoder: 16 -> 32 -> 64 -> 128
        self.decoder = nn.Sequential(
            self._upsample_block(ch[2], ch[1]),
            self._upsample_block(ch[1], ch[0]),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch[0], in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    @staticmethod
    def _conv_block(in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    @staticmethod
    def _upsample_block(in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    def encode(self, x):
        return self.compress(self.encoder(x))

    def decode(self, z):
        return self.decoder(self.expand(z))

    def forward(self, x):
        return self.decode(self.encode(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
