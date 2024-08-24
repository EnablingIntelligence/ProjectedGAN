import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def kaiming_init(m):
    """Initialize Conv2d layers using Kaiming normal initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Conv2DSN(nn.Module):
    """
    Create a Conv2d layer with the given arguments and apply spectral normalization

    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(*args, **kwargs))

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2DSN(nn.Module):
    """
    Create a ConvTranspose2d layer with the given arguments and apply spectral normalization
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

    def forward(self, x):
        return self.conv_transpose(x)
