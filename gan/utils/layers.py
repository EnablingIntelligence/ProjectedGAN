from torch import nn
from torch.nn.utils import spectral_norm


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
