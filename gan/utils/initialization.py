from torch import nn


def kaiming_init(module: nn.Module):
    """Initialize Conv2d layers using Kaiming normal initialization."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def weights_init(module: nn.Module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        module.weight.data.normal_(0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
