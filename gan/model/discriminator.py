import torch
import torch.nn as nn

from gan.utils import Conv2DSN


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = nn.Sequential(
            Conv2DSN(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class MinibatchStdLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, H, W) = x.size()
        std = torch.std(x, dim=0, keepdim=True)
        mean_std = torch.mean(std).expand((B, 1, H, W))
        return torch.cat([x, mean_std], dim=1)


class Discriminator(nn.Module):
    def __init__(self, in_channels, levels):
        """
        in_channels (int): Number of input channels for the first layer.
        levels (list of int): Number of output channels for each DownBlock.
        """
        super().__init__()

        layers = []
        for out_channels in levels:
            layers.append(DownBlock(in_channels, out_channels))
            in_channels = out_channels

        layers.append(MinibatchStdLayer())
        layers.append(Conv2DSN(in_channels + 1, 1, kernel_size=4, stride=1, padding=0))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, feature_channels: list[int], level_channels=[64, 128, 256, 512]):
        """
        MultiDiscriminator builds and holds multiple discriminators.
        feature_channels (list of int): List of feature channels for each Discriminator.
        level_channels (list of int): List of levels channels for each Discriminator.
        """
        super().__init__()
        levels = [level_channels[i:] for i in range(len(level_channels))]

        self.discriminators = nn.ModuleList(
            [
                Discriminator(feature_channels[i], levels[i])
                for i in range(len(feature_channels))
            ]
        )

    def forward(self, feats: dict):
        logits = {}
        for disc, feat_idx in zip(self.discriminators, feats):
            feat = feats[feat_idx]
            logit = disc(feat)
            logits[feat_idx] = logit

        return logits

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
