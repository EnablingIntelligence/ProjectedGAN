import torch
import torch.nn as nn
from utils import Conv2DSN
from projection import ProjectionModel
from generator import FastGanGenerator


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

        layers.append(Conv2DSN(in_channels, 1, kernel_size=4, stride=1, padding=0))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_channels, level_channels=[64, 128, 256, 512]):
        """
        MultiDiscriminator builds and holds multiple discriminators.
        input_channels (list of int): List of input channels for each Discriminator.
        level_channels (list of int): List of levels channels for each Discriminator.
        """
        super().__init__()
        levels = [level_channels[i:] for i in range(len(level_channels))]

        self.discriminators = nn.ModuleList(
            [
                Discriminator(input_channels[i], levels[i])
                for i in range(len(input_channels))
            ]
        )

    def forward(self, feats: dict):
        logits = {}
        for disc, feat_idx in zip(self.discriminators, feats):
            feat = feats[feat_idx]
            logit = disc(feat)
            logits[feat_idx] = logit

        return logits


def main():
    Batch_size = 4
    latent_dim = 256
    resolution = 256

    G = FastGanGenerator(resolution)
    projection = ProjectionModel(resolution)
    D = MultiScaleDiscriminator(projection.channels)

    z = torch.randn(Batch_size, latent_dim, 1, 1)
    x_fake = G(z)
    features = projection(x_fake)

    logits = D(features)
    for idx in logits.keys():
        print(logits[idx].shape)


if __name__ == "__main__":
    main()
