"""
FastGan: https://arxiv.org/pdf/2101.04775.pdf
ProjGan: https://arxiv.org/pdf/2111.01007
"""

import torch
from torch import nn

from gan.utils import Conv2DSN, ConvTranspose2DSN


class UpSampling(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Conv2DSN(in_ch, out_ch * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            nn.GLU(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class InputLayer(nn.Module):
    def __init__(self, in_ch: int = 256, out_ch: int = 3) -> None:
        super().__init__()
        self.model = nn.Sequential(
            ConvTranspose2DSN(in_ch, out_ch * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch * 2),
            nn.GLU(1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class SkipLayerExcitation(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        multiplier: int = 2
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            Conv2DSN(
                in_channels=in_ch,
                out_channels=out_ch * multiplier,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.1),
            Conv2DSN(
                in_channels=out_ch * multiplier,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x_high: torch.Tensor, x_low: torch.Tensor):
        return x_high * self.model(x_low)


class FastGanGenerator(nn.Module):
    """
    FastGAN generator designed to create images with dimensions 3x256x256
    """

    def __init__(self, ngf=64, z_dim=256, out_ch=3, resolution=256) -> None:
        super().__init__()
        self.nfc = self.get_feat_channels(ngf)
        self.resolution = resolution
        self.layer_layer_ch = self.nfc[resolution]

        self.init = InputLayer(z_dim, self.nfc[4])
        self.feat_layers = {
            8: UpSampling(self.nfc[4], self.nfc[8]),
            16: UpSampling(self.nfc[8], self.nfc[16]),
            32: UpSampling(self.nfc[16], self.nfc[32]),
            64: UpSampling(self.nfc[32], self.nfc[64]),
            128: UpSampling(self.nfc[64], self.nfc[128]),
            256: UpSampling(self.nfc[128], self.nfc[256]),
        }

        self.sle_layers = {
            128: SkipLayerExcitation(in_ch=self.nfc[8], out_ch=self.nfc[128]),
            256: SkipLayerExcitation(in_ch=self.nfc[16], out_ch=self.nfc[256]),
        }

        self.last_layer = nn.Sequential(
            nn.Conv2d(self.layer_layer_ch, out_ch, 3, 1, 1),
            nn.Tanh(),
        )

    def get_feat_channels(self, ngf):
        nfc_multiplier = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25}
        nfc = {k: int(v * ngf) for k, v in nfc_multiplier.items()}
        return nfc

    def forward(self, z: torch.Tensor):
        x = self.init(z)
        feat8 = self.feat_layers[8](x)
        feat16 = self.feat_layers[16](feat8)
        feat32 = self.feat_layers[32](feat16)
        feat64 = self.feat_layers[64](feat32)
        feat128 = self.feat_layers[128](feat64)
        feat128 = self.sle_layers[128](feat128, feat8)

        if self.resolution > 128:
            feat256 = self.feat256(feat128)
            feat = self.sle_layers[256](feat256, feat16)
        else:
            feat = feat128

        feat = self.last_layer(feat)
        return feat
