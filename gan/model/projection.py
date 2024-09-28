import torch
from torchvision import models
from torch import nn

from gan.model import DiffAugment
from gan.utils import kaiming_init


class EfficientNet(nn.Module):
    """
    This class represents a model using 4 layers from the pretrained EfficientNet BaseLine1 network.
    The forward method returns a dictionary containing the features extracted from each corresponding layer.
    """

    def __init__(self, resolution: int = 256):
        super().__init__()
        self.model = self.load_model()
        self.names = ["layer0", "layer1", "layer2", "layer3"]
        self.indices = [3, 4, 6, 8]
        self.layers = self.get_layers(self.model)
        self.resolutions = self.get_feat_resolutions(resolution)
        self.channels = self.get_feat_channels(resolution)
        self.freeze_parameters()

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def load_model(self):
        return models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)

    def get_layers(self, model):
        layers = nn.Module()
        start_idxs = [0] + self.indices[:-1]
        end_idxs = self.indices
        for i, (start, end) in enumerate(zip(start_idxs, end_idxs)):
            layer = nn.Sequential(*model.features[start:end])
            setattr(layers, self.names[i], layer)

        return layers

    def get_feat_resolutions(self, resolution: int):
        return [resolution // 4, resolution // 8, resolution // 16, resolution // 32]

    def get_feat_channels(self, resolution: int):
        channels = []
        tmp = torch.zeros(1, 3, resolution, resolution)
        for layer_name in self.names:
            layer = getattr(self.layers, layer_name)
            tmp = layer(tmp)
            channels.append(tmp.shape[1])
        return channels

    def forward(self, x):
        features = {}

        x = DiffAugment(x, policy="color,translation,cutout")

        input_tensor = x

        for i, layer_name in enumerate(self.names):
            layer = getattr(self.layers, layer_name)
            input_tensor = layer(input_tensor)
            features[i] = input_tensor

        return features


class RandomProjection(nn.Module):
    """
    CSM + CCM
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.freeze_parameters()

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.apply(kaiming_init)

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, high_res, low_res=None):
        high_res = self.conv1(high_res)
        if low_res is not None:
            high_res += high_res + low_res
        high_res = self.conv2(high_res)
        high_res = self.upsample(high_res)
        return high_res


class ProjectionModel(nn.Module):
    """
    EfficientNet + Random Projection.
    """

    def __init__(self, resolution: int = 256):
        super().__init__()

        self.pretrained_model = EfficientNet(resolution)
        channels = self.pretrained_model.channels

        self.csm320 = RandomProjection(channels[3], channels[2])
        self.csm112 = RandomProjection(channels[2], channels[1])
        self.csm40 = RandomProjection(channels[1], channels[0])
        self.csm24 = RandomProjection(channels[0], channels[0])

        self.channels = self.get_channels(resolution)

    def get_channels(self, resolution: int):
        tmp = torch.zeros(1, 3, resolution, resolution)
        tmp_feat = self.forward(tmp)

        return [
            self.get_ch(tmp_feat[0]),
            self.get_ch(tmp_feat[1]),
            self.get_ch(tmp_feat[2]),
            self.get_ch(tmp_feat[3]),
        ]

    def forward(self, x):
        features = self.pretrained_model(x)

        feat16 = self.csm320(features[3])
        feat32 = self.csm112(features[2], feat16)
        feat64 = self.csm40(features[1], feat32)
        feat128 = self.csm24(features[0], feat64)

        return {0: feat128, 1: feat64, 2: feat32, 3: feat16}

    def get_ch(self, feat):
        return feat.shape[1]


def print_feat_shape(features):
    for idx in features.keys():
        print(features[idx].shape)
    print()
