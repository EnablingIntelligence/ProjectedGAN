"""
GAN Hinge Loss: https://paperswithcode.com/method/gan-hinge-loss
"""

import torch
import torch.nn.functional as F
from torch import nn


class HingeLossD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_logits, fake_logits):
        real_loss = torch.mean(F.relu(1.0 - real_logits))
        fake_loss = torch.mean(F.relu(1.0 + fake_logits))
        d_loss = real_loss + fake_loss
        return d_loss


class HingeLossG(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_logits):
        return -torch.mean(fake_logits)
