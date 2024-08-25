"""
GAN Hinge Loss: https://paperswithcode.com/method/gan-hinge-loss
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from generator import FastGanGenerator
from discriminator import MultiScaleDiscriminator
from projection import ProjectionModel

import warnings

warnings.simplefilter("ignore", UserWarning)


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
        g_loss = -torch.mean(fake_logits)
        return g_loss


def main():
    C = 3
    Batch_size = 4
    latent_dim = 256
    resolution = 256
    lr = 0.0002
    num_epochs = 1

    lossG = HingeLossG()
    lossD = HingeLossD()

    G = FastGanGenerator(resolution)
    projection = ProjectionModel(resolution)
    D = MultiScaleDiscriminator(projection.channels)

    optimD = Adam(D.parameters(), lr)
    optimG = Adam(G.parameters(), lr)

    for epoch in range(num_epochs):
        # dummy real sample
        x_real = z = torch.randn(Batch_size, C, resolution, resolution)

        # For the discriminator training
        z = torch.randn(Batch_size, latent_dim, 1, 1)
        x_fake = G(z).detach()  # Detach to avoid backpropagation through the generator

        features_fake = projection(x_fake)
        logits_fake = D(features_fake)

        features_real = projection(x_real)
        logits_real = D(features_real)

        # Compute loss for Discriminator
        total_d_loss = 0.0
        for logit_real, logit_fake in zip(logits_real.values(), logits_fake.values()):
            loss_disc = lossD(logit_real, logit_fake)
            total_d_loss += loss_disc

        optimD.zero_grad()
        total_d_loss.backward()
        optimD.step()

        # For the generator training
        z = torch.randn(Batch_size, latent_dim, 1, 1)
        x_fake = G(z)
        features_fake = projection(x_fake)
        logits_fake = D(features_fake)

        total_g_loss = 0.0
        for logit in logits_fake.values():
            loss_gen = lossG(logit)
            total_g_loss += loss_gen

        optimG.zero_grad()
        total_g_loss.backward()
        optimG.step()

        print(f"Epoch {epoch} - DLoss: {total_d_loss.cpu().detach().numpy():.03f}")
        print(f"Epoch {epoch} - GLoss: {total_g_loss.cpu().detach().numpy():.03f}")


if __name__ == "__main__":
    main()
