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
        real_loss = torch.mean(F.relu(1.0 - real_logits)) # D(x)
        fake_loss = torch.mean(F.relu(1.0 + fake_logits)) # D(G(z))
        d_loss = real_loss + fake_loss
        return d_loss


class HingeLossG(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_logits):
        g_loss = -torch.mean(fake_logits) # D(G(z))
        return g_loss


def main():
    C = 3
    batch_size = 4
    latent_dim = 256
    resolution = 256
    lr = 0.0002
    num_epochs = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    loss_G = HingeLossG()
    loss_D = HingeLossD()

    G = FastGanGenerator(resolution).to(device)
    projection = ProjectionModel(resolution).to(device)
    D = MultiScaleDiscriminator(projection.channels).to(device)

    
    optim_D = Adam(D.parameters(), lr)
    optim_G = Adam(G.parameters(), lr)

    for epoch in range(num_epochs):
        # dummy real sample
        x_real = z = torch.randn(batch_size, C, resolution, resolution, device=device)

        # For the discriminator training
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        x_fake = G(z).detach()  # Detach to avoid backpropagation through the generator

        features_fake = projection(x_fake)
        logits_fake = D(features_fake)

        features_real = projection(x_real)
        logits_real = D(features_real)

        # Compute loss for Discriminator
        total_D_loss = 0.0
        for logit_real, logit_fake in zip(logits_real.values(), logits_fake.values()):
            loss_disc = loss_D(logit_real, logit_fake)
            total_D_loss += loss_disc

        optim_D.zero_grad()
        total_D_loss.backward()
        optim_D.step()

        # For the generator training
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        x_fake = G(z)
        features_fake = projection(x_fake)
        logits_fake = D(features_fake)

        total_G_loss = 0.0
        for logit in logits_fake.values():
            loss_gen = loss_G(logit)
            total_G_loss += loss_gen

        optim_G.zero_grad()
        total_G_loss.backward()
        optim_G.step()

        print(f"Epoch {epoch} - DLoss: {total_D_loss.cpu().detach().numpy():.03f}")
        print(f"Epoch {epoch} - GLoss: {total_G_loss.cpu().detach().numpy():.03f}")


if __name__ == "__main__":
    main()
