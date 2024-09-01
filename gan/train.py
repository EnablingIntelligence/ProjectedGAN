import argparse
import os
import time
from argparse import Namespace
from typing import Union

import torch
from torch.optim import Adam
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from data import get_data_loader
from gan.config import load_config
from gan.model import FastGanGenerator, MultiScaleDiscriminator, ProjectionModel
from gan.utils import HingeLossG, HingeLossD


def train(args: Union[Namespace, dict]):
    config = load_config(args.config)

    current_time_in_millis = int(round(time.time() * 1000))
    run_id = f"gan_run_{current_time_in_millis}"

    writer = SummaryWriter(os.path.join(config.general.result_path, run_id))

    G_cfg = config.generator
    D_cfg = config.discriminator
    train_cfg = config.training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader = get_data_loader(
        data_path=config.general.data_path,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        device=device,
        resolution=train_cfg.resolution,
        img_file_type=config.general.img_file_type,
    )

    G = FastGanGenerator(
        ngf=G_cfg.ngf,
        z_dim=G_cfg.z_dim,
        out_ch=train_cfg.num_channels,
        resolution=train_cfg.resolution,
    ).to(device)

    P = ProjectionModel(resolution=train_cfg.resolution).to(device)

    D = MultiScaleDiscriminator(feature_channels=P.channels).to(device)

    optim_D = Adam(
        params=D.parameters(),
        lr=D_cfg.optimizer.lr,
        betas=(D_cfg.optimizer.beta1, D_cfg.optimizer.beta2),
    )

    optim_G = Adam(
        params=G.parameters(),
        lr=G_cfg.optimizer.lr,
        betas=(G_cfg.optimizer.beta1, G_cfg.optimizer.beta2),
    )

    loss_G = HingeLossG()
    loss_D = HingeLossD()

    z_benchmark = torch.randn(
        train_cfg.logging.batch_size, train_cfg.latent_dim, 1, 1, device=device
    )

    n_epoch = 1
    for epoch in range(1, train_cfg.num_epochs + 1):
        for real_images in dataloader:
            # Discriminator training
            z = torch.randn(
                train_cfg.batch_size, train_cfg.latent_dim, 1, 1, device=device
            )
            # Detach to avoid backpropagation through the generator
            fake_images = G(z).detach()

            features_fake = P(fake_images)
            logits_fake = D(features_fake)

            features_real = P(real_images)
            logits_real = D(features_real)

            # Compute loss for Discriminator
            total_D_loss = 0.0
            for logit_real, logit_fake in zip(
                    logits_real.values(), logits_fake.values()
            ):
                loss_disc = loss_D(logit_real, logit_fake)
                total_D_loss += loss_disc

            optim_D.zero_grad()
            total_D_loss.backward()
            optim_D.step()

            # For the generator training
            z = torch.randn(
                train_cfg.batch_size, train_cfg.latent_dim, 1, 1, device=device
            )
            fake_images = G(z)
            features_fake = P(fake_images)
            logits_fake = D(features_fake)

            total_G_loss = 0.0
            for logit in logits_fake.values():
                loss_gen = loss_G(logit)
                total_G_loss += loss_gen

            optim_G.zero_grad()
            total_G_loss.backward()
            optim_G.step()

            writer.add_scalar(
                tag="DiscriminatorLoss",
                scalar_value=total_D_loss.cpu().item(),
                global_step=n_epoch,
            )
            writer.add_scalar(
                tag="GeneratorLoss",
                scalar_value=total_G_loss.cpu().item(),
                global_step=n_epoch,
            )

            n_epoch += 1

        if epoch % train_cfg.logging.interval == 0:
            
            # gen images
            fake_images = G(z_benchmark)
            image_grid = make_grid(fake_images, normalize=True)
            writer.add_image(tag="GenIamge", img_tensor=image_grid, global_step=epoch)

        print(f"Epoch {epoch} - DLoss: {total_D_loss.cpu().detach().numpy():.03f}")
        print(f"Epoch {epoch} - GLoss: {total_G_loss.cpu().detach().numpy():.03f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for training the Projected GAN"
    )
    parser.add_argument(
        "--config", type=str, default="./gan/config/cfg.yml", help="Path to the config file"
    )
    args = parser.parse_args()
    train(args)
