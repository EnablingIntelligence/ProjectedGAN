import argparse
import os
import time
from argparse import Namespace
from typing import Union

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from data import CustomDataset
from gan.config import load_config
from gan.model import FastGanGenerator, MultiScaleDiscriminator, ProjectionModel
from gan.utils import HingeLossG, HingeLossD


def train(args: Union[Namespace, dict]):
    config = load_config(args.config)
    general_cfg = config.general

    current_time_in_millis = int(round(time.time() * 1000))
    run_id = f"gan_run_{current_time_in_millis}"

    run_path = os.path.join(general_cfg.result_path, run_id)
    writer = SummaryWriter(run_path)

    g_cfg = config.generator
    d_cfg = config.discriminator
    train_cfg = config.training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CustomDataset(
        data_path=general_cfg.data_path,
        device=device,
        resolution=train_cfg.resolution,
        img_file_type=general_cfg.img_file_type,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
    )

    generator = FastGanGenerator(
        ngf=g_cfg.ngf,
        z_dim=g_cfg.z_dim,
        out_ch=train_cfg.num_channels,
        resolution=train_cfg.resolution,
    ).to(device)

    projection = ProjectionModel(resolution=train_cfg.resolution).to(device)

    discriminator = MultiScaleDiscriminator(feature_channels=projection.channels).to(
        device
    )

    checkpoint_cfg = general_cfg.checkpoint

    if checkpoint_cfg.load_checkpoint:
        print("Loading checkpoints")
        generator.load(checkpoint_cfg.generator_path)
        discriminator.load(checkpoint_cfg.discriminator_path)

    optim_d = Adam(
        params=discriminator.parameters(),
        lr=d_cfg.optimizer.lr,
        betas=(d_cfg.optimizer.beta1, d_cfg.optimizer.beta2),
    )

    optim_g = Adam(
        params=generator.parameters(),
        lr=g_cfg.optimizer.lr,
        betas=(g_cfg.optimizer.beta1, g_cfg.optimizer.beta2),
    )

    loss_g = HingeLossG()
    loss_d = HingeLossD()

    z_benchmark = torch.randn(
        train_cfg.logging.batch_size, g_cfg.z_dim, 1, 1, device=device
    )

    n_epoch = 1
    for epoch in range(1, train_cfg.num_epochs + 1):
        for real_images in dataloader:
            real_images = real_images.to(device)
            # Discriminator training
            z = torch.randn(
                train_cfg.batch_size, g_cfg.z_dim, 1, 1, device=device
            )
            # Detach to avoid backpropagation through the generator
            fake_images = generator(z).detach()

            features_fake = projection(fake_images)
            logits_fake = discriminator(features_fake)

            features_real = projection(real_images)
            logits_real = discriminator(features_real)

            # Compute loss for Discriminator
            total_d_loss = 0.0
            for logit_real, logit_fake in zip(
                    logits_real.values(), logits_fake.values()
            ):
                loss_disc = loss_d(logit_real, logit_fake)
                total_d_loss += loss_disc

            optim_d.zero_grad()
            total_d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
            optim_d.step()

            # For the generator training
            z = torch.randn(
                train_cfg.batch_size, g_cfg.z_dim, 1, 1, device=device
            )
            fake_images = generator(z)
            features_fake = projection(fake_images)
            logits_fake = discriminator(features_fake)

            total_g_loss = 0.0
            for logit in logits_fake.values():
                loss_gen = loss_g(logit)
                total_g_loss += loss_gen

            optim_g.zero_grad()
            total_g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
            optim_g.step()

            writer.add_scalar(
                tag="DiscriminatorLoss",
                scalar_value=total_d_loss.cpu().item(),
                global_step=n_epoch,
            )
            writer.add_scalar(
                tag="GeneratorLoss",
                scalar_value=total_g_loss.cpu().item(),
                global_step=n_epoch,
            )

            n_epoch += 1

        if epoch % train_cfg.logging.interval == 0:
            fake_images = generator(z_benchmark)
            image_grid = make_grid(fake_images, normalize=True)
            writer.add_image(tag="GenIamge", img_tensor=image_grid, global_step=epoch)

        if epoch % train_cfg.checkpoint.interval == 0:
            current_time_in_millis = int(round(time.time() * 1000))
            generator.save(os.path.join(run_path, f"G_{current_time_in_millis}.pth"))
            discriminator.save(os.path.join(run_path, f"D_{current_time_in_millis}.pth"))

        print(f"Epoch {epoch} - DLoss: {total_d_loss.cpu().detach().numpy():.03f}")
        print(f"Epoch {epoch} - GLoss: {total_g_loss.cpu().detach().numpy():.03f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for training the Projected GAN"
    )
    parser.add_argument(
        "--config", type=str, default="./config/cfg.yml", help="Path to the config file"
    )
    args = parser.parse_args()
    train(args)
