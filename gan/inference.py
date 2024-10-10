import torch
from gan.model import FastGanGenerator
from PIL import Image
import numpy as np


class ImageGenerator:
    def __init__(self, ckpt_path: str, ngf=128, z_dim=100, out_ch=3, resolution=256):
        self.z_dim = z_dim

        self.generator = FastGanGenerator(ngf, z_dim, out_ch, resolution).cpu()
        self.generator.load(ckpt_path)

    @staticmethod
    def transform(image_batch: np.ndarray):
        range = image_batch.max() - image_batch.min()
        norm_image = (image_batch - image_batch.min()) / range
        image = (norm_image * 255).astype(np.uint8)[0]
        return np.transpose(image, (1, 2, 0))

    def generate(self):
        z = torch.randn(1, self.z_dim, 1, 1)
        image_batch = self.generator(z).detach().numpy()
        image = self.transform(image_batch)
        return Image.fromarray(image)
