import torch
from gan.model import FastGanGenerator
from PIL import Image
import numpy as np


class ImageGenerator:
    def __init__(self, ckpt_path: str):
        self.z_dim = 100

        self.generator = FastGanGenerator(
            ngf=128, z_dim=self.z_dim, out_ch=3, resolution=256
        ).cpu()
        self.generator.load(ckpt_path)

    @staticmethod
    def transform(image_batch: np.ndarray):
        image_range = image_batch.max() - image_batch.min()
        norm_image = (image_batch - image_batch.min()) / image_range
        image = (norm_image * 255).astype(np.uint8)[0]
        return np.transpose(image, (1, 2, 0))

    def generate(self):
        z = torch.randn(1, self.z_dim, 1, 1)
        image_batch = self.generator(z).detach().numpy()
        image = self.transform(image_batch)
        return Image.fromarray(image)
