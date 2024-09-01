import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, Compose, Normalize, ToTensor, Resize


class CustomDataset(Dataset):

    def __init__(
        self, data_path: str, device: torch.device, resolution: int, img_file_type: str = "jpg"
    ):
        self.device = device
        self.data_path = data_path
        self.img_paths = glob.glob(
            os.path.join(data_path, "**", f"*.{img_file_type}"), recursive=True
        )

        self.transform = Compose(
            [
                Resize(resolution),
                ToTensor(),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        return self.transform(img).to(self.device)
