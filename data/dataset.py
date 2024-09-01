import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor


class CustomDataset(Dataset):

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.img_paths = glob.glob(os.path.join(data_path, "**", "*.png"), recursive=True)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        return PILToTensor()(img)
