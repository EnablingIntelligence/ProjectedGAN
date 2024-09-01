import torch
from torch.utils.data import DataLoader
from data.dataset import CustomDataset

def get_data_loader(data_path: str, batch_size: int, num_workers: int, device: torch.device, img_file_type) -> DataLoader:
    dataset = CustomDataset(data_path=data_path, device=device, img_file_type=img_file_type)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)