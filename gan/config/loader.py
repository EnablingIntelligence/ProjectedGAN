import yaml
from omegaconf import OmegaConf

def load_config(path: str) -> OmegaConf:
    return OmegaConf.load(path)
