from typing import Dict
from omegaconf import DictConfig

from torchtext import data, datasets
import pytorch_lightning as pl

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        raise NotImplementedError
