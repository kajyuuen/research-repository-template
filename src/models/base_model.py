from typing import Dict
from omegaconf import DictConfig

from torchtext import data, datasets
import pytorch_lightning as pl

from src.modules import BaseModule

class BaseModel(pl.LightningModule):
    def __init__(self,
                 cfg: DictConfig):
        super().__init__()

        self.modules = BaseModule()
    
    def forward(self):
        raise NotImplementedError

    def _prepare_datasets(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx) -> Dict:
        raise NotImplementedError

    @pl.data_loader
    def train_dataloader(self) -> data.Iterator:
        raise NotImplementedError