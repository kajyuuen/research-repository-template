import hydra
from hydra import utils
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

import src.models

@hydra.main(config_path="./configs/config.yaml")
def main(cfg: DictConfig) -> None:
    # logger = TensorBoardLogger(utils.to_absolute_path("tb_logs"), name = cfg.name)
    logger = WandbLogger(name = cfg.wandb.name,
                         project = cfg.name,
                         offline = cfg.wandb.offline)
    logger.log_hyperparams(cfg)

    trainer = pl.Trainer(logger)
    model = BaseModel(cfg)
    trainer.fit(model)

if __name__ == "__main__":
    main()
