import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from ensembles.tasks import GraphicsINR
from ensembles.datasets import emb_loader
from ensembles.archs import ImplicitNeuralModule


def main():
    parser = argparse.ArgumentParser(description='Train lvl 1 graphics')
    parser.add_argument('--config', type = str, default = 'graphics_inr',
                        help =  'path to the config file')

    args = parser.parse_args()
    with open(f"/project/scripts/configs/{args.config}.yaml", 'r') as file:
        config = yaml.safe_load(file)

    arch_name = config['arch']
    logger = CSVLogger(save_dir=config['logging_params']['save_dir'],
                       name= f'lvl1_graphics_{arch_name}')

    # For reproducibility
    seed_everything(12345, True)

    arch = ImplicitNeuralModule(**config['arch_params'])
    arch.train()
    task = GraphicsINR(arch, **config['exp_params'])
    runner = Trainer(logger=logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2,
                                         dirpath =os.path.join(logger.log_dir , "checkpoints"),
                                         monitor= "val_loss",
                                         save_last=True),
                     ],
                     accelerator = 'auto',
                     deterministic = True,
                     **config['trainer_params'])

    device = runner.device_ids[0] if torch.cuda.is_available() else None
    train_loader = emb_loader(config['path_params']['train_path'],
                              device,
                              **config['loader_params'])
    test_loader  = emb_loader(config['path_params']['test_path'],
                              device,
                              **config['loader_params'])

    Path(f"{logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)
    print(f"======= Training {logger.name} =======")
    runner.fit(task, train_loader, test_loader)

if __name__ == '__main__':
    main()
