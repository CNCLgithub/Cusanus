import os
import yaml
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from ffcv.loader import Loader
from ffcv.transforms import ToDevice

from cusanus.archs import ImplicitNeuralModule
from cusanus.tasks import OccupancyField
from cusanus.datasets import SphericalGeometryDataset


task_name = 'occupancy_field'
dataset_name = 'spherical_geometry'

def main():
    with open(f"/project/scripts/configs/{task_name}.yaml", 'r') as file:
        config = yaml.safe_load(file)

    logger = CSVLogger(save_dir=config['logging_params']['save_dir'],
                       name= f"occupancy_field-{dataset_name}")

    # For reproducibility
    seed_everything(config['manual_seed'], True)

    # initialize networks and task
    arch = ImplicitNeuralModule(**config['arch_params'])
    arch.train()
    task = OccupancyField(arch, **config['task_params'])

    runner = Trainer(logger=logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k = 5,
                                         dirpath = os.path.join(logger.log_dir ,
                                                                "checkpoints"),
                                         monitor= "val_loss",
                                         save_last=True),
                     ],
                     accelerator = 'auto',
                     deterministic = True,
                     **config['trainer_params'])

    device = runner.device_ids[0] if torch.cuda.is_available() else None

    # CONFIGURE FFCC DATA LOADERS
    #
    # add to gpu device for ffcv loader if possible
    pipelines = SphericalGeometryDataset.ffcv_pipelines
    ps = {}
    for k in pipelines:
        ps[k] = deepcopy(pipelines[k])
        if not device is None:
            ps[k].append(ToDevice(device))

    dpath_train = f"/spaths/datasets/{dataset_name}_train.beton"
    train_loader = Loader(dpath_train,
                          pipelines = ps,
                          **config['loader_params'])
    dpath_test = f"/spaths/datasets/{dataset_name}_test.beton"
    test_loader = Loader(dpath_test,
                         pipelines = ps,
                         **config['loader_params'])


    # BEGIN TRAINING
    Path(f"{logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
    Path(f"{logger.log_dir}/reconstructions").mkdir(exist_ok=True, parents=True)
    print(f"======= Training {logger.name} =======")
    runner.fit(task, train_loader, test_loader)

if __name__ == '__main__':
    main()
