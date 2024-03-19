#!/usr/bin/env python

import os
import yaml
import argparse
import torch

from cusanus.datasets import write_ffcv, RunningStats
from cusanus.datasets import (SceneDataset,
                              SimDataset,
                              KCodesDataset,
                              write_to_hdf5,
                              H5Dataset)
from cusanus.archs import KModule
from cusanus.tasks import KField

name = 'efield'

def load_kfield(config:dict, ckpt_path:str):
    arch = KModule(**config['arch_params'])
    field = KField.load_from_checkpoint(ckpt_path, module = arch)
    return field

def main():
    parser = argparse.ArgumentParser(
        description = 'Generates occupancy field dataset via ffcv',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num_workers', type = int,
                        help = 'Number of write workers',
                        default = -1)
    args = parser.parse_args()


    with open(f"/project/scripts/configs/kfield_dataset.yaml", 'r') as file:
        config = yaml.safe_load(file)
    with open(f"/spaths/datasets/kfield_running_stats.yaml", 'r') as file:
        stats = yaml.safe_load(file)
    with open(f"/project/scripts/configs/kfield_task.yaml", 'r') as file:
        kconfig = yaml.safe_load(file)
    with open(f"/project/scripts/configs/efield_dataset.yaml", 'r') as file:
        efield = yaml.safe_load(file)

    physics = config['physics']
    sim = config['simulations']
    kfield = load_kfield(kconfig, efield['kfield_ckpt'])
    device = 0 if torch.cuda.is_available() else None
    kfield = kfield.to(device)

    for dname in ['train', 'val', 'test']:
        c = config[dname]
        scenes = SceneDataset(**c, **physics)
        simulations = SimDataset(scenes, **sim)
        d = KCodesDataset(simulations,
                          kfield,
                          **efield['dataset'],
                          **stats)
        d[0]
        dpath = f"/spaths/datasets/{name}_{dname}_dataset.hdf5"
        # write_to_hdf5(d, dpath)
        dh5 = H5Dataset(d, dpath)
        dh5[0]
        dpath = f"/spaths/datasets/{name}_{dname}_dataset.beton"
        dh5.write_ffcv(dpath, num_workers = args.num_workers)

if __name__ == '__main__':
    main()
