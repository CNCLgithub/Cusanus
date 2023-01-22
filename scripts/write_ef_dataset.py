#!/usr/bin/env python

import os
import yaml
import argparse
import torch

from cusanus.datasets import write_ffcv, RunningStats
from cusanus.datasets import (KFieldDataset, SceneDataset, SimDataset,
                              EFieldDataset)
from cusanus.archs import KModule
from cusanus.tasks import KField

name = 'efield'

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

    physics = config['physics']
    sim = config['simulations']

    with open(f"/project/scripts/configs/efield_dataset.yaml", 'r') as file:
        efield = yaml.safe_load(file)

    for dname in ['train', 'val', 'test']:
        c = config[dname]
        scenes = SceneDataset(**c, **physics)
        simulations = SimDataset(scenes, **sim,
                                 )
        d = EFieldDataset(simulations,
                          **efield['dataset'],
                          **stats)
        d[0]
        dpath = f"/spaths/datasets/{name}_{dname}_dataset.beton"
        d.write_ffcv(dpath, num_workers = args.num_workers)

if __name__ == '__main__':
    main()
