#!/usr/bin/env python

import os
import yaml
import argparse
import torch

from ffcv.loader import Loader

from cusanus.datasets import write_ffcv
from cusanus.datasets import MeshGFieldDataset, SphericalGFieldDataset

name = 'gfield'

def main():
    parser = argparse.ArgumentParser(
        description = 'Generates occupancy field dataset via ffcv',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num_workers', type = int,
                        help = 'Number of write workers',
                        default = 4)
    args = parser.parse_args()


    with open(f"/project/scripts/configs/{name}_dataset.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # dataset is procedural so no source file
    d1 = MeshGFieldDataset(**config['mesh']['train'])
    d2 = SphericalGFieldDataset(**config['sphere']['train'])
    dc = torch.utils.data.ConcatDataset([d1, d2])
    dpath = f"/spaths/datasets/{name}_train_dataset.beton"
    write_ffcv(dc, d1.k_queries, dpath)

if __name__ == '__main__':
    main()
