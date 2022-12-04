#!/usr/bin/env python

import os
import yaml
import argparse
import torch
import numpy as np

from cusanus.datasets import SphericalGeometryDataset

def main():
    parser = argparse.ArgumentParser(
        description = 'Converts dataset to .beton format for FFCV',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dest', type = str, default = 'spherical_geometry',
                        help = 'Name of dataset output')
    parser.add_argument('--config', type = str, default = 'spherical_geometry',
                        help =  'path to the config file')
    parser.add_argument('--num_workers', type = int,
                        help = 'Number of write workers',
                        default = 4)
    args = parser.parse_args()

    with open(f"/project/scripts/configs/spherical_geometry.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # dataset is procedural so no source file
    dpath = os.path.join('/spaths/datasets', args.dest + '_train.beton')
    d = SphericalGeometryDataset(**config['train'])
    w = d.write_ffcv(dpath)
    w.from_indexed_dataset(d)
    dpath = os.path.join('/spaths/datasets', args.dest + '_test.beton')
    d = SphericalGeometryDataset(**config['test'])
    # d.write_ffcv(dpath)
    w = d.write_ffcv(dpath)
    w.from_indexed_dataset(d)

if __name__ == '__main__':
    main()
