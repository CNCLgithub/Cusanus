#!/usr/bin/env python

import os
import yaml
import argparse
import torch
import torchvision
import numpy as np
from copy import deepcopy

from ffcv.loader import Loader

from cusanus.datasets import SphericalGeometryDataset
from cusanus.datasets.geometry import spherical_occupancy_field
from cusanus.utils import grids_along_depth
from cusanus.utils.visualization import aggregrate_depth_scans

def debug_dataset(r:float = 0.7):
    qs = torch.Tensor(grids_along_depth(9, 128))
    ys = torch.Tensor(spherical_occupancy_field(r, qs))
    grid = aggregrate_depth_scans(qs, ys, 9, 128)
    img_path = '/spaths/datasets/test.png'
    torchvision.utils.save_image(grid, img_path, normalize=False)


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

    debug_dataset()

    with open(f"/project/scripts/configs/spherical_geometry.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # dataset is procedural so no source file
    dpath = os.path.join('/spaths/datasets', args.dest + '_train.beton')
    d = SphericalGeometryDataset(**config['train'])
    d.write_ffcv(dpath)

    dpath = os.path.join('/spaths/datasets', args.dest + '_test.beton')
    d = SphericalGeometryDataset(**config['test'])
    d.write_ffcv(dpath)

    pipelines = SphericalGeometryDataset.ffcv_pipelines
    ps = {}
    for k in pipelines:
        ps[k] = deepcopy(pipelines[k])
        # if not device is None:
        #     ps[k].append(ToDevice(device))
    loader = Loader(dpath, pipelines = ps, batch_size = 8)
    for (qs, ys) in loader:
        print(ys.mean())

if __name__ == '__main__':
    main()
