#!/usr/bin/env python

import os
import yaml
import argparse
import torch
import torchvision
import numpy as np
from copy import deepcopy

from ffcv.loader import Loader

from cusanus.datasets import MeshGeometryDataset
from cusanus.datasets.geometry import mesh_occupancy_field
from cusanus.utils import grids_along_depth, grids_along_axis
from cusanus.utils.visualization import aggregrate_depth_scans, plot_volume

import trimesh
import plotly.graph_objects as go

def viz_trial(d):
    m = d.sample_obstacle()
    qs = grids_along_axis(30, 30, delta=5.0)
    ys = mesh_occupancy_field(m, qs).astype(np.float64)
    fig = plot_volume(qs, ys)
    fig.write_html('/spaths/datasets/mesh.html')

srcs = ['/spaths/datasets/platform_one.obj']

def main():
    parser = argparse.ArgumentParser(
        description = 'Converts dataset to .beton format for FFCV',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dest', type = str, default = 'mesh_geometry',
                        help = 'Name of dataset output')
    parser.add_argument('--config', type = str, default = 'mesh_geometry',
                        help =  'path to the config file')
    parser.add_argument('--num_workers', type = int,
                        help = 'Number of write workers',
                        default = 4)
    args = parser.parse_args()


    with open(f"/project/scripts/configs/mesh_geometry.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # dataset is procedural so no source file
    d = MeshGeometryDataset(**config['train'])
    viz_trial(d)
    # dpath = os.path.join('/spaths/datasets', args.dest + '_train.beton')
    # d.write_ffcv(dpath)

    # dpath = os.path.join('/spaths/datasets', args.dest + '_test.beton')
    # d = MeshGeometryDataset(srcs, **config['test'])
    # d.write_ffcv(dpath)

    # pipelines = MeshGeometryDataset.ffcv_pipelines
    # ps = {}
    # for k in pipelines:
    #     ps[k] = deepcopy(pipelines[k])
    #     # if not device is None:
    #     #     ps[k].append(ToDevice(device))
    # loader = Loader(dpath, pipelines = ps, batch_size = 8)
    # for (qs, ys) in loader:
    #     print(qs.shape)
    #     print(ys.shape)
    #     print(ys.mean())

if __name__ == '__main__':
    main()
