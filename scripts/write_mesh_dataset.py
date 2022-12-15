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
from cusanus.utils.visualization import aggregrate_depth_scans

import trimesh
import plotly.graph_objects as go

def viz_trial(d):
    m = d.meshes[0]
    qs = grids_along_axis(30, 30, delta=6.0)
    ys = mesh_occupancy_field(m, qs).astype(np.float64)
    # ys = trimesh.proximity.signed_distance(m, qs)
    fig = go.Figure(data=go.Volume(
        x=qs[:, 0],
        y=qs[:, 1],
        z=qs[:, 2],
        value=ys,
        # isomin=-5.0,
        # isomax=3.0,
        opacity=0.2, # needs to be small to see through all surfaces
        surface_count=20, # needs to be a large number for good volume rendering
        autocolorscale = True,
        ))
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
    dpath = os.path.join('/spaths/datasets', args.dest + '_train.beton')
    d = MeshGeometryDataset(srcs, **config['train'])
    viz_trial(d)
    d.write_ffcv(dpath)

    dpath = os.path.join('/spaths/datasets', args.dest + '_test.beton')
    d = MeshGeometryDataset(srcs, **config['test'])
    d.write_ffcv(dpath)

    pipelines = MeshGeometryDataset.ffcv_pipelines
    ps = {}
    for k in pipelines:
        ps[k] = deepcopy(pipelines[k])
        # if not device is None:
        #     ps[k].append(ToDevice(device))
    loader = Loader(dpath, pipelines = ps, batch_size = 8)
    for (qs, ys) in loader:
        print(qs.shape)
        print(ys.shape)
        print(ys.mean())

if __name__ == '__main__':
    main()
