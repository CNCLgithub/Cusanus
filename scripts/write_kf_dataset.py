#!/usr/bin/env python

import os
import yaml
import argparse
import torch

from cusanus.datasets import write_ffcv
from cusanus.datasets import KFieldDataset, SceneDataset, SimDataset
from cusanus.utils.visualization import plot_motion_trace

name = 'kfield'

def main():
    parser = argparse.ArgumentParser(
        description = 'Generates occupancy field dataset via ffcv',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num_workers', type = int,
                        help = 'Number of write workers',
                        default = -1)
    args = parser.parse_args()


    with open(f"/project/scripts/configs/{name}_dataset.yaml", 'r') as file:
        config = yaml.safe_load(file)

    scenes = SceneDataset(**config['scenes'])
    simulations = SimDataset(scenes, **config['simulations'])
    d = KFieldDataset(simulations, **config['kfield'])
    # qs, ys = d[0]
    # qs = qs[:d.segment_steps]
    # ys = ys[:d.segment_steps]
    # fig = plot_motion_trace(qs, ys)
    # fig.write_html('/spaths/datasets/motion.html')
    dpath = f"/spaths/datasets/{name}_train_dataset.beton"
    d.write_ffcv(dpath, num_workers = args.num_workers)

if __name__ == '__main__':
    main()
