#!/usr/bin/env python

import os
import yaml
import argparse
import torch

from cusanus.datasets import write_ffcv, RunningStats
from cusanus.datasets import KFieldDataset, SceneDataset, SimDataset

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

    for dname in ['train', 'val']:
        c = config[dname]
        scenes = SceneDataset(**c['scenes'])
        simulations = SimDataset(scenes, **c['simulations'])
        if dname == 'train':
            d = KFieldDataset(simulations, **c['kfield'])
            stats = RunningStats(3)
            for i in range(min(len(d), 5000)):
                print('step',i)
                _, qs = d[i]
                for xyz in qs:
                    stats.push(xyz)
            mean = stats.mean()
            stdev = stats.standard_deviation()
            print(f'Mean: {mean}, Std. Dev.: {stdev}')
        d = KFieldDataset(simulations, **c['kfield'],
                          y_mean = mean,
                          y_std = stdev)
        dpath = f"/spaths/datasets/{name}_{dname}_dataset.beton"
        d.write_ffcv(dpath, num_workers = args.num_workers)

if __name__ == '__main__':
    main()
