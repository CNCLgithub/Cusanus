#!/usr/bin/env python

import os
import yaml
import argparse
import torch
from tqdm import tqdm

from cusanus.datasets import (GFieldDataset,
                              ShapeDataset,
                              RunningStats)

name = 'gfield'

def main():
    parser = argparse.ArgumentParser(
        description = 'Generates occupancy field dataset via ffcv',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num_workers', type = int,
                        help = 'Number of write workers',
                        default = -1)
    parser.add_argument('--num_steps', type = int,
                        help = 'Number of steps for running stats',
                        default = 5000)
    args = parser.parse_args()


    with open(f"/project/scripts/configs/{name}_dataset.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # dataset is procedural so no source file
    shapes = config['shapes']
    gfield = config['gfield']
    for dname in ['train', 'val', 'test']:
        scenes = ShapeDataset(**shapes, **config[dname])
        # if dname == 'train':
        #     d = GFieldDataset(scenes, **gfield)
        #     stats = RunningStats(d.qsize)
        #     steps = min(len(d), args.num_steps)
        #     print('Computing running stats')
        #     for i in tqdm(range(steps)):
        #         qs, ys = d[i]
        #         for q in qs:
        #             stats.push(q)
        #     mean = stats.mean()
        #     stdev = stats.standard_deviation()
        #     with open(f'/spaths/datasets/{name}_running_stats.yaml', 'w') as f:
        #         yaml.safe_dump({'mean': mean.tolist(), 'std':stdev.tolist()}, f)

        #     print(f'Mean: {mean}, Std. Dev.: {stdev}')
        # d = GFieldDataset(scenes, **gfield,
        #                   qmean = mean,
        #                   qstd = stdev)
        d = GFieldDataset(scenes, **gfield, test = dname == 'test')
        d[0]

        print('Writing to .beton')
        dpath = f"/spaths/datasets/{name}_{dname}_dataset.beton"
        d.write_ffcv(dpath, num_workers = args.num_workers)

if __name__ == '__main__':
    main()
