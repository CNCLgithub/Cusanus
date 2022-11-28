#!/usr/bin/env python

import os
import yaml
import argparse
import torch
import numpy as np

from ensembles.archs import BetaVAE
from ensembles.tasks.sym_embedding import SymEmbedding
from ensembles.datasets import EmbeddedDataset, write_emb_data

def main():
    parser = argparse.ArgumentParser(
        description = 'Converts dataset to .beton format for FFCV',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--src', type = str, default = 'pilot',
                        help = 'Which scene dataset to use')
    parser.add_argument('--config', type = str, default = 'sym_emb',
                        help =  'path to the config file')
    parser.add_argument('--ckpt', type = int, default = 2,
                        help =  'version checkpoint')
    parser.add_argument('--num_workers', type = int,
                        help = 'Number of write workers',
                        default = 4)

    args = parser.parse_args()
    dpath = os.path.join('/spaths/datasets', args.src)


    with open(f"/project/scripts/configs/{args.config}.yaml", 'r') as file:
        config = yaml.safe_load(file)

    ckpt = f"/spaths/checkpoints/sym-emb_BetaVAE/version_{args.ckpt}/checkpoints/last.ckpt"
    arch = BetaVAE(**config['arch_params'])
    embedder = SymEmbedding.load_from_checkpoint(ckpt, vae_model = arch)

    d = EmbeddedDataset(dpath, embedder)

    gs_kwargs = dict(
        shape = (128,128),
        dtype = np.dtype('float32')
    )
    dk_kwargs = dict(
        shape = (512,),
        dtype = np.dtype('float32')
    )
    writer_kwargs = dict(
        num_workers = args.num_workers
    )
    path = dpath + '_embedded.beton'
    write_emb_data(d, path, dk_kwargs, gs_kwargs, writer_kwargs)


if __name__ == '__main__':
    main()
