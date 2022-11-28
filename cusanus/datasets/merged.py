import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.datasets.folder import pil_loader
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.fields import RGBImageField, NDArrayField
from ffcv.pipeline.operation import Operation
from ffcv.fields.decoders import NDArrayDecoder, SimpleRGBImageDecoder
from ffcv.transforms import (Convert, NormalizeImage, ToTensor, ToTorchImage,
    ToDevice)

from . pytypes import *

class EmbeddedDataset(Dataset):
    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, src: str, embedder: SymEmbedding,
                 group_size:int = 1):
        with open(src + '_manifest.json', 'r') as f:
            manifest = json.load(f)
        self.src = src
        self.manifest = manifest
        self.sym_d = SymbolicDataset(src)
        self.embedder = embedder
        self.group_size = group_size
        group_vec = torch.arange(manifest['n_dots'])
        group_vec = torch.combinations(group_vec, r = group_size)
        self.group_vec = group_vec
        self.n_groups = len(group_vec)

    #length function w parameter specified in manifest file
    def __len__(self):
        return self.manifest['trials'] * self.manifest['k'] * \
            self.n_groups

    #loads a trial
    def __getitem__(self, idx):
        gidx = idx % self.n_groups
        root_idx = np.floor(idx / self.n_groups).astype(int)
        oidxs = self.group_vec[gidx] + root_idx
        kstate, gstate = self.sym_d[oidxs]
        emb = self.embedder(kstate)
        emb = emb[:, 0]
        emb = torch.cat((torch.zeros(emb), emb), dim = 1)
        return emb, gstate


def write_emb_data(d: EmbeddedDataset,
                   path: str,
                   dk_kwargs: dict,
                   gs_kwargs: dict,
                   w_kwargs: dict) -> None:
    writer = DatasetWriter(path,
                           {'es': NDArrayField(**dk_kwargs),
                            'gs': NDArrayField(**gs_kwargs)},
                           **w_kwargs)
    writer.from_indexed_dataset(d)

# def img_pipeline(mu, sd) -> List[Operation]:
#     return [SimpleRGBImageDecoder(),
#             NormalizeImage(mu, sd, np.float32),
#             ToTensor(),
#             ToTorchImage(convert_back_int16 =False),
#             Convert(torch.float32),
#             ]

def object_pipeline() -> List[Operation]:
    return [NDArrayDecoder(),
            ToTensor(),
            Convert(torch.float32)]

def sym_emb_loader(path: str, device,  **kwargs) -> Loader:
    with open(path + '_manifest.json', 'r') as f:
        manifest = json.load(f)
    obj_pipe = object_pipeline()
    if not device is None:
        obj_pipe.append(ToDevice(device))
    l =  Loader(path + '.beton',
                pipelines= {'ks' : obj_pipe,
                            'gs': None},
                **kwargs)
    return l
