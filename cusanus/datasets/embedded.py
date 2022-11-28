import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader
from ffcv.fields import NDArrayField
from ffcv.pipeline.operation import Operation
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import (Convert, ToTensor, ToDevice)

from ensembles.pytypes import *
from ensembles.tasks import SymEmbedding
from ensembles.datasets import SymbolicDataset


class EmbeddedDataset(Dataset):
    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, src: str, embedder: SymEmbedding):
        with open(src + '_manifest.json', 'r') as f:
            manifest = json.load(f)
        self.src = src
        self.manifest = manifest
        self.sym_d = SymbolicDataset(src)
        embedder.eval()
        embedder.freeze()
        self.embedder = embedder

    #length function w parameter specified in manifest file
    def __len__(self):
        return len(self.sym_d)

    #loads a trial
    def __getitem__(self, idx):
        kstate, gstate = self.sym_d[idx]
        kstate = torch.Tensor(kstate)
        kstate.unsqueeze(0)
        # mu layer
        _, _, emb, _ = self.embedder(kstate)
        emb.squeeze(0)
        # add zero granularity features for level 1 objects
        emb = torch.cat((torch.zeros(len(emb)), emb))
        emb = emb.numpy()
        gstate = gstate.reshape((-1,1))
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

def array_pipeline() -> List[Operation]:
    return [NDArrayDecoder(),
            ToTensor(),
            Convert(torch.float32)]

def emb_loader(path: str, device,  **kwargs) -> Loader:
    array_pipe = array_pipeline()
    if not device is None:
        array_pipe.append(ToDevice(device))
    l =  Loader(path + '.beton',
                pipelines= {'ks' : array_pipe,
                            'gs': array_pipe},
                **kwargs)
    return l
