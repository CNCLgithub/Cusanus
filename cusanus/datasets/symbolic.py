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

from ensembles.pytypes import *

class SymbolicDataset(Dataset):
    # inheriting pytorch dataset; return vector of object and gstate
    def __init__(self, src: str):
        with open(src + '_manifest.json', 'r') as f:
            manifest = json.load(f)
        self.src = src
        self.manifest = manifest
        self.embedding_order = ["radius", "mass", "pos", "vel", "gstate"]
    #length function w parameter specified in manifest file
    def __len__(self):
        return self.manifest['trials'] * self.manifest['k'] * \
            self.manifest['n_dots']

    #loads a trial
    def __getitem__(self, idx):
        n = len(self)
        scene = np.floor(idx/n).astype(int) + 1
        nk = self.manifest['k'] * self.manifest['n_dots']
        time = np.floor((idx % nk)/self.manifest['n_dots']).astype(int) + 1
        item = idx % self.manifest['n_dots'] + 1
        obj_path = f"{self.src}/{scene}/serialized/{time}_{item}.json"
        img_path = f"{self.src}/{scene}/images/{time}_{item}.png"
        # gstate = np.asarray(pil_loader(img_path))
        gstate = np.asarray(Image.open(img_path).convert('L')).astype('float32')/255.0
        #print(np.squeeze(gstate, axis=2).shape)
        with open(obj_path, 'r') as f:
            obj = json.load(f)

        kstate = []
        for i in range(time-5,time):
            obj_path = f"{self.src}/{scene}/serialized/{i+1}_{item}.json"
            if i<0:
                obj_i = obj
            else:
                with open(obj_path, 'r') as f:
                    obj_i = json.load(f)
            kstate.append(np.asarray(obj_i['pos'])/300.0)
            kstate.append(np.asarray(obj_i['vel'])/100.0)

        kstate = np.asarray(kstate)
        kstate = np.ndarray.flatten(kstate).astype(np.float32)
        return kstate, gstate


def write_sym_data(d: SymbolicDataset,
                    path: str,
                    dk_kwargs: dict,
                    gs_kwargs: dict,
                    w_kwargs: dict) -> None:
    writer = DatasetWriter(path,
                           {'ks': NDArrayField(**dk_kwargs),
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

def sym_loader(path: str, device,  **kwargs) -> Loader:
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
