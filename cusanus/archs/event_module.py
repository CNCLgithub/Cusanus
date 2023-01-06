import torch
from torch import nn
import torch.nn.functional as F

from cusanus.pytypes import *


class PhysObj(nn.Module):

    def __init__(self,
                 g: Tensor,
                 k: Tensor,
                 t: Tensor):
        self.register_buffer('g', g)
        self.register_buffer('k', k)
        self.register_buffer('t', t)

class PhysObjEncoder(nn.Module):

    def __init__(self, gsize:int, ksize:int, hidden:int):
        self.g_enc = nn.Sequential(
            nn.Linear(gsize, hidden),
            nn.ReLU())
        self.k_enc = nn.Sequential(
            nn.Linear(ksize + 1, hidden),
            nn.ReLU())
        self.hidden = nn.Sequential(
            nn.Linear(hidden * 2, hidden))

    def forward(self, obj:PhysObj):
        genc = self.g_enc(obj.g)
        kenc = self.k_enc(torch.cat([obj.k, obj.t]))
        return self.hidden(torch.cat([genc, kenc]))


class UpdateModule(nn.Module):

    def __init__(self, gsize:int, ksize:int, obj_hidden:int,
                 fdim:int, ddim:int, inr_params:dict) -> None:
        self.object_enc = PhysObjEncoder(gsize, ksize, obj_hidden)
        self.inr = ImplicitNeuralModule(q_in = obj_hidden,
                                        out = ksize,
                                        mod = fdim + ddim,
                                        **inr_params)

    def forward(self, obj:PhysObj, fmod, dmod):
        q = self.obj_enc(obj)
        new_k = self.inr(q, torch.cat(fmod, dmod))
        return PhysObj(obj.g, new_k, torch.zeros_like(obj.t))



class EventModule(nn.Module):

    def __init__(self,
                 c_params:dict,
                 f_params:dict,
                 u_params:dict):

        self.Q = PhysObjEncoder(**q_params)
        self.C = ImplicitNeuralModule(**c_params)
        self.F = ImplicitNeuralModule(**f_params)

    def encode_obj(self, obj:PhysObj) -> Tensor:
        return self.Q(obj)

    def forward(self, a:PhysObj, b:PhysObj, emod, amod, bmod):

        query = torch.cat([self.encode_obj(a),
                           self.encode_obj(b)])

        pval = self.C(query, emod)
        fmod = self.F(query, torch.cat([emod, amod, bmod]))
        return pval, fmod
