import time
import torch
import trimesh
import numpy as np
import pybullet as p
from copy import deepcopy
import operator as op
from functools import reduce
from itertools import combinations

from torch.utils.data import Dataset
from cusanus.pytypes import *
from cusanus.utils.physics import (mesh_to_bullet,
                                   sphere_to_bullet,
                                   rect_to_bullet)

def _rect(x, y, z) -> trimesh.Trimesh:
    extents = [x[1]-x[0], y[1]-y[0], z[1]-z[0]]
    m = trimesh.primitives.Box(extents=extents)
    pos = [np.mean(x), np.mean(y), np.mean(z)]
    tm = trimesh.transformations.translation_matrix(pos)
    m.apply_transform(tm)
    return m

class SceneDataset(Dataset):

    def __init__(self,
                 n_scenes:int = 100,
                 table_extents:List[float] = [1.0, 1.0],
                 table_z:float = 0.05,
                 wall_depth:float=0.05,
                 wall_z:float = 0.1,
                 gate_spacing:float=0.2,
                 gate_rng:List[float]=[0.1, 0.2],
                 target_radius_rng:List[float]=[0.05, 0.1],
                 target_vel_rng:List[float]=[1.0, 2.0],
                 target_gate_prob:float=0.3,
                 table_phys:dict={'mass': 0.},
                 wall_phys:dict={'mass': 0.},
                 target_phys:dict={'mass': 1.},
                 ) -> None:
        self.n_scenes = n_scenes
        # Scene params
        self.table_extents = table_extents
        self.table_z = table_z
        self.wall_depth = wall_depth
        self.wall_z = wall_z
        self.gate_spacing = gate_spacing
        # Random params
        self.gate_rng = gate_rng
        self.target_radius_rng = target_radius_rng
        self.target_vel_rng = target_vel_rng
        self.target_gate_prob = target_gate_prob
        self.table_phys = table_phys
        self.wall_phys = wall_phys
        self.target_phys = target_phys
        self.scene_template = self.init_table()

    def __len__(self):
        return self.n_scenes

    def init_table(self):
        """ Initializes the base table and N,S,E walls"""
        xext, yext = self.table_extents
        table = _rect([-0.5*xext, 0.5*xext],
                      [-0.5*yext, 0.5*yext],
                      [-self.table_z, 0.])
        nwall = _rect([-0.5*xext, 0.5*xext],
                      [0.5*yext, 0.5*yext+self.wall_depth],
                      [0., self.wall_z])
        swall = _rect([-0.5*xext, 0.5*xext],
                      [-0.5*yext-self.wall_depth, -0.5*yext],
                      [0., self.wall_z])
        ewall = _rect([0.5*xext, 0.5*xext+self.wall_depth],
                      [-0.5*yext, 0.5*yext],
                      [0., self.wall_z])
        wwall = _rect([-0.5*xext-self.wall_depth, -0.5*xext],
                      [-0.5*yext, 0.5*yext],
                      [0., self.wall_z])
        scene = {
            'table' : {'geometry' : (table,),
                       'physics' : self.table_phys,
                       'record' : False,
                       'loader' : rect_to_bullet},
            'nwall' : {'geometry' : (nwall,),
                       'physics' : self.wall_phys,
                       'record' : False,
                       'loader' : rect_to_bullet},
            'swall' : {'geometry' : (swall,),
                       'physics' : self.wall_phys,
                       'record' : False,
                       'loader' : rect_to_bullet},
            'ewall' : {'geometry' : (ewall,),
                       'physics' : self.wall_phys,
                       'record' : False,
                       'loader' : rect_to_bullet},
            'wwall' : {'geometry' : (wwall,),
                       'physics' : self.wall_phys,
                       'record' : False,
                       'loader' : rect_to_bullet},
            # geometry sampled for each trial
            'gate_a' : {'physics' : self.wall_phys,
                        'record' : False,
                        'loader' : rect_to_bullet},
            'gate_b' : {'physics' : self.wall_phys,
                        'record' : False,
                        'loader' : rect_to_bullet},
        }
        return scene


    def sample_gate(self):
        """ Samples the western wall with a gate

        Returns two rectangular prisms that are seperated
        by a random distance
        """
        width = np.random.uniform(*self.gate_rng)
        xext, yext = self.table_extents
        ymin = -0.5*yext + 0.05 + width
        ymax =  0.5*yext - 0.05 - width
        ygate = np.random.uniform(ymin, ymax)
        ygate_a = _rect([-0.5*xext+self.gate_spacing,
                         -0.5*xext+self.gate_spacing+self.wall_depth],
                        [-0.5*yext, ygate - 0.5 * width],
                        [0., self.wall_z])
        ygate_b = _rect([-0.5*xext+self.gate_spacing,
                         -0.5*xext+self.gate_spacing+self.wall_depth],
                        [ygate + 0.5 * width, 0.5*yext],
                        [0., self.wall_z])
        gate_extents = [ygate - width, ygate + width]
        return (gate_extents, ygate_a, ygate_b)

    def sample_target(self, gate_extents):
        xext, yext = self.table_extents
        radius = np.random.uniform(*self.target_radius_rng)
        delta = radius + 4*self.wall_depth
        xpos = np.random.uniform(-0.5*xext + delta,
                                 0.5*xext - delta)
        ypos = np.random.uniform(-0.5*yext + delta,
                                 0.5*yext - delta)
        angle = np.random.uniform(0., 2*np.pi)

        mag = np.random.uniform(*self.target_vel_rng)
        vel = [np.cos(angle) * mag, np.sin(angle) * mag, 0.]
        # vel = [10.0, 0., 0.]
        pos = [xpos, ypos, radius+0.01]
        target = {
            'geometry': (radius, pos),
            'physics' : self.target_phys,
            'state' : {'linearVelocity' : vel},
            'loader': sphere_to_bullet
        }
        return target

    def __getitem__(self, idx):
        gate_extents, gate_a, gate_b = self.sample_gate()
        target = self.sample_target(gate_extents)
        scene = {'target' : target}
        scene.update(**deepcopy(self.scene_template))
        scene['gate_a']['geometry'] = (gate_a,)
        scene['gate_b']['geometry'] = (gate_b,)
        return scene

class SimDataset(Dataset):

    def __init__(self,
                 scene_dataset:SceneDataset,
                 max_dur:float = 5000.,
                 gravity:float = -10.0,
                 debug = False
                 ):
        self.scenes = scene_dataset
        self.max_dur = max_dur
        self.gravity = gravity
        self.debug = debug

    def __len__(self):
        return len(self.scenes)

    def init_client(self, **kwargs):
        """" Returns an initialized pybullet client. """
        if self.debug:
            t = p.GUI
        else:
            t = p.DIRECT
        cid = p.connect(t, **kwargs)
        return cid

    def __getitem__(self, idx):
        # load random initial scene
        scene = self.scenes[idx]
        # initialize physics server
        cid = self.init_client()
        # initialize collision bodies
        #   associate scene keys with pybullet ids
        registry = {}
        for (k, o) in scene.items():
            oid = o['loader'](*o['geometry'], cid)
            registry[k] = oid
            p.changeDynamics(oid, -1, **o['physics'],
                             physicsClientId = cid)
            if 'state' in o:
                p.resetBaseVelocity(oid, **o['state'],
                                    physicsClientId=cid)

        object_ids = list(registry.values())
        nobjects = len(object_ids)

        p.setGravity(0, 0, self.gravity,
                     physicsClientId = cid)

        pparams = p.getPhysicsEngineParameters(physicsClientId = cid)
        delta_t = pparams['fixedTimeStep']
        n = np.ceil(self.max_dur / delta_t).astype(int)
        # HACK
        n += 1

        if self.debug:
            # add one step to resolve any initial forces
            p.stepSimulation(physicsClientId = cid)
            p.setRealTimeSimulation(1, physicsClientId = cid)
            while (1):
                keys = p.getKeyboardEvents()
                print(keys)
                time.sleep(0.01)
            return None

        steps = 0
        dur = 0.
        position = np.empty((n, nobjects, 3))
        collision = np.zeros((n, _ncr(nobjects, 2)),
                             dtype=bool)

        while dur < self.max_dur:
            p.stepSimulation(physicsClientId = cid)

            # record positions
            for oid in object_ids:
                pos = p.getBasePositionAndOrientation(oid,cid)[0]
                position[steps, oid] = pos

            # record collisions
            for c,(a,b) in enumerate(combinations(object_ids, 2)):
                nc = len(p.getContactPoints(bodyA=a, bodyB=b,
                                            physicsClientId = cid))
                collision[steps, c] = nc > 0

            dur += delta_t
            steps += 1

        state = {
            'position' : position[:steps],
            'collision': collision[:steps],
        }
        # disconnect
        p.disconnect(physicsClientId = cid)
        return scene, registry, state


def _ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom
