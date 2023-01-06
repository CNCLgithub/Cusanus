import time
import torch
import trimesh
import numpy as np
import pybullet as p

from torch.utils.data import Dataset
from cusanus.pytypes import *
from cusanus.utils.meshes import sample_ramp, sample_obstacle
from cusanus.utils.physics import mesh_to_bullet, sphere_to_bullet

class SceneDataset(Dataset):

    def __init__(self,
                 n_scenes:int = 100,
                 delta_pos:float = 1.0,
                 delta_size:float = 0.1,
                 ramp_extents = [6.0, 1.0, 0.25],
                 ramp_init_pos = [-2.0, 0., 0.0],
                 obs_extents = [1.5, 1, 2.5],
                 obstacle_init_pos = [2.75, 0., -0.5],
                 sphere_z_max = 7.0,
                 ):
        self.n_scenes = n_scenes
        self.delta_pos = delta_pos
        self.delta_size = delta_size
        self.ramp_extents = ramp_extents
        self.ramp_init_pos = ramp_init_pos
        self.obs_extents = obs_extents
        self.obstacle_init_pos = obstacle_init_pos
        self.sphere_z_max = sphere_z_max

    def __len__(self):
        return self.n_scenes

    def sample_ramp(self):
        mesh =  sample_ramp(self.ramp_extents, 0.1 * np.pi,
                            np.pi * 0.25)
        # sample position
        dx,dz = np.random.uniform(low = -self.delta_pos,
                                  high = self.delta_pos,
                                  size = 2)
        ipos = self.ramp_init_pos + np.array([dx, 0., dz])
        tm = trimesh.transformations.translation_matrix(ipos)
        # sample random size scale
        scale = np.random.uniform(low = 1.0 - self.delta_size,
                                  high = 1.0 + self.delta_size)
        sm = trimesh.transformations.scale_matrix(scale)
        mat = trimesh.transformations.concatenate_matrices(tm, scale)
        mesh.apply_transform(mat)
        return mesh


    def sample_obstacle(self):
        mesh =  sample_obstacle(self.obs_extents, 0.,
                                np.pi * 0.35)
        # sample position
        dx,dz = np.random.uniform(low = -self.delta_pos,
                                  high = self.delta_pos,
                                  size = 2)
        ipos = self.obstacle_init_pos + np.array([dx, 0., dz])
        tm = trimesh.transformations.translation_matrix(ipos)
        # sample random size scale
        scale = np.random.uniform(low = 1.0 - self.delta_size,
                                  high = 1.0 + self.delta_size)
        sm = trimesh.transformations.scale_matrix(scale)
        mat = trimesh.transformations.concatenate_matrices(tm, sm)
        mesh.apply_transform(mat)
        return mesh

    def sample_sphere(self, xbounds:Tuple[float,float], zmin:float):
        radius = np.random.uniform(low = 0.25 - self.delta_size,
                                   high = 0.25 + self.delta_size)
        x = np.random.uniform(*xbounds)
        z = np.random.uniform(zmin+radius, self.sphere_z_max)
        pos = np.array([x, 0., z])
        return (radius, pos)

    def __getitem__(self, idx):
        ramp = self.sample_ramp()
        obstacle = self.sample_obstacle()
        ramp_bounds = ramp.bounds
        obs_bounds = obstacle.bounds
        xmin = min(min(ramp_bounds[:, 0]),
                   min(obs_bounds[:, 0]))
        xmax = max(max(ramp_bounds[:, 0]),
                   max(obs_bounds[:, 0]))
        zmax = max(max(ramp_bounds[:, 2]),
                   max(obs_bounds[:, 2]))
        sphere = self.sample_sphere((xmin,xmax), zmax)
        return (ramp, obstacle, sphere)

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
        ramp, obstacle, sphere = self.scenes[idx]
        # initialize physics server
        cid = self.init_client()
        # initialize collision bodies
        ramp_id = mesh_to_bullet(ramp, cid)
        obstacle_id = mesh_to_bullet(obstacle, cid)
        sphere_id = sphere_to_bullet(*sphere, cid)

        # configure dynamics
        p.setGravity(0, 0, -self.gravity,
                     physicsClientId = cid)
        p.changeDynamics(ramp_id, -1, mass=0.0, restitution=0.5,
                         physicsClientId = cid)
        p.changeDynamics(obstacle_id, -1, mass=0.0, restitution=0.5,
                         physicsClientId = cid)
        p.changeDynamics(sphere_id, -1, mass=1.0, restitution=0.5,
                         physicsClientId = cid)

        pparams = p.getPhysicsEngineParameters(physicsClientId = cid)
        delta_t = pparams['fixedTimeStep']
        n = np.ceil(self.max_dur / delta_t).astype(int)

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
        pos, quat = p.getBasePositionAndOrientation(sphere_id,
                                                    physicsClientId = cid)
        current_sphere_z = pos[2]
        position = np.empty((n, 3))
        # velocity = np.empty((n, 3))

        while dur < self.max_dur:
            p.stepSimulation(physicsClientId = cid)
            pos, _ = p.getBasePositionAndOrientation(sphere_id,
                                                     physicsClientId = cid)
            l_vel, _ = p.getBaseVelocity(sphere_id,
                                         physicsClientId = cid)
            position[steps] = pos
            # velocity[steps] = l_vel
            current_sphere_z = pos[2]
            dur += delta_t
            steps += 1

        position = position[:steps]
        # disconnect
        p.disconnect(physicsClientId = cid)
        return (ramp, obstacle, sphere), position
