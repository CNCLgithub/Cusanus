from cusanus.pytypes import *

import trimesh
import numpy as np

def center_mesh(mesh):
    c = mesh.centroid
    T = trimesh.transformations.translation_matrix(-c)
    m = mesh.copy()
    m.apply_transform(T)
    return m


def sheered_rect_prism(extents, theta:float):
    box = trimesh.primitives.Box(extents=extents)
    # theta 0 = dz 0
    dz = np.sin(theta) * extents[2]
    delta = np.array([0., 0., -dz])
    # assuming left-hand rule, want vertices 1,3
    vs = []
    for (i,v) in enumerate(box.vertices):
        x = np.array(v)
        if i == 1 or i == 3:
            x += delta
        vs.append(x)

    faces = np.array(box.faces)
    mesh = trimesh.Trimesh(vertices = vs,
                           faces = faces)
    return mesh

def ramp_mesh(extents:List[float], theta:float):
    box = trimesh.primitives.Box(extents=extents)
    ax = [0., 1., 0.]
    rotm = trimesh.transformations.rotation_matrix(theta, ax)
    box.apply_transform(rotm)
    return box

def sample_obstacle(extents, low:float, high:float):
    theta = np.random.uniform(low = low, high = high)
    return sheered_rect_prism(extents, theta)

def sample_ramp(extents, low:float, high:float):
    theta = np.random.uniform(low = low, high = high)
    return ramp_mesh(extents, theta)
