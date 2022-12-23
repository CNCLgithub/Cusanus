import numpy as np
import pybullet as p
from trimesh import Trimesh

# https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/createTexturedMeshVisualShape.py#L138
# creating mesh from vertices
def mesh_to_bullet(mesh:Trimesh, cid:int):
    vertices = np.array(mesh.vertices)
    col_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                    vertices=vertices,
                                    physicsClientId = cid)
    oid = p.createMultiBody(baseCollisionShapeIndex=col_id,
                            physicsClientId = cid)
    return oid

def sphere_to_bullet(radius:float, pos, cid:int):
    col_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                    radius = radius * 0.5,
                                    physicsClientId = cid)
    oid = p.createMultiBody(baseCollisionShapeIndex=col_id,
                            basePosition = pos,
                            physicsClientId = cid)
    return oid
