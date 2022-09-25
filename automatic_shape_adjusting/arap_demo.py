import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch

from arap_core.arap_numpy import ARAP as numpy_arap
from arap_core.arap_torch import ARAP as torch_arap


def load_tri_mesh(path):
    """load tri-mesh from .obj
    returns: Open3D mehs object
    """
    V = [] # vertices
    F = [] # faces
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            line_array = line.split(' ')
            if line_array[0] == 'v':
                # vertex
                vertex = []
                for j in range(3):
                    vertex.append(float(line_array[j+1].split('/')[0]))
                V.append(vertex)
            elif line_array[0] == 'f':
                # face
                face = []
                for j in range(3):
                    face.append(int(line_array[j+1].split('/')[0]))
                F.append(face)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(V))
    # notice that .obj start id from 1, so minus 1
    mesh.triangles = o3d.utility.Vector3iVector(np.array(F)-1)
    return mesh



# load the example .obj
mesh = load_tri_mesh('./bunny.obj')
#mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh])

V = np.asarray(mesh.vertices)
F = np.asarray(mesh.triangles)

# plot
#plt.figure(figsize=(5,5))
#plt.scatter(x=V[:,0], y=V[:,1], s=1.0)
#plt.show()



def open3d_form_mesh(V, F):
    """
    Form a triangle mesh using Open3D
    -----------------------------------
    input:
        - V: the vertex coordinates, shape should be [n ,3]
        - F: the triangle faces, shape should be [m ,3]
    output:
        - mesh: an Open3D mesh object
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    return mesh

def open3d_arap(V, F, handles):
    """
    input:
        - V: the vertex coordinates on the original mesh
        - F: the triangle faces
        - handles: the dictionary of handle vertex indices (as key) and coordinates (as value)
    output:
        - V_new: the new vertex coordinates
    """
    # form Open3D mesh
    mesh = open3d_form_mesh(V = V, F = F)
    mesh.compute_vertex_normals()

    # form the constraints
    handle_ids = np.array(list(handles.keys()), dtype=np.int32)
    handle_pos = np.array([handles[vid] for vid in handle_ids])
    constraint_ids = o3d.utility.IntVector(handle_ids)
    constraint_pos = o3d.utility.Vector3dVector(handle_pos)

    # solve the system
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        # max_iter can be changed
        mesh = mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=4)

    # get the new vertices
    V_new = np.asarray(mesh.vertices, dtype=np.float32)

    return V_new

















handles = {}
# anchor
handles[1336] = V[1336] + (-0.05, 0, 0)
# static
handles[909] = V[909]



###############
# Open3D ARAP
V_new_by_open3d = open3d_arap(V = V, F = F, handles = handles)

###############
# Numpy ARAP
numpy_arap_solver = numpy_arap(V, F, handles, laplacian='combinatorial')
#numpy_arap_solver = numpy_arap(V, F, handles, laplacian='cotangent')
V_new_by_numpy = numpy_arap_solver(num_iter=3)


###############
# Torch ARAP
pnts = torch.from_numpy(V).type(torch.float32)
tensor_handles = {}
tensor_handles[1336] = pnts[1336] + torch.from_numpy(np.array([-0.05,0,0],np.float32)).to(pnts.device)
tensor_handles[909] = pnts[909]
torch_arap_solver = torch_arap(pnts, F, tensor_handles, laplacian='combinatorial')
V_new_by_torch = torch_arap_solver(num_iter=3)





mesh = open3d_form_mesh(V = V_new_by_torch, F = F)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])




# plot results
plt.figure(figsize=(8, 8))
plt.scatter(x=V[:, 0], y=V[:, 1], s=15, color='gray', label='old')
plt.scatter(x=V_new_by_open3d[:, 0], y=V_new_by_open3d[:, 1], s=5, color='orange', label='Open3D')
plt.scatter(x=V_new_by_numpy[:, 0], y=V_new_by_numpy[:, 1], s=3, color='red', label='Numpy')
plt.scatter(x=V_new_by_torch[:, 0], y=V_new_by_torch[:, 1], s=1, color='blue', label='Torch')
plt.legend()
plt.show()



