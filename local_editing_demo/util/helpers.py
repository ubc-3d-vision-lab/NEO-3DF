import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions


import numpy as np
import pickle
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
from PIL import Image

import open3d as o3d


class ShapeDecoder(nn.Module):
    def __init__(self, latent_dim, n_vert):
        super(ShapeDecoder, self).__init__()
        self.n_vert = n_vert
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, n_vert*3)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_ = self.fc3(h)
        return x_.reshape(-1, self.n_vert, 3)

    
## Part Offset Prediction Module
class OffsetRegressorA(nn.Module):
    def __init__(self):
        super(OffsetRegressorA, self).__init__()
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 32)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_ = F.relu(self.fc2(h))
        return x_.reshape(-1, 32)
    
class OffsetRegressorB(nn.Module):
    def __init__(self):
        super(OffsetRegressorB, self).__init__()
        self.fc1 = nn.Linear(32, 8)
        self.fc2 = nn.Linear(8, 1)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_ = self.fc2(h)
        return x_.reshape(-1, 1)

    
class OffsetRegressor(nn.Module):
    def __init__(self, offRegA, offRegsB):
        super(OffsetRegressor, self).__init__()
        self.offRegA = offRegA
        self.S_brows_y = offRegsB['S_eyebrows-y']
        self.S_brows_z = offRegsB['S_eyebrows-z']
        self.S_eyes_y = offRegsB['S_eyes-y']
        self.S_eyes_z = offRegsB['S_eyes-z']
        self.S_llip_y = offRegsB['S_llip-y']
        self.S_llip_z = offRegsB['S_llip-z']
        self.S_ulip_y = offRegsB['S_ulip-y']
        self.S_ulip_z = offRegsB['S_ulip-z']
        self.S_nose_y = offRegsB['S_nose-y']
        self.S_nose_z = offRegsB['S_nose-z']
    
    def forward(self, z):
        h = self.offRegA(z)
        by = self.S_brows_y(h)
        bz = self.S_brows_z(h)
        ey = self.S_eyes_y(h)
        ez = self.S_eyes_z(h)
        ly = self.S_llip_y(h)
        lz = self.S_llip_z(h)
        uy = self.S_ulip_y(h)
        uz = self.S_ulip_z(h)
        ny = self.S_nose_y(h)
        nz = self.S_nose_z(h)
        cat = torch.cat((by, bz, ey, ez, ly, lz, uy, uz, ny, nz), 1)
        return torch.reshape(cat, (-1, 5, 2))


def standardize_part_shape(part_shapes):
    # part_shapes: [n, n_vert, 3]  n -- batch size
    y_max, _ = torch.max(part_shapes[...,1], dim=1) # [n, 1]
    y_min, _ = torch.min(part_shapes[...,1], dim=1) # [n, 1]
    y_offsets = (y_max + y_min) / 2 # [n, 1]
    z_max, _ = torch.max(part_shapes[...,2], dim=1) # [n, 1]
    z_min, _ = torch.min(part_shapes[...,2], dim=1) # [n, 1]
    z_offsets = (z_max + z_min) / 2 # [n, 1]
    batch_size = part_shapes.shape[0]
    for i in range(batch_size):
        part_shapes[i, :, 1] -= y_offsets[i]
        part_shapes[i, :, 2] -= z_offsets[i]
    return part_shapes, y_offsets, z_offsets



def load_coefficients(coeff_path):
    matlab = sio.loadmat(coeff_path)
    id_ = np.reshape(matlab['id'], (1,-1))
    tex_ = np.reshape(matlab['tex'], (1,-1))
    exp_ = np.reshape(matlab['exp'], (1,-1))
    rot_ = np.reshape(matlab['angle'], (1,-1))
    trans_ = np.reshape(matlab['trans'], (1,-1))
    gamma_ = np.reshape(matlab['gamma'], (1,-1))
    return id_ , tex_ , exp_ , rot_ , trans_, gamma_


def parsing2color(parsing):
    """
    Convert parsing label to color for visualization
    -----------------------------------------
    0: N/A         --- black   [0,0,0]
    1: face skin   --- green   [0,1,0]
    2: eye brows   --- red     [1,0,0]
    3: eyes        --- blue    [0,0,1]
    4: nose        --- yellow  [1,1,0]
    5: upper lip   --- purple  [1,0,1]
    6: lower lip   --- cyan    [0,1,1]
    """
    mapping = {
        0: np.array([0,0,0]),
        1: np.array([0,1,0]),
        2: np.array([1,0,0]),
        3: np.array([0,0,1]),
        4: np.array([1,1,0]),
        5: np.array([1,0,1]),
        6: np.array([0,1,1])
    }
    colors = np.zeros([parsing.shape[0], 3], dtype=np.float32)
    for i in range(parsing.shape[0]):
        colors[i,:] = mapping[parsing[i]]
    return colors#*255

def save_mesh_to_path(mesh, path):
    """ 
    Save Open3D mesh to a given path with format [int].obj
    --------------------------------------------------------
    If there exists multiple .obj files, avoid collision by
    increasing the integer value.
    """
    max_ = 0
    for fname in os.listdir(path):
        if fname.endswith('.obj'):
            try:
                tmp = int(fname[:-4])
                if tmp >= max_:
                    max_ = tmp + 1
            except:
                pass
    save_path = os.path.join(path, '{}.obj'.format(max_))
    mesh.compute_vertex_normals() # compute vertex normals
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    VN = np.asarray(mesh.vertex_normals) 
    with open((save_path), 'w') as f:
        f.write("# OBJ file\n")
        f.write("# Vertices: {}\n".format(len(V)))
        f.write("# Faces: {}\n".format(len(F)))
        for vid in range(len(V)):
            v = V[vid]
            vn = VN[vid]
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            f.write("vn {} {} {}\n".format(vn[0], vn[1], vn[2]))
        for p in F:
            f.write("f")
            for i in p:
                f.write(" {}".format((i + 1)))
            f.write("\n")
    print('Saved to ' + save_path)



