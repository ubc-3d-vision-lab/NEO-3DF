import pickle
import numpy as np
import os
import scipy.sparse as sp
from pygcn.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)

import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
from PIL import Image

import open3d as o3d


def bfm_edges(bfm):
    """
    Peizhi
    get the list of edges in BFM mesh
    """
    def reverse_edge(e):
        # reverse the start and end vertices
        return [e[1], e[0]]
    # Get the list of all edges in the BFM mesh
    triangle_faces = bfm['tri'] - 1
    edges = []
    adj = np.zeros([35709, 35709], dtype = np.int32)
    for triangle in triangle_faces:
        # triangle edges
        e1 = [triangle[0], triangle[1]]
        e2 = [triangle[0], triangle[2]]
        e3 = [triangle[1], triangle[2]]
        tri_edges = [e1, e2, e3]
        # add edges to edges list (also avoid redundancy)
        for e in tri_edges:
            if adj[e[0], [e[1]]] == 0:
                adj[e[0], [e[1]]] = 1
            if adj[e[1], [e[0]]] == 0:
                adj[e[1], [e[0]]] = 1    
    for i in range(35709):
        for j in range(i, 35709):
            if adj[i, j] == 1:
                edges.append([i, j])
    edges = np.array(edges, dtype=np.int32)
    return edges

def bfm_adjacency_matrix(bfm, edges_path):
    """
    Peizhi
    get the adjacency matrix of BFM mesh
    """
    edges = np.load(edges_path)
    n_vertices = bfm['meanshape'].shape[1]//3
    # build graph
    idx = np.array([i for i in range(n_vertices)], dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n_vertices, n_vertices),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj    


def pad_bbox(bbox, img_wh, padding_ratio=0.2):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    size_bb = int(max(width, height) * (1+padding_ratio))
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(img_wh[0] - x1, size_bb)
    size_bb = min(img_wh[1] - y1, size_bb)

    return [x1, y1, x1+size_bb, y1+size_bb]


def mymkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    colors = np.zeros([parsing.shape[0], parsing.shape[1], 3], dtype=np.int8)
    for i in range(parsing.shape[0]):
        for j in range(parsing.shape[1]):
            colors[i,j,:] = mapping[parsing[i,j]]
    return colors*255

