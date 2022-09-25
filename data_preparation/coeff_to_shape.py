"""
Convert the 3DMM coefficients to shapes
"""

import sys
import gc
import os
import torch
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import argparse
import face_alignment
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio

from array import array

# From models.py
import torch
import torch.nn as nn
import numpy as np
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
import torchvision.transforms as transforms
from PIL import Image

import torch.nn.functional as F

import pickle

import open3d as o3d

from time import time

# Setup PyTorch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print('CUDA is available')
else:
    device = torch.device("cpu")
    print('CUDA is not available')

sys.path.append('../')
from models.ReconModel import ReconModel

########################
# Load BFM model
########################

with open('../BFM/bfm09.pkl', 'rb') as f:
    bfm = pickle.load(f)


bfm['idBase'] = bfm['idBase'][...,:80]   # only use 80 coefficients
bfm['texBase'] = bfm['texBase'][...,:80] # only use 80 coefficients
bfm['exBase'] = bfm['exBase'][...,:64] # only use 64 coefficients


print('BFM model loaded')

TAR_SIZE = 224 # size for rendering window


model = ReconModel(bfm, img_size=TAR_SIZE)
model.eval()
model.cuda()


def load_coefficients(coeff_path):
    matlab = sio.loadmat(coeff_path)
    id_ = np.reshape(matlab['id'], (1,-1))
    tex_ = np.reshape(matlab['tex'], (1,-1))
    exp_ = np.reshape(matlab['exp'], (1,-1))
    rot_ = np.reshape(matlab['angle'], (1,-1))
    trans_ = np.reshape(matlab['trans'], (1,-1))
    gamma_ = np.reshape(matlab['gamma'], (1,-1))
    return id_ , tex_ , exp_ , rot_ , trans_, gamma_



shape_save_path = '../datasets/CelebA/raw_bfm_shape/'
albedo_save_path = '../datasets/CelebA/raw_bfm_albedo/'
color_save_path = '../datasets/CelebA/raw_bfm_color/'
try:
    os.mkdir(shape_save_path)
except:
    pass
try:
    os.mkdir(albedo_save_path)
except:
    pass
try:
    os.mkdir(color_save_path)
except:
    pass
shape_save_path_ = shape_save_path + '{}.npy'
albedo_save_path_ = albedo_save_path + '{}.npy'
color_save_path_ = color_save_path + '{}.npy'


#####################
## CelebA dataset

img_index = 0

# Natural expression
exp_tensor = torch.zeros((1,64), dtype=torch.float32, requires_grad=False, device='cuda')

for fname in tqdm(os.listdir('../datasets/CelebA/images224x224/')):
    if fname.endswith('.jpg'):
        
        img_index = fname[:-4]

        img_path = '../datasets/CelebA/images224x224/{}.jpg'.format(img_index)
        coeff_path = '../datasets/CelebA/bfm_fitting_coeffs/{}.mat'.format(img_index)

        shape_save_path = shape_save_path_.format(img_index)
        albedo_save_path = albedo_save_path_.format(img_index)
        color_save_path = color_save_path_.format(img_index)

        ## Load the deep3dmm(2019) fitted bfm coefficients
        id_ , tex_ , exp_ , rot_ , trans_, gamma_ = load_coefficients(coeff_path)

        
        id_tensor = torch.tensor(id_, dtype=torch.float32, requires_grad=False, device='cuda')
        tex_tensor = torch.tensor(tex_, dtype=torch.float32, requires_grad=False, device='cuda')
        rot_tensor = torch.tensor(rot_, dtype=torch.float32, requires_grad=False, device='cuda')
        gamma_tensor = torch.tensor(gamma_, dtype=torch.float32, requires_grad=False, device='cuda')

        face_shape = model.Shape_formation(id_coeff=id_tensor, ex_coeff=exp_tensor)
        face_texture = model.Texture_formation(tex_coeff=tex_tensor) # albedo only
        
        face_norm = model.Compute_norm(face_shape)
        rotation = model.Compute_rotation_matrix(rot_tensor)
        face_norm_r = face_norm.bmm(rotation)
        face_color = model.Illumination_layer(face_texture, face_norm_r, gamma_tensor)
        
        
        # Save as Numpy array
        np.save(shape_save_path, face_shape.detach().cpu().numpy())
        np.save(albedo_save_path, face_texture.detach().cpu().numpy())
        np.save(color_save_path, face_color.detach().cpu().numpy())
        
        # Release CUDA memory
        del id_tensor, tex_tensor, face_shape, face_texture
        torch.cuda.empty_cache()
        
        # Release RAM
        gc.collect()
        



#####################
## FFHQ dataset

shape_save_path = '../datasets/FFHQ/raw_bfm_shape/'
albedo_save_path = '../datasets/FFHQ/raw_bfm_albedo/'
color_save_path = '../datasets/FFHQ/raw_bfm_color/'
try:
    os.mkdir(shape_save_path)
except:
    pass
try:
    os.mkdir(albedo_save_path)
except:
    pass
try:
    os.mkdir(color_save_path)
except:
    pass
shape_save_path_ = shape_save_path + '{}.npy'
albedo_save_path_ = albedo_save_path + '{}.npy'
color_save_path_ = color_save_path + '{}.npy'


img_index = 0

# Natural expression
exp_tensor = torch.zeros((1,64), dtype=torch.float32, requires_grad=False, device='cuda')

for fname in tqdm(os.listdir('../datasets/FFHQ/images224x224/')):
    if fname.endswith('.png'):
        
        img_index = fname[:-4]

        img_path = '../datasets/FFHQ/images224x224/{}.jpg'.format(img_index)
        coeff_path = '../datasets/FFHQ/bfm_fitting_coeffs/{}.mat'.format(img_index)

        shape_save_path = shape_save_path_.format(img_index)
        albedo_save_path = albedo_save_path_.format(img_index)
        color_save_path = color_save_path_.format(img_index)
        
        ## Load the deep3dmm(2019) fitted bfm coefficients
        id_ , tex_ , exp_ , rot_ , trans_, gamma_ = load_coefficients(coeff_path)

        
        id_tensor = torch.tensor(id_, dtype=torch.float32, requires_grad=False, device='cuda')
        tex_tensor = torch.tensor(tex_, dtype=torch.float32, requires_grad=False, device='cuda')
        rot_tensor = torch.tensor(rot_, dtype=torch.float32, requires_grad=False, device='cuda')
        gamma_tensor = torch.tensor(gamma_, dtype=torch.float32, requires_grad=False, device='cuda')

        face_shape = model.Shape_formation(id_coeff=id_tensor, ex_coeff=exp_tensor)
        face_texture = model.Texture_formation(tex_coeff=tex_tensor) # albedo only
        
        face_norm = model.Compute_norm(face_shape)
        rotation = model.Compute_rotation_matrix(rot_tensor)
        face_norm_r = face_norm.bmm(rotation)
        face_color = model.Illumination_layer(face_texture, face_norm_r, gamma_tensor)
        
        
        # Save as Numpy array
        np.save(shape_save_path, face_shape.detach().cpu().numpy())
        np.save(albedo_save_path, face_texture.detach().cpu().numpy())
        np.save(color_save_path, face_color.detach().cpu().numpy())
        
        # Release CUDA memory
        del id_tensor, tex_tensor, face_shape, face_texture
        torch.cuda.empty_cache()
        
        # Release RAM
        gc.collect()

