"""
Author: Peizhi Yan
"""

import sys
import gc
import os
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import argparse
import face_alignment
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
from array import array
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



class Encoder(nn.Module):
    def __init__(self, latent_dim, n_vert):
        super(Encoder, self).__init__()
        self.n_vert = n_vert
        self.fc1 = nn.Linear(3*self.n_vert, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3_mu = nn.Linear(128, latent_dim)
        self.fc3_std = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        #h = x
        h = torch.flatten(x, start_dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mu = self.fc3_mu(h)
        sigma = torch.exp(self.fc3_std(h))
        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, n_vert):
        super(Decoder, self).__init__()
        self.n_vert = n_vert
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 3*self.n_vert)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        x_ = self.fc3(h)
        return torch.reshape(x_, [-1, self.n_vert, 3])
    
    
class VAE(nn.Module):
    def __init__(self, latent_dim, n_vert, device):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, n_vert)
        self.decoder = Decoder(latent_dim, n_vert)
        
        self.normal = torch.distributions.Normal(0, 1) # a normal distribution with mean = 0, std = 1
        if device.type == 'cuda':
            self.normal.loc = self.normal.loc.cuda() # use CUDA GPU for sampling
            self.normal.scale = self.normal.scale.cuda()
        self.kl_loss = 0 # KL divergence loss
        
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = mu + sigma * self.normal.sample(mu.size())
        self.kl_loss = torch.mean(torch.sum(-0.5 * (1 + sigma - mu**2 - torch.exp(sigma)), dim=1))
        x_ = self.decoder(z)
        return x_


class VariationalDisentangleModule(nn.Module):
    ## Input:  512-dimensional image encoding
    ## Output: Mean and Std. vectors
    def __init__(self, image_encoding_dims, part_encoding_dims):
        super(VariationalDisentangleModule, self).__init__()
        self.fc1 = nn.Linear(image_encoding_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mu = nn.Linear(64, part_encoding_dims)
        self.fc3_std = nn.Linear(64, part_encoding_dims)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        mu = self.fc3_mu(h)
        sigma = torch.exp(self.fc3_std(h))
        return mu, sigma
    
    
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


