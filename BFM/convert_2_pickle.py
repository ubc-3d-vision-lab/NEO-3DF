import os
from scipy.io import loadmat
from array import array
import numpy as np
import pickle

########################
# Load BFM model
########################

original_BFM = loadmat('./01_MorphableModel.mat')
shapePC = original_BFM['shapePC'] # shape basis
shapeEV = original_BFM['shapeEV'] # corresponding eigen value
shapeMU = original_BFM['shapeMU'] # mean face
texPC = original_BFM['texPC'] # texture basis
texEV = original_BFM['texEV'] # eigen value
texMU = original_BFM['texMU'] # mean texture

def LoadExpBasis():
    n_vertex = 53215
    Expbin = open('./Exp_Pca.bin','rb')
    exp_dim = array('i')
    exp_dim.fromfile(Expbin,1)
    expMU = array('f')
    expPC = array('f')
    expMU.fromfile(Expbin,3*n_vertex)
    expPC.fromfile(Expbin,3*exp_dim[0]*n_vertex)
    expPC = np.array(expPC)
    expPC = np.reshape(expPC,[exp_dim[0],-1])
    expPC = np.transpose(expPC)
    expEV = np.loadtxt('./std_exp.txt')
    return expPC,expEV

expPC,expEV = LoadExpBasis()

# transfer BFM09 to our face model

idBase = shapePC*np.reshape(shapeEV,[-1,199])
idBase = idBase/1e5 # unify the scale to decimeter
idBase = idBase[:,:199] # use only first 80 basis

exBase = expPC*np.reshape(expEV,[-1,79])
exBase = exBase/1e5 # unify the scale to decimeter
exBase = exBase[:,:79] # use only first 64 basis

texBase = texPC*np.reshape(texEV,[-1,199])
texBase = texBase[:,:199] # use only first 80 basis

# our face model is cropped align face landmarks which contains only 35709 vertex.
# original BFM09 contains 53490 vertex, and expression basis provided by JuYong contains 53215 vertex.
# thus we select corresponding vertex to get our face model.

index_exp = loadmat('./BFM_front_idx.mat')
index_exp = index_exp['idx'].astype(np.int32) - 1 #starts from 0 (to 53215)

index_shape = loadmat('./BFM_exp_idx.mat')
index_shape = index_shape['trimIndex'].astype(np.int32) - 1 #starts from 0 (to 53490)
index_shape = index_shape[index_exp]


idBase = np.reshape(idBase,[-1,3,199])
idBase = idBase[index_shape,:,:]
idBase = np.reshape(idBase,[-1,199])

texBase = np.reshape(texBase,[-1,3,199])
texBase = texBase[index_shape,:,:]
texBase = np.reshape(texBase,[-1,199])

exBase = np.reshape(exBase,[-1,3,79])
exBase = exBase[index_exp,:,:]
exBase = np.reshape(exBase,[-1,79])

meanshape = np.reshape(shapeMU,[-1,3])/1e5
meanshape = meanshape[index_shape,:]
meanshape = np.reshape(meanshape,[1,-1])

meantex = np.reshape(texMU,[-1,3])
meantex = meantex[index_shape,:]
meantex = np.reshape(meantex,[1,-1])

# other info contains triangles, region used for computing photometric loss,
# region used for skin texture regularization, and 68 landmarks index etc.
other_info = loadmat('./facemodel_info.mat')
frontmask2_idx = other_info['frontmask2_idx']
skinmask = other_info['skinmask']
keypoints = other_info['keypoints']
point_buf = other_info['point_buf']
tri = other_info['tri']
tri_mask2 = other_info['tri_mask2']


bfm = {}
bfm['keypoints'] = keypoints
bfm['meanshape'] = meanshape
bfm['idBase'] = idBase
bfm['exBase'] = exBase
bfm['meantex'] = meantex
bfm['texBase'] = texBase
bfm['tri'] = tri
bfm['point_buf'] = point_buf



with open('bfm09.pkl', 'wb') as handle:
    pickle.dump(bfm, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    
    
