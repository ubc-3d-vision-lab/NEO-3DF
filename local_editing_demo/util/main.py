##################################################
##################################################
## Author: Peizhi Yan
## Date:
##
##################################################
##################################################


import numpy as np
from PIL import Image
import os
import sys

# Pytorch 1.9
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

import open3d as o3d

# facenet-pytorch 2.5.2
from facenet_pytorch import MTCNN, InceptionResnetV1

from util.helpers import *

#######################################
## Setup PyTorch to use CPU
device = torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"]=""






## Load the BFM model
import pickle
with open('../BFM/bfm09.pkl', 'rb') as f:
    bfm = pickle.load(f)
print('BFM model loaded\n')

## Triangal Facets
Faces = bfm['tri'] - 1 ## -1 is critical !!!

## Load the face parsing labels (per-vertex)
vert_labels = np.load('../BFM/bfm_vertex_labels.npy')

seg_colors = parsing2color(vert_labels) # get segmentation labels as colors

# find the vertices of part
label_map = {
    'skin': 1,
    'eye_brow': 2,
    'eye': 3,
    'nose': 4,
    'u_lip': 5,
    'l_lip': 6
}
part_vertices = {
    'S_overall':[],
    'S_eyebrows':[],
    'S_eyes':[],
    'S_llip':[],
    'S_nose':[],
    'S_ulip':[]
}
part_vertices_rest = []
for idx in range(len(vert_labels)):
    part_vertices['S_overall'].append(idx)
    if vert_labels[idx] in [label_map['eye_brow']]:
        part_vertices['S_eyebrows'].append(idx)
    if vert_labels[idx] in [label_map['eye']]:
        part_vertices['S_eyes'].append(idx)
    if vert_labels[idx] in [label_map['l_lip']]:
        part_vertices['S_llip'].append(idx)
    if vert_labels[idx] in [label_map['u_lip']]:
        part_vertices['S_ulip'].append(idx)
    if vert_labels[idx] in [label_map['nose']]:
        part_vertices['S_nose'].append(idx)
    else:
        part_vertices_rest.append(idx)

for key in part_vertices:
    part_vertices[key] = np.array(part_vertices[key])
    print(key, ' n_vert: ', len(part_vertices[key]))

part_vertices_rest = np.array(part_vertices_rest)

######################
## Load the mappings 
M_overall = torch.from_numpy(np.load('../saved_models/mappings/S_overall.npy')).to(device)
M_nose = torch.from_numpy(np.load('../saved_models/mappings/S_nose.npy')).to(device)
M_eyebrows = torch.from_numpy(np.load('../saved_models/mappings/S_eyebrows.npy')).to(device)
M_eyes = torch.from_numpy(np.load('../saved_models/mappings/S_eyes.npy')).to(device)
M_ulip = torch.from_numpy(np.load('../saved_models/mappings/S_ulip.npy')).to(device)
M_llip = torch.from_numpy(np.load('../saved_models/mappings/S_llip.npy')).to(device)
def mapping(M, f):
    # input (torch tensor):   measurement with size [1, f_dim]
    # output (torch tensor):  latent offset with size [1, z_dim]
    return f @ M


#########################
## Latent dimensions
latent_dims = {}
latent_dims['S_overall'] = 30
latent_dims['S_eyebrows'] = 10
latent_dims['S_eyes'] = 10
latent_dims['S_llip'] = 10
latent_dims['S_ulip'] = 10
latent_dims['S_nose'] = 10





#################
## Part Decoders
MODEL_PATH = '../saved_models/part_decoders/{}'
part_decoders = {}
for key in part_vertices:
    model = ShapeDecoder
    part_decoders[key] = model(latent_dim=latent_dims[key], n_vert=len(part_vertices[key])).eval().to(device)
    
    # Load pre-trained parameters
    part_decoders[key].load_state_dict(torch.load(MODEL_PATH.format(key), map_location=device))
    print('Decoder {} loaded'.format(key))
    
    # freeze the network parameters
    for param in part_decoders[key].parameters():
        param.requires_grad = False

##############################
## Offset Prediction Module
##############################
offsetKeys = {'S_eyebrows': 0, 
              'S_eyes': 1, 
              'S_llip': 2, 
              'S_ulip': 3, 
              'S_nose': 4}
offsetRegressorA = OffsetRegressorA().to(device)
offsetRegressorsB = {}
for key in offsetKeys:
    offsetRegressorsB[key+'-y'] = OffsetRegressorB()
    offsetRegressorsB[key+'-z'] = OffsetRegressorB()
offsetRegressor = OffsetRegressor(offsetRegressorA, offsetRegressorsB).eval().to(device)
## Load the pre-trained decoder weights
offsetRegressor.load_state_dict(torch.load('../saved_models/offset_regressor', map_location=device))



""" ARAP """
transition_region = list(np.load('../BFM/trans_region_r0.3.npy'))
boundary_vertices = np.load('../BFM/boundary_vert_ids.npy').tolist()
static_ids = boundary_vertices
for idx in static_ids:
    if idx in transition_region:
        transition_region.remove(idx)
handle_ids = [idx for idx in range(35709)] # handle vertices are all vertices except the vertices in transition region
handle_ids = list(set(handle_ids).difference(set(transition_region)))
handle_ids = list(set(handle_ids).union(set(static_ids)))

def ARAP_optimization(V_overall, V_new, handle_ids):
    ## start with the mean shape
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V_overall) # dtype vector3d (float)
    mesh.triangles = o3d.utility.Vector3iVector(Faces) # dtype vector3i (int)
    ###########
    ### ARAP
    vertices = mesh.vertices
    ## set the handle vertices to be part vertices
    handle_pos = []
    for id in handle_ids:
        handle_pos.append(V_new[id])
    constraint_ids = o3d.utility.IntVector(handle_ids)
    constraint_pos = o3d.utility.Vector3dVector(handle_pos)
    _original_stdout = sys.stdout
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh = mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=2) 
    V_optimized = np.asarray(mesh.vertices, dtype=np.float32)
    return V_optimized



def o3d_form_mesh(V, T, Faces):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(V) # dtype vector3d (float)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(Faces) # dtype vector3i (int)
    
    ###############
    ### Smoothing
    o3d_mesh = o3d_mesh.filter_smooth_laplacian(1, 0.5,
                filter_scope=o3d.geometry.FilterScope.Vertex)

    if T is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(T) # dtype vector3i (int)
    o3d_mesh.compute_vertex_normals() # computing normal will give specular effect while rendering

    return o3d_mesh


def o3d_render(V, T, Faces, width=512, height=512):
    ###############################
    ## Visualize the render result
    o3d_mesh = o3d_form_mesh(V, T, Faces)
    o3d_mesh.compute_triangle_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible = False)
    
    opt = vis.get_render_option()
    
    ## show Normal if no texture
    if T is None:
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
    
    vis.add_geometry(o3d_mesh)

    ## Smooth shading
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

    #depth = vis.capture_depth_float_buffer(True)
    image = vis.capture_screen_float_buffer(True)

    return o3d_mesh, image


def reconstruction(x):
    """ Reconstruction from image """
    ## x: [batch_size, 3, 224, 224]
    ## returns: 
    ##   shapes: [batch_size, 35709, 3]
    batch_size = x.shape[0]
    
    preds = {}
    part_latents = {}
    img_encodings = facenet(x)
    for key in latent_dims:
        part_mu, part_sigma = disentangleNets[key](img_encodings)
        part_latents[key] = part_mu
        preds[key] = part_decoders[key](part_mu)

    pred_offsets = offsetRegressor(img_encodings)
    for key in offsetKeys:
        for i in range(batch_size):
            preds[key][i,:,1:] += pred_offsets[i, offsetKeys[key], :]

    shapes_overall = torch.zeros([batch_size, 35709, 3], dtype=torch.float32).to(device)
    shapes_composed = torch.zeros([batch_size, 35709, 3], dtype=torch.float32).to(device)

    shapes_overall = preds['S_overall']
    for key in latent_dims:
        shapes_composed[:,part_vertices[key],:] = preds[key]
        
    return shapes_overall, shapes_composed, part_latents, pred_offsets


def reconstruction_from_latents(part_latents):
    """ Reconstruction from latents """
    preds = {}
    for key in latent_dims:
        part_mu = part_latents[key]
        preds[key] = part_decoders[key](part_mu)
    
    ## use S_overall to compute part offsets
    shapes_overall = torch.zeros([1, 35709, 3], dtype=torch.float32).to(device)
    shapes_overall = preds['S_overall']
    shapes_composed = torch.zeros([1, 35709, 3], dtype=torch.float32).to(device)
    #shapes_composed = preds['S_overall']
    pred_offsets = torch.zeros([1,5,2], dtype=torch.float32).to(device)
    for key in offsetKeys:
        _, y_offsets, z_offsets = standardize_part_shape(shapes_overall[:,part_vertices[key],:])
        pred_offsets[0, offsetKeys[key], 0] = y_offsets[0]
        pred_offsets[0, offsetKeys[key], 1] = z_offsets[0]        
        preds[key][0,:,1:] += pred_offsets[0, offsetKeys[key], :]

    for key in latent_dims:
        shapes_composed[:,part_vertices[key],:] = preds[key]

    return shapes_overall, shapes_composed, part_latents, pred_offsets



def load_img(img_path):
    batch_x = torch.zeros([1, 3, 224, 224], dtype=torch.float32).to(device)
    img = Image.open(img_path)
    img = img.resize((224,224))
    img = np.asarray(img, dtype=np.float32)
    img = torch.from_numpy(img)
    img = img.permute(2,0,1)
    batch_x[0] = img
    return batch_x


 


#################################
## Face Editors
#################################

def overall_editor( maximum_facial_width,
                    madibular_width,
                    upper_facial_depth,
                    middle_facial_depth,
                    lower_facial_depth,
                    facial_height,
                    upper_facial_height,
                    lower_facial_height,
                    loaded_latent,
                    V):

    # measurement vector
    f = torch.tensor([[maximum_facial_width,
                madibular_width,
                upper_facial_depth,
                middle_facial_depth,
                lower_facial_depth,
                facial_height,
                upper_facial_height,
                lower_facial_height]]).to(device)
    
    # predict latent offset
    latent_offset = mapping(M_overall, f).to(device)

    # generate new latent vector
    new_latent = loaded_latent + latent_offset
        
    # generate new shape
    new_part_shape = part_decoders['S_overall'](new_latent)[0].detach().cpu().numpy()
    
    # optimize shape
    V_new = np.copy(V)
    V_new[part_vertices_rest] = new_part_shape[part_vertices_rest]
    V = ARAP_optimization(V, V_new, handle_ids)
    
    return V
        


def nose_editor(    nose_tip_y, 
                    nose_tip_z, 
                    nose_height, 
                    nose_width, 
                    tip_width, 
                    bridge_width,
                    loaded_latent,
                    pred_offsets,
                    V):

    # measurement vector
    f = torch.tensor([[nose_tip_y, 
                        nose_tip_z, 
                        nose_height, 
                        nose_width, 
                        tip_width, 
                        bridge_width]]).to(device)
    
    # predict latent offset
    latent_offset = mapping(M_nose, f).to(device)

    # generate new latent vector
    new_latent = loaded_latent + latent_offset
        
    # generate new shape
    new_part_shape = part_decoders['S_nose'](new_latent)
    new_part_shape[0,:,1:] += pred_offsets[0, offsetKeys['S_nose'], :]
    new_part_shape = new_part_shape[0].detach().cpu().numpy()

    # optimize shape
    V_new = np.copy(V)
    V_new[part_vertices['S_nose']] = new_part_shape
    V = ARAP_optimization(V, V_new, handle_ids)
    
    return V



def eyebrows_editor(front_thickness, tail_thickness, eyebrow_length, curve_strength,
                    loaded_latent,
                    pred_offsets,
                    V):

    # measurement vector
    f = torch.tensor([[front_thickness, tail_thickness, eyebrow_length, curve_strength]]).to(device)
    
    # predict latent offset
    latent_offset = mapping(M_eyebrows, f).to(device)

    # generate new latent vector
    new_latent = loaded_latent + latent_offset
        
    # generate new shape
    new_part_shape = part_decoders['S_eyebrows'](new_latent)
    new_part_shape[0,:,1:] += pred_offsets[0, offsetKeys['S_eyebrows'], :]
    new_part_shape = new_part_shape[0].detach().cpu().numpy()

    # optimize shape
    V_new = np.copy(V)
    V_new[part_vertices['S_eyebrows']] = new_part_shape
    V = ARAP_optimization(V, V_new, handle_ids)
    
    return V


def eyes_editor(pupils_distance, eye_height, canthus_distance, medial_canthus_y, lateral_canthus_y,
                    loaded_latent,
                    pred_offsets,
                    V):

    # measurement vector
    f = torch.tensor([[pupils_distance, eye_height, canthus_distance, medial_canthus_y, lateral_canthus_y]]).to(device)
    
    # predict latent offset
    latent_offset = mapping(M_eyes, f).to(device)

    # generate new latent vector
    new_latent = loaded_latent + latent_offset
        
    # generate new shape
    new_part_shape = part_decoders['S_eyes'](new_latent)
    new_part_shape[0,:,1:] += pred_offsets[0, offsetKeys['S_eyes'], :]
    new_part_shape = new_part_shape[0].detach().cpu().numpy()

    # optimize shape
    V_new = np.copy(V)
    V_new[part_vertices['S_eyes']] = new_part_shape
    V = ARAP_optimization(V, V_new, handle_ids)
    
    return V


def ulip_editor(labial_fissure_width, upper_lip_height, upper_lip_width, upper_lip_end_height,
                    loaded_latent,
                    pred_offsets,
                    V):

    # measurement vector
    f = torch.tensor([[labial_fissure_width, upper_lip_height, upper_lip_width, upper_lip_end_height]]).to(device)
    
    # predict latent offset
    latent_offset = mapping(M_ulip, f).to(device)

    # generate new latent vector
    new_latent = loaded_latent + latent_offset
        
    # generate new shape
    new_part_shape = part_decoders['S_ulip'](new_latent)
    new_part_shape[0,:,1:] += pred_offsets[0, offsetKeys['S_ulip'], :]
    new_part_shape = new_part_shape[0].detach().cpu().numpy()

    # optimize shape
    V_new = np.copy(V)
    V_new[part_vertices['S_ulip']] = new_part_shape
    V = ARAP_optimization(V, V_new, handle_ids)
    
    return V


def llip_editor(lower_lip_height, lower_lip_width, lower_lip_end_height,
                    loaded_latent,
                    pred_offsets,
                    V):

    # measurement vector
    f = torch.tensor([[lower_lip_height, lower_lip_width, lower_lip_end_height]]).to(device)
    
    # predict latent offset
    latent_offset = mapping(M_llip, f).to(device)

    # generate new latent vector
    new_latent = loaded_latent + latent_offset

    # generate new shape
    new_part_shape = part_decoders['S_llip'](new_latent)
    new_part_shape[0,:,1:] += pred_offsets[0, offsetKeys['S_llip'], :]
    new_part_shape = new_part_shape[0].detach().cpu().numpy()

    # optimize shape
    V_new = np.copy(V)
    V_new[part_vertices['S_llip']] = new_part_shape
    V = ARAP_optimization(V, V_new, handle_ids)
    
    return V


