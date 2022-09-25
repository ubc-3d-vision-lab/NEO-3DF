## Author: Peizhi
############################
## Genereate facenet encodings for all the images
## Code Used: https://github.com/timesler/facenet-pytorch

import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1

import torch

# Setup PyTorch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print('CUDA is available')
else:
    device = torch.device("cpu")
    print('CUDA is not available')


resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=224)

##########
## CelebA

data_path = '../datasets/CelebA/images224x224/{}.jpg'
save_path = '../datasets/CelebA/facenet_encodings/'
try:
    os.mkdir(save_path)
except:
    pass
save_path += '{}.npy'

f_names = os.listdir(data_path[:-6])
for idx in tqdm(range(len(f_names))):
    f_name = f_names[idx]
    if f_name.endswith('.jpg'):
        f_index = f_name[:-4]
        
        # Skip the processed files
        if os.path.isfile(save_path.format(f_index)):
            continue
        
        # Load image
        img = Image.open(data_path.format(f_index))

        # Crop image and convert to [1, C, H, W] tensor
        try:
            img_cropped = mtcnn(img)[None]
            
            # Get face embedding
            img_embedding = resnet(img_cropped)
            
            # Save face embedding
            np.save(save_path.format(f_index), img_embedding.detach().numpy())
        
        except:
            print('Bad file ', f_name)


##########
## FFHQ

data_path = '../datasets/FFHQ/images224x224/{}.png'
save_path = '../datasets/FFHQ/facenet_encodings/'
try:
    os.mkdir(save_path)
except:
    pass
save_path += '{}.npy'

f_names = os.listdir(data_path[:-6])
for idx in tqdm(range(len(f_names))):
    f_name = f_names[idx]
    if f_name.endswith('.png'):
        f_index = f_name[:-4]
        
        # Skip the processed files
        if os.path.isfile(save_path.format(f_index)):
            continue
        
        # Load image
        img = Image.open(data_path.format(f_index))

        # Crop image and convert to [1, C, H, W] tensor
        try:
            img_cropped = mtcnn(img)[None]

            # Get face embedding
            img_embedding = resnet(img_cropped)

            # Save face embedding
            np.save(save_path.format(f_index), img_embedding.detach().numpy())
        
        except:
            print('Bad file ', f_name)


