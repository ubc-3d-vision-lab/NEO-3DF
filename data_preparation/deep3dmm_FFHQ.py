"""
Original Code: https://github.com/sicxu/Deep3DFaceRecon_pytorch
Modified by: Peizhi Yan
------------------------------------------------------------
This script is to get the 3DMM coefficients of all the images 
in the FFHQ dataset using Deep 3DMM. Please follow (https://github.com/sicxu/Deep3DFaceRecon_pytorch/blob/master/data_preparation.py)
to prepare the data.
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat
from tqdm import tqdm



def get_data_path(img_path, lms_path):
    im_path = []
    lm_path = []
    for fname in os.listdir(img_path):
        if fname.endswith('.jpg') or fname.endswith('.png'):
            im_path.append(os.path.join(img_path, fname))
            lm_path.append(os.path.join(lms_path, fname[:-4] + '.txt'))
    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True, im_save_path=None):
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]

    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


class Options:
    def __init__(self):
        self.add_image=True
        self.bfm_folder='../BFM'
        self.bfm_model='../BFM_model_front.mat'
        self.camera_d=10.0
        self.center=112.0
        self.checkpoints_dir='./checkpoints'
        self.dataset_mode=None
        self.ddp_port='12355'
        self.display_per_batch=True
        self.epoch='20'
        #self.eval_batch_nums=float("inf")
        self.eval_batch_nums=10
        self.focal=1015.0
        self.gpu_ids='0'
        self.img_path='../datasets/FFHQ/images224x224/'
        self.lms_path='../datasets/FFHQ/5_landmarks224x224/'
        self.visualize_save_path='../datasets/FFHQ/bfm_fitting_results/'
        self.bfm_coeffs_save_path='../datasets/FFHQ/bfm_coeffs/'
        self.realigned_img_save_path='../Datasets/FFHQ/img-224x224-aligned/'
        self.init_path='./checkpoints/init_model/resnet50-0676ba61.pth'
        self.isTrain=False
        self.model='facerecon'
        self.name='pretrained' # the folder of pretrained model
        self.net_recon='resnet50'
        self.phase='test'
        self.suffix=''
        self.use_ddp=False
        self.use_last_fc=False
        self.verbose=False
        self.vis_batch_nums=1
        self.world_size=1
        self.z_far=15.0
        self.z_near=5.0

opt = Options()


device = torch.device('cuda:0')
torch.cuda.set_device(device)

model = create_model(opt)
model.setup(opt)
model.device = device
model.parallelize()
model.eval()
visualizer = MyVisualizer(opt)

im_path, lm_path = get_data_path(opt.img_path, opt.lms_path)
lm3d_std = load_lm3d(opt.bfm_folder) 

for i in tqdm(range(len(im_path))):
   
    img_name = im_path[i].split('/')[-1].replace('.png','').replace('.jpg','')

    im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)

    data = {
        'imgs': im_tensor,
        'lms': lm_tensor
    }
    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    visualizer.display_current_results(visuals, 0, opt.epoch, save_path=opt.visualize_save_path, 
    save_results=True, count=i, name=img_name, add_image=False)

    model.save_coeff(os.path.join(opt.bfm_coeffs_save_path, img_name+'.mat')) # save predicted coefficients


