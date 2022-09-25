"""
Author: Peizhi Yan
--------------------
- resize images and masks from 1024^2 to 224^2
- convert the original parsing masks use the following labels:
    0: other
    1: face skin
    2: eye brows
    3: eyes
    4: nose
    5: upper lip
    6: lower lip
- generate five landmarks for each face image
"""

import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import face_alignment

# load the face alighment model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)

def avg_location(lms, indices):
    # lms: [68, 2] --- 68 2D-landmarks
    x_ = 0
    y_ = 0
    for i in range(len(indices)):
        x_ += lms[indices[i]-1, 0]
        y_ += lms[indices[i]-1, 1]
    return (x_/len(indices), y_/len(indices))


data_path = '../datasets/CelebA/CelebA-HQ-img/'
img_save_path = '../datasets/CelebA/images224x224/'
lms_save_path = '../datasets/CelebA/5_landmarks224x224/'
lms68_save_path = '../datasets/CelebA/68_landmarks224x224/'
mask_save_path = '../datasets/CelebA/parsing_masks/'
original_masks = '../datasets/CelebA/CelebAMask-HQ-mask-anno/{}.png'

try:
    os.mkdir(img_save_path) # to save resized images
except:
    print(img_save_path + '  exists')
try:
    os.mkdir(lms_save_path) # to save landmarks
except:
    print(lms_save_path + '  exists')
try:
    os.mkdir(lms68_save_path) # to save landmarks
except:
    print(lms68_save_path + '  exists')
lms68_save_path = lms68_save_path + '{}.npy'
try:
    os.mkdir(mask_save_path) # to save parsing masks
except:
    print(mask_save_path + '  exists')
mask_save_path = mask_save_path + '{}.npy'



f_names = os.listdir(data_path)

for idx in tqdm(range(len(f_names))):
    f_name = f_names[idx]
    if f_name.endswith('.jpg'):
        # skip the processed files
        if os.path.isfile(os.path.join(lms_save_path,  f_name[:-4] + '.txt')):
            continue
        
        # load image
        img = cv2.imread(os.path.join(data_path, f_name))
        img_224x224 = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img_224x224, cv2.COLOR_BGR2RGB)
        
        # detect 68 landmarks
        lms = fa.get_landmarks_from_image(img)
        
        if lms == None:
            # skip the images that no face can be detected
            print('bad file ', f_name)
            continue
        
        lms = lms[0][..., :2]
        
        # save 68 landmarks
        np.save(lms68_save_path.format(f_name[:-4]), lms)
        
        # save the resized image
        cv2.imwrite(os.path.join(img_save_path, f_name), img_224x224)

        # compute 5 landmarks (for face image re-align)
        l_eye = (0,0)
        r_eye = (0,0)
        nose = (0,0)
        l_mouth = (0,0)
        r_mouth = (0,0)
        l_eye = avg_location(lms, [37,38,39,40,41,42])
        r_eye = avg_location(lms, [43,44,45,46,47,48])
        nose = avg_location(lms, [32,33,34,35,36])
        l_mouth = avg_location(lms, [49])
        r_mouth = avg_location(lms, [55])
        with open(os.path.join(lms_save_path, f_name[:-4] + '.txt'), 'w') as out_file:
            out_file.write("{:.2f} {:.2f}\n".format(l_eye[0], l_eye[1]))
            out_file.write("{:.2f} {:.2f}\n".format(r_eye[0], r_eye[1]))
            out_file.write("{:.2f} {:.2f}\n".format(nose[0], nose[1]))
            out_file.write("{:.2f} {:.2f}\n".format(l_mouth[0], l_mouth[1]))
            out_file.write("{:.2f} {:.2f}\n".format(r_mouth[0], r_mouth[1]))
            


def get_mask(padded_name, mask_label):
    mask = cv2.imread(original_masks.format(padded_name+'_'+mask_label), 0)
    mask = np.where(cv2.resize(mask, (224,224)) > 0, 1, 0)
    return mask

"""
Process all the masks
"""
processing_list = {
    1: ['skin'],
    2: ['l_brow', 'r_brow'],
    3: ['l_eye', 'r_eye'],
    4: ['nose'],
    5: ['u_lip'],
    6: ['l_lip'],
    0: ['mouth'] # to clear in inner mouth
}

img_indices = []
for fname in os.listdir('../datasets/CelebA/images224x224/'):
    if fname.endswith('.jpg'):
        img_indices.append(fname[:-4])

for idx_str in tqdm(img_indices):
    idx = int(idx_str)
    padded_name = '{0:05d}'.format(idx)

    #if os.path.exists(mask_save_path.format(idx)):
    #    continue
    
    new_mask = np.zeros([224,224], dtype=np.int8)
    for label in processing_list:
        one_mask = np.zeros([224,224], dtype=np.int8)
        for old_label in processing_list[label]:
            try:
                one_mask += get_mask(padded_name, old_label)
            except:
                pass
        mask = one_mask * label
        new_mask = new_mask * (1-one_mask) + mask
        
        

    np.save(mask_save_path.format(idx), new_mask)
    

