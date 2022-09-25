"""
Author: Peizhi Yan
--------------------
resize images from 1024^2 to 224^2
generate five landmarks for each face image
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


data_path = '../datasets/FFHQ/images224x224/' # need to change it

img_save_path = '../datasets/FFHQ/images224x224/'
lms_save_path = '../datasets/FFHQ/5_landmarks224x224/'

try:
    os.mkdir(img_save_path) # to save resized images
except:
    print(img_save_path + '  exists')
try:
    os.mkdir(lms_save_path) # to save landmarks
except:
    print(lms_save_path + '  exists')


f_names = os.listdir(data_path)

for idx in tqdm(range(len(f_names))):
    f_name = f_names[idx]
    if f_name.endswith('.png'):
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
            




