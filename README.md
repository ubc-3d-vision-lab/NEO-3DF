# NEO-3DF: Novel Editing-Oriented 3D Face Creation and Reconstruction Framework

## [Project Homepage](https://peizhiyan.github.io/docs/neo3df/)
[![licensebuttons by-nc](https://licensebuttons.net/l/by-nc/3.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0)

---

This repo. is the official implementation of our paper:
> Yan, P., Gregson, J., Tang, Q., Ward, R., Xu, Z., & Du, S. *“NEO-3DF: Novel Editing-Oriented 3D Face Creation and Reconstruction”*. Accepted, 2022 Asian Conference on Computer Vision (ACCV), Macau SAR, China.

## Citation
---
Please cite the following paper if you find this work is helpful to your research: 

```
@inproceedings{yan2022neo3df,
    title={NEO-3DF: Novel Editing-Oriented 3D Face Creation and Reconstruction},
    author={Peizhi Yan and James Gregson and Qiang Tang and Rabab Ward and Zhan Xu and Shan Du},
    booktitle={Asian Conference on Computer Vision (ACCV)},
    year={2022}
}
```

## Dependencies (versions are recommended but may not be necessary)
---
- Python                    == 3.6.7
- Cython                    == 0.29.22
- dlib                      == 19.22.0
- face-alignment            == 1.3.4
- facenet-pytorch           == 2.5.2
- jupyter                   == 1.0.0
- matplotlib                == 3.2.1
- networkx                  == 2.3
- ninja                     == 1.10.2
- numpy                     == 1.19.5
- nvdiffrast                == 0.2.5
- open3d                    == 0.12.0
- opencv-python             == 4.1.0.25
- pandas                    == 0.25.0
- Pillow                    == 8.3.2
- pytorch3d                 == 0.5.0
- scikit-image              == 0.17.2
- scikit-learn              == 0.24.2
- scipy                     == 1.4.1
- seaborn                   == 0.10.0
- sklearn                   == 0.0
- tensorboard               == 2.6.0
- torch                     == 1.9.0+cu111
- torchvision               == 0.10.0+cu111
- tqdm                      == 4.62.2
- trimesh                   == 3.9.19

## Preparation
---

### STEP-1: Prepare BFM (Basel Face Model)
- Please follow [link to ./BFM/README.md](./BFM/README.md) to prepare the ./BFM folder.

### STEP-2: Prepare dataset.
- Please create a folder **_./datasets_** and create two subfolders: **_./datasets/FFHQ_** and **_./datasets/CelebA_**

- Please download the FFHQ dataset ([link](https://github.com/NVlabs/ffhq-dataset)) and the CelebAMask-HQ dataset ([link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)), extract to **_./datasets/FFHQ_** and **_./datasets/CelebA_** respectively.

- Run [./data_preparation/preprocess_FFHQ.py](./data_preparation/preprocess_FFHQ.py) to resize the images and detect landmarks.

- Run [./data_preparation/preprocess_CelebA.py](./data_preparation/preprocess_CelebA.py) to resize the images and masks, and detect landmarks.

- Extract [Deep 3DMM](https://github.com/sicxu/Deep3DFaceRecon_pytorch) code to ./data_preparation/ , NOTE that, you can prepare the BFM folder using the files prepared in **STEP-1**. Then, run [./data_preparation/deep3dmm_CelebA.py](./data_preparation/deep3dmm_CelebA.py) to generate 3DMM coefficients for the images. Similarly, run [./data_preparation/deep3dmm_FFHQ.py](./data_preparation/deep3dmm_FFHQ.py) to process FFHQ dataset as well. 

- Run [./data_preparation/coeff_to_shape.py](./data_preparation/coeff_to_shape.py) to convert coefficients to numpy files.

- Run [./data_preparation/facenet_encoding.py](./data_preparation/facenet_encoding.py) to extract the FaceNet encodings of the images.

### Pre-Trained Models (Optional)

Download pre-trained models, extract to **_./saved_models/_**
- Link option 1 (Google Drive): [download](https://drive.google.com/file/d/1tJG3fI_PcgRIn6whwPIuwXN0pxuhJRBn/view?usp=sharing)
- Link option 2 (UBC ECE server): [download](https://people.ece.ubc.ca/yanpz/ACCV2022/saved_models.zip)


## Train
---

### Stage-1: Train the VAEs

- Run [./train/train_vae_overall.ipynb](./train/train_vae_overall.ipynb) to train the VAE for the overall shape. Save the trained model to ./saved_models/part_vaes and ./saved_models/part_decoders

- Run [./train/train_vaes_parts.ipynb](./train/train_vaes_parts.ipynb) to train the VAE for all the five parts. Save the trained models to ./saved_models/part_vaes and ./saved_models/part_decoders

### Stage-2: Train part encoders, offset regressor, Facenet

- Run [./train/train_other.ipynb](./train/train_other.ipynb) to train the part encoders (a.k.a. disentangle networks), offset regressor, and fine-tune FaceNet.

### Stage-3: Fine-tune the entire network

- Run [./train/fine_tune_on_CelebA.ipynb](./train/fine_tune_on_CelebA.ipynb) fine-tune the network with additional guidance from CelebA's parsing masks.


## Generate Linear Mappings for Local Editing
---

### Step-1: Compute the linear mapping

- Run [./mapping/compute_FFHQ_mean_shape_and_latent.ipynb](./mapping/compute_FFHQ_mean_shape_and_latent.ipynb).

### Step-2: Dataset measurement

- Run all the code in [./mapping/measure/](./mapping/measure) to generate the measured features (e.g., nose height, nose bridge width, etc.).

### Step-3: Compute the linear mappings for each part

- Run all the code start with **mapping** in [./mapping/](./mapping/) to generate the linear mappings for local control/editing. 



## Local Editing Demo
---

- Run [./local_editing_demo/run.py](./local_editing_demo/run.py) to try our local 3D face editing system. 



## Automatic Shape Adjusting with Differentiable ARAP-based Blending
---

Download pre-computed inverse A and save it to ./automatic_shape_adjusting: [download](https://people.ece.ubc.ca/yanpz/ACCV2022/iA.npy)

- [./automatic_shape_adjusting/arap_demo.py](./automatic_shape_adjusting/arap_demo.py) demonstrates our differentiable ARAP method using an example 3D shape.

- [./automatic_shape_adjusting/shape_adjusting.ipynb](./automatic_shape_adjusting/shape_adjusting.ipynb) demonstrates the automatic shape adjusting.











## Acknowledgement
---
This work is partially based on the following works:
- Basel Fase Model (BFM): https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model
- Expression bases for BFM: https://github.com/Juyong/3DFace
- Deep 3DMM (Pytorch implementation): https://github.com/sicxu/Deep3DFaceRecon_pytorch
- FaceNet (Pytorch implementation): https://github.com/timesler/facenet-pytorch
- FaceParsing: https://github.com/hhj1897/face_parsing
- 3DMM fitting code: https://github.com/ascust/3DMM-Fitting-Pytorch
- CelebA-Mask-HQ dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- FFHQ dataset: https://github.com/NVlabs/ffhq-dataset


## Contact
---
Peizhi Yan (yanpz@ece.ubc.ca)



