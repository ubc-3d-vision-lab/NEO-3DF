Due to the copyright concern, our repo does not include the BFM model ([BFM09](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model)). Please follow [Deep 3DMM](https://github.com/sicxu/Deep3DFaceRecon_pytorch) (see **Prepare prerequisite models**) to prepare the BFM folder. The structure should look like this:
```
    BFM
    │
    └─── 01_MorphableModel.mat     (download from   https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model)
    │
    └─── Exp_Pca.bin   (download from   https://github.com/Juyong/3DFace)
    |
    └─── ...   (other files download from   https://github.com/microsoft/Deep3DFaceReconstruction/tree/master/BFM)      

```

Then, run **convert_2_pickle.py** to convert the BFM model to a single pickle file (**bfm09.pkl**). 

