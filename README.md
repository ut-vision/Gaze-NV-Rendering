# NV-Gaze

This is the official repository for the implementation of creating MPII-NV in [Learning-by-Novel-View-Synthesis for Full-Face Appearance-based 3D Gaze Estimation](https://arxiv.org/abs/2201.07927).

Check []() for the single-view and multi-view reconstruction of XGaze.

# The pipeline of Reconstruction and Rendering
you need to first run the 3D face reconstruction following the [readme](./3ddfa/README.md) in the "3ddfa" subdirectory. After reconstructing the MPII to 3D obj files, you can run the rendering following the below steps.
Note that the 3DDFA is directly copied from [3DDFA](https://github.com/cleardusk/3DDFA) with some modification. Thanks to their excellent work!

## requirements
### Directly installation
make sure the pytorch3d is 0.3.0 version
```
pip install -r requirements.txt
```
### (optional) Use Singularity
- For information and installation of Singularity, check []().
- After installing Singularity, run below line to build the singularity image
```
sudo singularity build pt_03.sif singularity.def
```
- run the code using 
```
SCRIPT="python xx.py"
singularity exec --nv -bind <> pt_03.sif ${SCRIPT}
```




# Reconstruction (using 3DDFA, )
- refer to [my3ddfa readme](my3ddfa/readme.md) and reconstruct the 3D face.
- refer to `my3ddfa/requirements.txt` for the environment

###  The output obj files are in the following form:
```
obj_path (MPII)
|
└─── p00
|     |      
|     └─── day01
|     |     |
|     |     └─── 0005_crop_params.txt
|     |     |    0005_lm.txt
|     |     |    0005.obj
|     |     └───   ... 
|     └─── day02
|     
└─── p01
```


# Render
## Config
After reconstruction, modify `./configs/config.yaml`
```
mpii:
    raw: ## directory to mpii
    obj: ## directory to the obj files from above reconstruction
```

## Exp1: rotate MPII to large head pose
- there are supplementary files for MPII (storing indices to filtered source image)
      - The supplementary files p*.h5 are be in `./supplementary/mpii/source_supply`

### How to rotate?
- `target_rotating.py` specifies the rotating patterns, and is set in `./configs/config.yaml`
      - now there are four patterns `['xgaze-train', 'eyediap-cs', 'eyediap-ft', 'gaussian']`, refer to `./configs/config.yaml`

```
SAVE_DIR=./output/mpii
rm -r $SAVE_DIR
mkdir -p $SAVE_DIR
singularity exec --nv --bind /media/jqin /home/jqin/wk/simgs/pt3d.simg \
python main_mpii_nv.py -save $SAVE_DIR 
python readh5.py --data_dir $SAVE_DIR/full/ 
```
### Output directories

The synthesized files will be like
```
$SAVE_DIR
|
└─── components
|     |      
|     └─── syn
|     |     └──── p00.h5
|     |     └────   ...
|     └─── dark
|     └─── cl    
|     └─── ...
|     
└─── full # final version
      └──── p00.h5
      └────  ...
└─── ab1 # ablation 1
      └────  ...
└─── ab2 # ablation 1
      └────  ...
```

### Structure of created `p00.h5`
Inside the h5 files, this person has1500 source images, N is the number of new head poses.
the key is like: 
```
subject0000.h5   
|     
└─── 'face_gaze': the gaze label [ 1500 * N, 2]
|      
└─── 'face_head_pose' : the head pose label [ 1500 * N , 1]
|     
└─── 'face_mask': the mask image [ 1500 * N , 224, 224]
|     
└─── 'face_mat_norm': the normalization matrix [ 1500 * N , 3, 3]
|     
└─── 'face_patch': the face image [ 1500 * N, 224, 224, 3]
|     
└─── 'rotation_matrix': the rotation matrix from source to THIS image (only inside each frame)  [ 1500 * N , 3, 3]
|     
└─── 'landmarks_norm': the normalized 2D landmarks location (only in the source image)   [1500, 68, 2]
```

### visualize the synthetic images
`readh5.py` will load the synthesized `p*.h5` files and output some images in the `$SAVE_DIR/full/samples` for visualization

---
---




```bash
conda install -c pytorch pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d=0.3.0 -c pytorch3d
```