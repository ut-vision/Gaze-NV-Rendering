# The pipeline of Reconstruction and Rendering

## requirements
make sure the pytorch3d is 0.3.0 version
```
pip install -r requirements.txt
```
(Sorry I haven't checked this requirements, if some package is missing, please install specifically)


## Download MPIIFaceGaze and XGaze 
- Download MPIIFaceGaze by `wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip` and uncompress.
- Download ETH-XGaze raw images


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


```
obj_path (XGaze, train)
|
└─── subject0000
|     |      
|     └─── frame0000
|     |     |
|     |     └─── cam00_crop_params.txt
|     |          cam00_lm.txt
|     |          cam00.obj
|     |     
|     └─── frame0001
|     
└─── subject0003
```

# Render
## Config
After reconstruction, modify `./configs/config.yaml`
```
mpii:
    raw: ## directory to mpii
    obj: ## directory to the obj files from above reconstruction
xgaze:
    raw:  ## directory to xgaze
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


## Exp2: rotate XGaze cam00.JPG to other 17 cameras 

```
SAVE_DIR=./output/xgaze_syn
rm -r $SAVE_DIR
mkdir -p $SAVE_DIR
python main_xgaze_syn.py -save $SAVE_DIR 
python readh5.py --data_dir $SAVE_DIR/full
```

### output directories

The synthesized files will be like
```
$SAVE_DIR
|
└─── components
|     |      
|     └─── syn
|     |     └──── subject0000.h5
|     |     └────   ...
|     └─── dark
|     └─── cl    
|     └─── ...
|     
└─── full # final version
      └──── subject0000.h5
      └────  ...
└─── ab1 # ablation 1
      └────  ...
└─── ab2 # ablation 1
      └────  ...
```
## sturcture of output `subject*.h5`
Inside the h5 files, suppose this subject has N * (m+1) images, where N is the number of frames, m is the number of new head poses.
the key is like: 
```
subject0000.h5   
|     
└─── 'frame_index': the frame index  [N,1]
|     
└─── 'cam_index': the frame index [N,1]
|
└─── 'face_gaze': the gaze label [ N * (m+1) ,2]
|      
└─── 'face_head_pose' : the head pose label [ N * (m+1) , 1]
|     
└─── 'face_mask': the mask image [ N * (m+1) , 224, 224]
|     
└─── 'face_mat_norm': the normalization matrix [ N * (m+1) , 3, 3]
|     
└─── 'face_patch': the face image [ N * (m+1) , 224, 224]
|     
└─── 'rotation_matrix': the rotation matrix from source to THIS image (only inside each frame)  [ N * (m+1) , 3, 3]
|     
└─── 'landmarks_norm': the normalized 2D landmarks location (only in the source image)   [N, 68, 2]

```

