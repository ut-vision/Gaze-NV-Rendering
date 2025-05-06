# Gaze-NV-Rendering <!-- omit in toc -->

Official implementation for **“Learning-by-Novel-View-Synthesis for Full-Face Appearance-based 3D Gaze Estimation”**  
([arXiv 2201.07927](https://arxiv.org/abs/2201.07927))

> This repository recreates the **MPII-NV** synthetic dataset via 3D face reconstruction (3DDFA) and photorealistic novel-view rendering with PyTorch3D.

:fire: Check [XGaze3D](https://github.com/ut-vision/XGaze3D.git) for the multi-view reconstruction and rendering of XGaze.

## 1. Installation
- PyTorch3D 0.3.0 are required (due to renderer API changes).
- The PyTorch3D is recommended to install by Conda.

```bash
conda create -n nv_render python==3.7
conda activate nv_render
pip install -r requirements.txt
conda install -c pytorch pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d=0.3.0 -c pytorch3d

## Build 3DDFA Cython ops
cd 3ddfa/utils/cython
python setup.py build_ext -i 
```



## 2. Dataset Preparation
#### MPIIFaceGaze

```bash
wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip
unzip MPIIFaceGaze.zip -d <TARGET_DIR>
```
#### Places365
Download the validation split (val_256) from the [official site](http://places2.csail.mit.edu/) .

#### Supplementary assets
Download the `supplementary/` folder from [Google Drive](https://drive.google.com/drive/folders/1oeS92mgjoysL1UXWFA104ptA_OYWJZ_Y?usp=sharing) and place it at the repo root:
```
Gaze-NV-rendering/
├── supplementary/
│   ├── mpii/                     # indices & render configs
│   ├── face_model.yml
│   ├── OpenFace.txt
│   ├── mmod_human_face_detector.dat
│   └── shape_predictor_68_face_landmarks.dat
└── ...
```



## 3.  Pipeline of Reconstruction and NV-Rendering
### Reconstruction (3DDFA with slight modification)

```bash
cd ./3ddfa
python recon_mpii.py \
      --mpii_path "<PATH_TO_MPIIFaceGaze>" \
      --output_dir "<PATH_TO_RECONSTRUCTED_OBJS>"

```
Output (per subject):
```yaml
"<PATH_TO_RECONSTRUCTED_OBJS>"
└─ p00/
   └─ day01/
      ├─ 0005.obj               # reconstructed mesh
      ├─ 0005_lm.txt            # 3D landmarks
      └─ 0005_crop_params.txt   # crop transformation matrix
```


###  Render
1. After reconstruction, modify `./config_path.yaml`:
```yaml
data:
  mpii:
    raw: "<PATH_TO_MPIIFaceGaze>"
    obj: "<PATH_TO_RECONSTRUCTED_OBJS>"
  background_path: "<PATH_TO_Places365/val_256>"
```

2. Run renderer & save outputs into HDF5 file:

```bash
SAVE_DIR=./output/mpii_nv
mkdir -p $SAVE_DIR
python main.py   -save $SAVE_DIR
python readh5.py --data_dir $SAVE_DIR/full
```





## 4. Results structure

The synthesized files will be like
```yaml
output/mpii_nv/
└─ full/
   ├─ p00.h5
   ├─ p01.h5
   └─ ...
```

#### Structure of created `p00.h5`
Inside the h5 files, this person has 1500 x N images, where 1500 is the number of source images and N is the number of new head poses.


| Key               | Shape                            | Description                                          |
|-------------------|----------------------------------|------------------------------------------------------|
| face_gaze         | (1500 × N, 2)                    | Gaze angles (pitch, yaw)                             |
| face_head_pose    | (1500 × N, 3)                    | Head pose (roll, pitch, yaw)                         |
| face_patch        | (1500 × N, 224 × 224 × 3)        | Rendered face                                        |
| face_mask         | (1500 × N, 224 × 224)            | Corresponding masks                                  |
| rotation_matrix   | (1500 × N, 3 × 3)                | Source → target rotation                             |
| face_mat_norm     | (1500 × N, 3 × 3)                | Camera normalization matrix                          |
| landmarks_norm    | (1500, 68, 2) *(source only)*    | 2D landmark positions in normalized space            |


#### Visualize the rendered images
```bash 
SAVE_DIR=./output/mpii_nv
python readh5.py --data_dir $SAVE_DIR
```

## Misc.
- To preserve reproducibility, we stored the source indices for MPIIFaceGaze: `supplementary/mpii/source_supply`
- We also store the Rendering configs (BG IDs, lighting, colors, etc.): `supplementary/mpii/*`
- Novel-view patterns are defined in `novel_view.py`: (xgaze-train, eyediap-cs, eyediap-ft, gaussian). Select a pattern via `config_path.yaml`.


## Acknowledgements
We acknowledge the excellent work of [3DDFA](https://github.com/cleardusk/3DDFA) for face reconstruction, and [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for rendering.

## Contact
If you have any questions, feel free to contact Jiawei Qin at jqin@iis.u-tokyo.ac.jp.