## Face Alignment in Full Pose Range: A 3D Total Solution

This is the modified work from https://github.com/cleardusk/3DDFA



## Download MPIIFaceGaze and XGaze 
- Download MPIIFaceGaze by `wget http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip` and uncompress.



### Usage
Now I only wrote the code for MPII and XGaze.

You can run `python main_mpii.py` or `python main_xgaze.py`.

#### `recon_mpii.py`
```
cd 3ddfa
python main_mpii.py --mpii_path '<>/MPIIFaceGaze' --output_dir <>
```
by setting the `mpii_path` (the raw image of MPIIFaceGaze) and `output_dir`, the code will read image by **person** and **day**, and output objs files `*.obj` with 3D landmarks `*_lm.txt`and the cropping transformation matrix `*_crop_params.txt` during cropping.


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



## run inference

pip install -r requirements.txt

###  Build cython module (just one line for building)

cd utils/cython
python3 setup.py build_ext -i

