## Face Alignment in Full Pose Range: A 3D Total Solution

This is the modified work from https://github.com/cleardusk/3DDFA

### Usage
Now I only wrote the code for MPII and XGaze.

You can run `python main_mpii.py` or `python main_xgaze.py`.

#### `recon_mpii.py`
```
cd my3ddfa
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

#### `recon_xgaze.py`
by setting the raw image path` xgaze_path` (raw xgaze base directory) and output `output_dir`, the code will read image by **subject** and **frame**, and output objs files `cam00.obj` with 3D landmarks `cam00_lm.txt`and the cropping transformation matrix `cam00_crop_params.txt` during cropping.

```
cd /my3ddfa
python recon_xgaze.py --xgaze_path '<>' \
        --output_dir '<>'
```

Each frame only contains the first camera cam00
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
