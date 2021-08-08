# data_preprocessing
Tools for pre-processing raw mesh data and converting them to the representation used for training.

## Requirements
- Ubuntu
- Python 3 with numpy, h5py and Cython

Build Cython module in each subfolder:
```
python setup.py build_ext --inplace
```

## Usage

You can download [ABC](https://deep-geometry.github.io/abc-dataset/) and [Thingi10K](https://ten-thousand-models.appspot.com/) from their websites. Or you could prepare your own dataset. All the shapes need to be closed triangle meshes and in obj format (so ShapeNet is not an option...). In the following, I would assume you are using [ABC](https://deep-geometry.github.io/abc-dataset/) dataset.

Run *simplify_obj.py* to normalize the meshes and remove empty folders.

Go to either *get_groundtruth_NMC* or *get_groundtruth_NMC_lite*. Then run *get_gt_LOD.py* to get training data (very slow), or *get_gt_LOD_sdf_only.py* to only get SDF grid input data (much faster).

Since *get_gt_LOD.py* is so slow, I would recommend using multiple processes and possibly multiple machines. You can modify the code in line 234 and 235:
```
    even_distribution = [16]
    this_machine_id = 0
```
It means the script will run on one machine with 16 processes.

You can distribute the workload into three machines by modifying the code for each machine. Say you want to run 16 processes on machine A, 8 processes on machine B, and 6 processes on machine C, you can do the following.

On machine A:
```
    even_distribution = [16,8,6]
    this_machine_id = 0
```

On machine B:
```
    even_distribution = [16,8,6]
    this_machine_id = 1
```

On machine C:
```
    even_distribution = [16,8,6]
    this_machine_id = 2
```

## Removing invalid shapes

Some shapes in [ABC](https://deep-geometry.github.io/abc-dataset/) and [Thingi10K](https://ten-thousand-models.appspot.com/) are not suitable for marching cubes. They contain too many cubes that cannot be represented by the cube tessellation templates. Those shapes are removed via *remove_bad.py*, which also removes trivial shapes (thin rods).

The shapes used in our experiments are recorded in *abc_obj_list.txt* and *thingi10k_obj_list.txt*.

## Executable files IntersectionXYZ, SDFGen, and VOXGen

See [data_utils](https://github.com/czq142857/NMC/tree/master/data_utils).

## Look-Up Tables LUT.npz, LUT_CC.npz, and LUT_tess.npz

See [tessellation](https://github.com/czq142857/NMC/tree/master/tessellation).




