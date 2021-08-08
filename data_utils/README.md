# data_utils
Tools for obtaining SDF grids, voxels, and intersection points. These codes are modified from [SDFGen](https://github.com/christopherbatty/SDFGen).


## Requirements
- Ubuntu
- cmake


## Usage

To compile the code, go to each subfolder and run:
```
cmake .
make
```

You can find the executable in *bin*, along with a testing script *get_mesh.py*, and some sample results. Run the executable to get more details on how to use it, e.g., ```./SDFGen```. You can also read *main.cpp* or *get_mesh.py* to understand how the data are stored in *.sdf*, *.binvox*, *.intersection*, and *.intersectionpn* format.

The generated *.binvox* file by *VOXGen* is in the commonly used binvox format. However, it stores the signs of the signed distances, not the occupancies for each voxel. The signs and the occupancies are different! We convert the signs of a high-resolution grid into occupancies of a low-resolution grid in [data_preprocessing](https://github.com/czq142857/NMC/tree/master/data_preprocessing).
