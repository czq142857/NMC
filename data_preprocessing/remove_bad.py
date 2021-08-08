import numpy as np
#import cv2
import os
import h5py
from multiprocessing import Process, Queue, Lock
import queue
import time
#import binvox_rw
#import mcubes
import argparse

gt_dir = "gt_simplified"
if not os.path.exists("garbage")
    os.makedirs("garbage")

hdf5_names = os.listdir(gt_dir)
hdf5_names = sorted(hdf5_names)
for name in hdf5_names:
    hdf5_file = h5py.File(gt_dir+"/"+name, 'r')
    
    grid_size = 64

    a = hdf5_file[str(grid_size)+"_case_id"][:]
    a = a[:-1,:-1,:-1]

    valid_count = np.sum(a>0)
    invalid_count = np.sum(a<0)

    if invalid_count>valid_count*0.1 or valid_count<500:
        print(name,invalid_count,valid_count)
        #continue
        cmd = "mv "+gt_dir+"/"+name+" "+"garbage/"
        os.system(cmd)
    
    b = hdf5_file[str(grid_size)+"_int"][:,:,:,0]
    valid_flag = b>0

    #x
    ray = np.max(valid_flag,(1,2))
    xmin = -1
    xmax = -1
    for i in range(grid_size+1):
        if ray[i]>0:
            if xmin==-1:
                xmin = i
            xmax = i
    #y
    ray = np.max(valid_flag,(0,2))
    ymin = -1
    ymax = -1
    for i in range(grid_size+1):
        if ray[i]>0:
            if ymin==-1:
                ymin = i
            ymax = i
    #z
    ray = np.max(valid_flag,(0,1))
    zmin = -1
    zmax = -1
    for i in range(grid_size+1):
        if ray[i]>0:
            if zmin==-1:
                zmin = i
            zmax = i

    if ((xmax-xmin)<=4 and (ymax-ymin)<=4) or ((xmax-xmin)<=4 and (zmax-zmin)<=4) or ((zmax-zmin)<=4 and (ymax-ymin)<=4):
        print(name,xmin,xmax,ymin,ymax,zmin,zmax)
        #continue
        cmd = "mv "+gt_dir+"/"+name+" "+"garbage/"
        os.system(cmd)

