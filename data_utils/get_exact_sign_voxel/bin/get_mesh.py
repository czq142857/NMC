import numpy as np
import cv2
import os
import h5py
import time
import mcubes
import argparse
import binvox_rw

def write_ply_triangle(name, vertices, triangles):
    vertices = vertices.astype(np.float32)
    triangles = triangles.astype(np.int32)
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
    fout.close()


#get voxel models
voxel_model_file = open('0.binvox', 'rb')
voxel_model_1024 = binvox_rw.read_as_3d_array(voxel_model_file, fix_coords=False).data.astype(np.uint8)
voxel_model_file.close()

print(voxel_model_1024.shape)

cv2.imwrite("i.png", voxel_model_1024[:,:,voxel_model_1024.shape[2]//2]*255)

vertices, triangles = mcubes.marching_cubes(voxel_model_1024, 0.5)
vertices = vertices/(voxel_model_1024.shape[0]-1)-0.5
write_ply_triangle("out.ply", vertices, triangles)

