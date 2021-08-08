import numpy as np
import cv2
import os
import h5py
import time
import mcubes
import argparse

def write_ply_point_normal(name, vertices, normals=None):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("property float nx\n")
	fout.write("property float ny\n")
	fout.write("property float nz\n")
	fout.write("end_header\n")
	if normals is None:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
	else:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
	fout.close()


def read_intersectionpn_file_as_2d_array(name):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#intersectionpn'):
        raise IOError('Not an intersectionpn file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    point_nums = np.array(list(map(int, fp.readline().strip().split(b' '))),np.int32)
    line = fp.readline()
    data = np.frombuffer(fp.read(), dtype=np.float32)
    data = data.reshape([np.sum(point_nums),6])
    fp.close()
    separated = []
    count = 0
    for i in range(len(point_nums)):
        separated.append(np.ascontiguousarray(data[count:count+point_nums[i]]))
        count += point_nums[i]
    return separated


vertices_X, vertices_Y, vertices_Z = read_intersectionpn_file_as_2d_array("0.intersectionpn")
write_ply_point_normal("vertices_X.ply", vertices_X)
write_ply_point_normal("vertices_Y.ply", vertices_Y)
write_ply_point_normal("vertices_Z.ply", vertices_Z)

#select X==128
vertices = np.concatenate([vertices_Y[vertices_Y[:,0]==128],vertices_Z[vertices_Z[:,0]==128]], axis=0)
write_ply_point_normal("out.ply", vertices)
