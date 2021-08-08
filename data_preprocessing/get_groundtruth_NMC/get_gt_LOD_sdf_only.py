import numpy as np
#import cv2
import os
import h5py
from multiprocessing import Process, Queue, Lock
import queue
import time
import binvox_rw
#import mcubes
import argparse

import utils
import cutils


#note that the cells and voxels are not aligned
# = = = = = = = = =  voxel
# |-----| |-----|    voxel /4
# |-------|          mc cell 0
#         |-------|  mc cell 1


#this script outputs LOD of ground truth
# grid_size     voxel_size  intersection_size
# 128           1024        4096  <- too slow, not used
# 64            512         2048
# 32            256         1024

# each grid cell has 1 + 8+6+1 + (8+12+8*3)*3 values, with repetition
# each grid cell has 1 + 1+3+1 + 8*3+3*1+12*2 = 57 values, without repetition
# 1 cube case id     1/1
# 1 cube corner V    8/8
# 3 cube face V      6/2
# 1 cube internal V  1/1
# 8 cube internal P  8/1
# 3 cube edge P      12/4
# 12 cube face P     24/2

#CNN input M x N x P need padding, output M+1 x N+1 x P+1

#compute the following in order:
# --- corner ---
# cube corner V
# --- edge ---
# cube edge P
# --- face ---
# cube face V
# cube face P
# -- internal ---
# cube internal V
# cube case id
# cube internal P



lock = Lock()
def get_gt_from_voxel_and_intersection(q, name_list):
    name_num = len(name_list)

    cell_size = 8
    configs = np.load("LUT_CC.npz")["LUT_CC"].astype(np.int32) #32768 x 3, [case_id, #insideCC, #outsideCC]
    tessellations = np.load("LUT_tess.npz")["LUT_tess"].astype(np.int32) #32768 x max_num_of_triangles x 3

    num_of_int_params = 1 + 1+3+1
    num_of_float_params = 8*3+3*1+12*2

    grid_size_list = [32,64,128]
    LOD_gt_int = {}
    LOD_gt_float = {}
    LOD_input_sdf = {}
    LOD_input_voxel = {}
    for grid_size in grid_size_list:
        voxel_size = grid_size*cell_size
        grid_size_1 = grid_size+1
        LOD_gt_int[grid_size] = np.zeros([grid_size_1,grid_size_1,grid_size_1,num_of_int_params], np.int32)
        LOD_gt_float[grid_size] = np.zeros([grid_size_1,grid_size_1,grid_size_1,num_of_float_params], np.float32)
        LOD_input_sdf[grid_size] = np.ones([grid_size_1,grid_size_1,grid_size_1], np.float32)
        LOD_input_voxel[grid_size] = np.zeros([grid_size_1,grid_size_1,grid_size_1], np.uint8)


    for nid in range(name_num):
        pid = name_list[nid][0]
        idx = name_list[nid][1]
        in_name = name_list[nid][2]
        out_name = name_list[nid][3]

        in_obj_name = in_name + ".obj"
        in_sdf_name = in_name + ".sdf"
        in_intersection_name = in_name + ".intersection"
        in_binvox_name = in_name + ".binvox"
        out_hdf5_name = out_name + ".hdf5"

        #if nid+1<name_num    and os.path.exists(name_list[nid+1][3] + ".hdf5"): continue
        #if os.path.exists(out_hdf5_name): continue

        print(pid,'  ',nid,'/',name_num, idx, in_binvox_name)


        #run exe to get sdf, intersection, and binvox
        command = "./SDFGen "+in_obj_name+" 128 0"
        os.system(command)


        #read
        sdf_129 = utils.read_sdf_file_as_3d_array(in_sdf_name) #128

        #compute gt
        for grid_size in grid_size_list:
            voxel_size = grid_size*cell_size
            grid_size_1 = grid_size+1
            downscale = 1024//voxel_size

            start_time = time.time()

            i = 0
            j = 0
            k = 0

            tmp_sdf = sdf_129[i::downscale,j::downscale,k::downscale]
            LOD_input_sdf[grid_size][:tmp_sdf.shape[0],:tmp_sdf.shape[1],:tmp_sdf.shape[2]] = tmp_sdf


        #record data
        hdf5_file = h5py.File(out_hdf5_name, 'w')
        for grid_size in grid_size_list:
            voxel_size = grid_size*cell_size
            grid_size_1 = grid_size+1
            hdf5_file.create_dataset(str(grid_size)+"_sdf", [grid_size_1,grid_size_1,grid_size_1], np.float32, compression=9)
            hdf5_file[str(grid_size)+"_sdf"][:] = LOD_input_sdf[grid_size]
        hdf5_file.close()


        #delete sdf, intersection, and binvox to save space
        os.remove(in_sdf_name)


        q.put([1,pid,idx])




if __name__ == '__main__':

    target_dir = "../objs/"
    if not os.path.exists(target_dir):
        print("ERROR: this dir does not exist: "+target_dir)
        exit()

    write_dir = "./gt/"
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    #obj_names = os.listdir(target_dir)
    #obj_names = sorted(obj_names)

    #obj_names = ["00000016"]

    fin = open("abc_obj_list.txt", 'r')
    obj_names = [name.strip() for name in fin.readlines()]
    fin.close()

    obj_names_len = len(obj_names)


    #prepare list of names
    even_distribution = [16] #[16, 16, 6, 6, 16, 16]
    this_machine_id = 0
    num_of_process = 0
    P_start = 0
    P_end = 0
    for i in range(len(even_distribution)):
        num_of_process += even_distribution[i]
        if i<this_machine_id:
            P_start += even_distribution[i]
        if i<=this_machine_id:
            P_end += even_distribution[i]
    print(this_machine_id, P_start, P_end)

    list_of_list_of_names = []
    for i in range(num_of_process):
        list_of_list_of_names.append([])
    for idx in range(obj_names_len):
        process_id = idx%num_of_process
        in_name = target_dir + obj_names[idx] + "/model"
        out_name = write_dir + obj_names[idx]

        list_of_list_of_names[process_id].append([process_id, idx, in_name, out_name])

    
    #map processes
    q = Queue()
    workers = []
    for i in range(P_start,P_end):
        list_of_names = list_of_list_of_names[i]
        workers.append(Process(target=get_gt_from_voxel_and_intersection, args = (q, list_of_names)))

    for p in workers:
        p.start()


    counter = 0
    while True:
        item_flag = True
        try:
            success_flag,pid,idx = q.get(True, 1.0)
        except queue.Empty:
            item_flag = False
        
        if item_flag:
            #process result
            counter += success_flag

        allExited = True
        for p in workers:
            if p.exitcode is None:
                allExited = False
                break
        if allExited and q.empty():
            break


    print("finished")
    print("returned", counter,"/",obj_names_len)
    

    #q = Queue()
    #get_gt_from_voxel_and_intersection(q,list_of_list_of_names[0])


