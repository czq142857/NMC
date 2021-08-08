import numpy as np
import h5py

tessellations = np.load("LUT_tess.npz")["LUT_tess"].astype(np.int32) #32768 x max_num_of_triangles x 3

#     vertices              edges                faces
#         7 ________ 6           _____6__             ________
#         /|       /|         7/|       /|          /|       /|
#       /  |     /  |        /  |     /5 |        /  | 1   /  |
#   4 /_______ /    |      /__4____ /    10     /_______ 5    |
#    |     |  |5    |     |    11  |     |     |  3  |  |     |
#    |    3|__|_____|2    |     |__|__2__|     |     |__|__2__|
#   k|    /   |    /      8   3/   9    /      |    4   |    /
#    |  / j   |  /        |  /     |  /1       |  /    0|  /
#    |/_______|/          |/___0___|/          |/___ ___|/
#   0    i     1
#


# ---------- define all points ----------
all_points_len = 0
all_points = []

cube_points = np.array([
[0,1,0], [1,1,0], [1,1,1], [0,1,1],
[0,0,0], [1,0,0], [1,0,1], [0,0,1],
],np.float32)
cube_points = cube_points-0.5
all_points.append(cube_points)
all_points_len += len(cube_points)


small_cube_points = cube_points*0.2 #this multiplier must be positive
all_points.append(small_cube_points)
all_points_len += len(small_cube_points)


edge_from_cube_points_idx = np.array([[0,1], [1,2], [2,3], [0,3], [4,5], [5,6], [6,7], [4,7], [0,4], [1,5], [2,6], [3,7]],np.int32)
edge_points = []
find_edge_point_from_cube_points = np.full([8,8],-1,np.int32)
for i in range(len(edge_from_cube_points_idx)):
    p1i = edge_from_cube_points_idx[i,0]
    p2i = edge_from_cube_points_idx[i,1]
    p1 = cube_points[p1i]
    p2 = cube_points[p2i]
    edge_points.append( (p1+p2)/2 )
    find_edge_point_from_cube_points[p1i,p2i] = i
    find_edge_point_from_cube_points[p2i,p1i] = i
find_edge_point_from_cube_points[find_edge_point_from_cube_points>=0] += all_points_len
edge_points = np.array(edge_points,np.float32)
all_points.append(edge_points)
all_points_len += len(edge_points)


face_from_cube_points_idx = np.array([[0,1,2,3], [4,5,6,7], [1,2,6,5], [0,3,7,4], [0,1,5,4], [3,2,6,7]],np.int32)
face_points = np.zeros([3,8,3],np.float32)
find_face_point_from_cube_point_and_face = np.full([8,6],-1,np.int32)
face_points[0] = small_cube_points #split dim 0
face_points[1] = small_cube_points #split dim 1
face_points[2] = small_cube_points #split dim 2
#face 0 -> dim_1 = 0.5
#face 1 -> dim_1 = -0.5
#face 2 -> dim_0 = 0.5
#face 3 -> dim_0 = -0.5
#face 4 -> dim_2 = -0.5
#face 5 -> dim_2 = 0.5
for i in range(8):
    if face_points[0,i,0]>0:
        face_points[0,i,0]=0.5
        find_face_point_from_cube_point_and_face[i,2] = 0*8+i
    if face_points[0,i,0]<0:
        face_points[0,i,0]=-0.5
        find_face_point_from_cube_point_and_face[i,3] = 0*8+i
    if face_points[1,i,1]>0:
        face_points[1,i,1]=0.5
        find_face_point_from_cube_point_and_face[i,0] = 1*8+i
    if face_points[1,i,1]<0:
        face_points[1,i,1]=-0.5
        find_face_point_from_cube_point_and_face[i,1] = 1*8+i
    if face_points[2,i,2]>0:
        face_points[2,i,2]=0.5
        find_face_point_from_cube_point_and_face[i,5] = 2*8+i
    if face_points[2,i,2]<0:
        face_points[2,i,2]=-0.5
        find_face_point_from_cube_point_and_face[i,4] = 2*8+i
face_points[np.abs(face_points)<0.5] = 0
find_face_point_from_cube_point_and_face[find_face_point_from_cube_point_and_face>=0] += all_points_len
face_points = np.reshape(face_points,[-1,3])
all_points.append(face_points)
all_points_len += len(face_points)

all_points = np.concatenate(all_points,axis=0)

all_points_ = np.copy(all_points)
all_points[:,1] = all_points_[:,2]
all_points[:,2] = -all_points_[:,1]

all_points = all_points+0.5


# ---------- define partial points ----------
#all_points: (8+8+12+24)*3
#partial_points: 8*3+3*1+12*2 = 51
# center_points_vi_x/y/z (8*3),
# edge0_x (1), edge3_y (1), edge8_z (1),
# face_points_f0_x/y (4*2), face_points_f3_y/z (4*2), face_points_f4_x/z (4*2)

partial_points = np.zeros([51],np.float32)
for i in range(8): #center points
    partial_points[i*3:i*3+3] = all_points[i+8,:]
edge_list = [0,3,8]
for i in range(3): #edge points
    v0,v1 = edge_from_cube_points_idx[edge_list[i]]
    v2 = find_edge_point_from_cube_points[v0,v1]
    partial_points[8*3+i] = all_points[v2,i]
face_list = [0,3,4]
for i in range(3): #face points
    for j in range(4):
        f0 = face_list[i]
        v0 = face_from_cube_points_idx[f0,j]
        v2 = find_face_point_from_cube_point_and_face[v0,f0]
        if i==0:
            partial_points[8*3+3+(i*4+j)*2+0] = all_points[v2,0]
            partial_points[8*3+3+(i*4+j)*2+1] = all_points[v2,1]
        elif i==1:
            partial_points[8*3+3+(i*4+j)*2+0] = all_points[v2,1]
            partial_points[8*3+3+(i*4+j)*2+1] = all_points[v2,2]
        elif i==2:
            partial_points[8*3+3+(i*4+j)*2+0] = all_points[v2,0]
            partial_points[8*3+3+(i*4+j)*2+1] = all_points[v2,2]



def read_sdf_file_as_3d_array(name):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#sdf'):
        raise IOError('Not a sdf file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    line = fp.readline()
    data = np.frombuffer(fp.read(), dtype=np.float32)
    data = data.reshape(dims)
    fp.close()
    return data


def read_data_input_only(hdf5_dir,grid_size,input_type,out_bool,out_float):
    hdf5_file = h5py.File(hdf5_dir, 'r')
    LOD_gt_case_id = np.zeros([grid_size+1,grid_size+1,grid_size+1],np.int32)
    if out_bool:
        LOD_gt_int = np.zeros([grid_size+1,grid_size+1,grid_size+1,5],np.int32)
    else:
        LOD_gt_int = None
    if out_float:
        LOD_gt_float = np.zeros([grid_size+1,grid_size+1,grid_size+1,51],np.float32)
    else:
        LOD_gt_float = None
    if input_type=="sdf":
        LOD_input = hdf5_file[str(grid_size)+"_sdf"][:]
        LOD_input = LOD_input*grid_size #denormalize
    elif input_type=="voxel":
        LOD_input = hdf5_file[str(grid_size)+"_voxel"][:]
    hdf5_file.close()
    return LOD_gt_case_id, LOD_gt_int, LOD_gt_float, LOD_input

def read_data_bool_only(hdf5_dir,grid_size,input_type,out_bool,out_float):
    hdf5_file = h5py.File(hdf5_dir, 'r')
    LOD_gt_case_id = hdf5_file[str(grid_size)+"_case_id"][:]
    if out_bool:
        LOD_gt_int = hdf5_file[str(grid_size)+"_int"][:]
    else:
        LOD_gt_int = None
    if out_float:
        LOD_gt_float = np.zeros([grid_size+1,grid_size+1,grid_size+1,51],np.float32)
    else:
        LOD_gt_float = None
    if input_type=="sdf":
        LOD_input = hdf5_file[str(grid_size)+"_sdf"][:]
        LOD_input = LOD_input*grid_size #denormalize
    elif input_type=="voxel":
        LOD_input = hdf5_file[str(grid_size)+"_voxel"][:]
    hdf5_file.close()
    return LOD_gt_case_id, LOD_gt_int, LOD_gt_float, LOD_input

def read_data(hdf5_dir,grid_size,input_type,out_bool,out_float):
    hdf5_file = h5py.File(hdf5_dir, 'r')
    LOD_gt_case_id = hdf5_file[str(grid_size)+"_case_id"][:]
    if out_bool:
        LOD_gt_int = hdf5_file[str(grid_size)+"_int"][:]
    else:
        LOD_gt_int = None
    if out_float:
        LOD_gt_float = hdf5_file[str(grid_size)+"_float"][:]
    else:
        LOD_gt_float = None
    if input_type=="sdf":
        LOD_input = hdf5_file[str(grid_size)+"_sdf"][:]
        LOD_input = LOD_input*grid_size #denormalize
    elif input_type=="voxel":
        LOD_input = hdf5_file[str(grid_size)+"_voxel"][:]
    hdf5_file.close()
    return LOD_gt_case_id, LOD_gt_int, LOD_gt_float, LOD_input

def read_and_augment_data(hdf5_dir,grid_size,input_type,out_bool,out_float,aug_permutation=True,aug_reversal=True,aug_inversion=True):
    grid_size_1 = grid_size+1

    #read input hdf5
    LOD_gt_case_id, LOD_gt_int, LOD_gt_float, LOD_input = read_data(hdf5_dir,grid_size,input_type,out_bool,out_float)

    newdict = {}

    newdict['int_case_id'] = LOD_gt_case_id[:-1,:-1,:-1]

    if out_bool:
        newdict['int_V_signs'] = LOD_gt_int[:,:,:,0]
        newdict['int_F_i_j_signs'] = LOD_gt_int[:-1,:-1,:,1]
        newdict['int_F_j_k_signs'] = LOD_gt_int[:,:-1,:-1,2]
        newdict['int_F_i_k_signs'] = LOD_gt_int[:-1,:,:-1,3]
        newdict['int_tunnel_signs'] = LOD_gt_int[:-1,:-1,:-1,4]

    if out_float:
        newdict['float_center_i0_j0_k0_x_'] = LOD_gt_float[:-1,:-1,:-1,0]
        newdict['float_center_i0_j0_k0_y_'] = LOD_gt_float[:-1,:-1,:-1,1]
        newdict['float_center_i0_j0_k0_z_'] = LOD_gt_float[:-1,:-1,:-1,2]
        newdict['float_center_i1_j0_k0_x_'] = LOD_gt_float[:-1,:-1,:-1,3]
        newdict['float_center_i1_j0_k0_y_'] = LOD_gt_float[:-1,:-1,:-1,4]
        newdict['float_center_i1_j0_k0_z_'] = LOD_gt_float[:-1,:-1,:-1,5]
        newdict['float_center_i1_j1_k0_x_'] = LOD_gt_float[:-1,:-1,:-1,6]
        newdict['float_center_i1_j1_k0_y_'] = LOD_gt_float[:-1,:-1,:-1,7]
        newdict['float_center_i1_j1_k0_z_'] = LOD_gt_float[:-1,:-1,:-1,8]
        newdict['float_center_i0_j1_k0_x_'] = LOD_gt_float[:-1,:-1,:-1,9]
        newdict['float_center_i0_j1_k0_y_'] = LOD_gt_float[:-1,:-1,:-1,10]
        newdict['float_center_i0_j1_k0_z_'] = LOD_gt_float[:-1,:-1,:-1,11]
        newdict['float_center_i0_j0_k1_x_'] = LOD_gt_float[:-1,:-1,:-1,12]
        newdict['float_center_i0_j0_k1_y_'] = LOD_gt_float[:-1,:-1,:-1,13]
        newdict['float_center_i0_j0_k1_z_'] = LOD_gt_float[:-1,:-1,:-1,14]
        newdict['float_center_i1_j0_k1_x_'] = LOD_gt_float[:-1,:-1,:-1,15]
        newdict['float_center_i1_j0_k1_y_'] = LOD_gt_float[:-1,:-1,:-1,16]
        newdict['float_center_i1_j0_k1_z_'] = LOD_gt_float[:-1,:-1,:-1,17]
        newdict['float_center_i1_j1_k1_x_'] = LOD_gt_float[:-1,:-1,:-1,18]
        newdict['float_center_i1_j1_k1_y_'] = LOD_gt_float[:-1,:-1,:-1,19]
        newdict['float_center_i1_j1_k1_z_'] = LOD_gt_float[:-1,:-1,:-1,20]
        newdict['float_center_i0_j1_k1_x_'] = LOD_gt_float[:-1,:-1,:-1,21]
        newdict['float_center_i0_j1_k1_y_'] = LOD_gt_float[:-1,:-1,:-1,22]
        newdict['float_center_i0_j1_k1_z_'] = LOD_gt_float[:-1,:-1,:-1,23]

        newdict['float_edge_i_x_'] = LOD_gt_float[:-1,:,:,24]
        newdict['float_edge_j_y_'] = LOD_gt_float[:,:-1,:,25]
        newdict['float_edge_k_z_'] = LOD_gt_float[:,:,:-1,26]

        newdict['float_F_i_j_i0_j0_k0_x_'] = LOD_gt_float[:-1,:-1,:,27]
        newdict['float_F_i_j_i0_j0_k0_y_'] = LOD_gt_float[:-1,:-1,:,28]
        newdict['float_F_i_j_i1_j0_k0_x_'] = LOD_gt_float[:-1,:-1,:,29]
        newdict['float_F_i_j_i1_j0_k0_y_'] = LOD_gt_float[:-1,:-1,:,30]
        newdict['float_F_i_j_i1_j1_k0_x_'] = LOD_gt_float[:-1,:-1,:,31]
        newdict['float_F_i_j_i1_j1_k0_y_'] = LOD_gt_float[:-1,:-1,:,32]
        newdict['float_F_i_j_i0_j1_k0_x_'] = LOD_gt_float[:-1,:-1,:,33]
        newdict['float_F_i_j_i0_j1_k0_y_'] = LOD_gt_float[:-1,:-1,:,34]

        newdict['float_F_j_k_i0_j0_k0_y_'] = LOD_gt_float[:,:-1,:-1,35]
        newdict['float_F_j_k_i0_j0_k0_z_'] = LOD_gt_float[:,:-1,:-1,36]
        newdict['float_F_j_k_i0_j1_k0_y_'] = LOD_gt_float[:,:-1,:-1,37]
        newdict['float_F_j_k_i0_j1_k0_z_'] = LOD_gt_float[:,:-1,:-1,38]
        newdict['float_F_j_k_i0_j1_k1_y_'] = LOD_gt_float[:,:-1,:-1,39]
        newdict['float_F_j_k_i0_j1_k1_z_'] = LOD_gt_float[:,:-1,:-1,40]
        newdict['float_F_j_k_i0_j0_k1_y_'] = LOD_gt_float[:,:-1,:-1,41]
        newdict['float_F_j_k_i0_j0_k1_z_'] = LOD_gt_float[:,:-1,:-1,42]

        newdict['float_F_i_k_i0_j0_k0_x_'] = LOD_gt_float[:-1,:,:-1,43]
        newdict['float_F_i_k_i0_j0_k0_z_'] = LOD_gt_float[:-1,:,:-1,44]
        newdict['float_F_i_k_i1_j0_k0_x_'] = LOD_gt_float[:-1,:,:-1,45]
        newdict['float_F_i_k_i1_j0_k0_z_'] = LOD_gt_float[:-1,:,:-1,46]
        newdict['float_F_i_k_i1_j0_k1_x_'] = LOD_gt_float[:-1,:,:-1,47]
        newdict['float_F_i_k_i1_j0_k1_z_'] = LOD_gt_float[:-1,:,:-1,48]
        newdict['float_F_i_k_i0_j0_k1_x_'] = LOD_gt_float[:-1,:,:-1,49]
        newdict['float_F_i_k_i0_j0_k1_z_'] = LOD_gt_float[:-1,:,:-1,50]

    if input_type=="sdf":
        newdict['input_sdf'] = LOD_input[:,:,:]
    elif input_type=="voxel":
        newdict['input_voxel'] = LOD_input[:-1,:-1,:-1]

    #augment data
    permutation_list = [ [0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0] ]
    reversal_list = [ [0,0,0],[0,0,1],[0,1,0],[0,1,1], [1,0,0],[1,0,1],[1,1,0],[1,1,1] ]
    if aug_permutation:
        permutation = permutation_list[np.random.randint(len(permutation_list))]
    else:
        permutation = permutation_list[0]
    if aug_reversal:
        reversal = reversal_list[np.random.randint(len(reversal_list))]
    else:
        reversal = reversal_list[0]
    if aug_inversion:
        inversion_flag = np.random.randint(2)
    else:
        inversion_flag = 0

    if reversal[0]:
        for k in newdict: #inverse
            newdict[k] = newdict[k][::-1,:,:]
            if '_x_' in k:
                mask = (newdict[k]>=0)
                newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask
        for k in newdict: #switch pos
            if '_i0_' in k:
                k2 = k.replace('_i0_','_i1_')
                if k2 in newdict:
                    n1 = newdict[k]
                    n2 = newdict[k2]
                    newdict[k] = n2
                    newdict[k2] = n1

    if reversal[1]:
        for k in newdict: #inverse
            newdict[k] = newdict[k][:,::-1,:]
            if '_y_' in k:
                mask = (newdict[k]>=0)
                newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask
        for k in newdict: #switch pos
            if '_j0_' in k:
                k2 = k.replace('_j0_','_j1_')
                if k2 in newdict:
                    n1 = newdict[k]
                    n2 = newdict[k2]
                    newdict[k] = n2
                    newdict[k2] = n1

    if reversal[2]:
        for k in newdict: #inverse
            newdict[k] = newdict[k][:,:,::-1]
            if '_z_' in k:
                mask = (newdict[k]>=0)
                newdict[k] = newdict[k]*(1-mask)+(1-newdict[k])*mask
        for k in newdict: #switch pos
            if '_k0_' in k:
                k2 = k.replace('_k0_','_k1_')
                if k2 in newdict:
                    n1 = newdict[k]
                    n2 = newdict[k2]
                    newdict[k] = n2
                    newdict[k2] = n1

    if permutation == [0,1,2]:
        pass
    else:
        for k in newdict: #transpose
            newdict[k] = np.transpose(newdict[k], permutation)
        olddict = newdict
        newdict = {}
        for k in olddict:
            newdict[k] = olddict[k]

        if permutation == [0,2,1]:
            #j<->k
            if out_bool:
                newdict['int_F_i_j_signs'] = olddict['int_F_i_k_signs']
                newdict['int_F_i_k_signs'] = olddict['int_F_i_j_signs']
            if out_float:
                newdict['float_edge_j_y_'] = olddict['float_edge_k_z_']
                newdict['float_edge_k_z_'] = olddict['float_edge_j_y_']
                for k in newdict:
                    if 'float_F_' in k or 'float_center_' in k:
                        k2 = k
                        if '_i_j_' in k:
                            k2 = k2.replace('_i_j_','_i_k_')
                        elif '_i_k_' in k:
                            k2 = k2.replace('_i_k_','_i_j_')
                        if '_j1_' in k and '_k0_' in k:
                            k2 = k2.replace('_j1_','_j0_')
                            k2 = k2.replace('_k0_','_k1_')
                        elif '_j0_' in k and '_k1_' in k:
                            k2 = k2.replace('_j0_','_j1_')
                            k2 = k2.replace('_k1_','_k0_')
                        if '_y_' in k:
                            k2 = k2.replace('_y_','_z_')
                        elif '_z_' in k:
                            k2 = k2.replace('_z_','_y_')
                        newdict[k] = olddict[k2]

        elif permutation == [1,0,2]:
            #i<->j
            if out_bool:
                newdict['int_F_j_k_signs'] = olddict['int_F_i_k_signs']
                newdict['int_F_i_k_signs'] = olddict['int_F_j_k_signs']
            if out_float:
                newdict['float_edge_i_x_'] = olddict['float_edge_j_y_']
                newdict['float_edge_j_y_'] = olddict['float_edge_i_x_']
                for k in newdict:
                    if 'float_F_' in k or 'float_center_' in k:
                        k2 = k
                        if '_j_k_' in k:
                            k2 = k2.replace('_j_k_','_i_k_')
                        elif '_i_k_' in k:
                            k2 = k2.replace('_i_k_','_j_k_')
                        if '_i1_' in k and '_j0_' in k:
                            k2 = k2.replace('_i1_','_i0_')
                            k2 = k2.replace('_j0_','_j1_')
                        elif '_i0_' in k and '_j1_' in k:
                            k2 = k2.replace('_i0_','_i1_')
                            k2 = k2.replace('_j1_','_j0_')
                        if '_x_' in k:
                            k2 = k2.replace('_x_','_y_')
                        elif '_y_' in k:
                            k2 = k2.replace('_y_','_x_')
                        newdict[k] = olddict[k2]

        elif permutation == [2,1,0] or  permutation == [1,2,0] or permutation == [2,0,1]:
            #i<->k
            if out_bool:
                newdict['int_F_i_j_signs'] = olddict['int_F_j_k_signs']
                newdict['int_F_j_k_signs'] = olddict['int_F_i_j_signs']
            if out_float:
                newdict['float_edge_i_x_'] = olddict['float_edge_k_z_']
                newdict['float_edge_k_z_'] = olddict['float_edge_i_x_']
                for k in newdict:
                    if 'float_F_' in k or 'float_center_' in k:
                        k2 = k
                        if '_j_k_' in k:
                            k2 = k2.replace('_j_k_','_i_j_')
                        elif '_i_j_' in k:
                            k2 = k2.replace('_i_j_','_j_k_')
                        if '_i1_' in k and '_k0_' in k:
                            k2 = k2.replace('_i1_','_i0_')
                            k2 = k2.replace('_k0_','_k1_')
                        elif '_i0_' in k and '_k1_' in k:
                            k2 = k2.replace('_i0_','_i1_')
                            k2 = k2.replace('_k1_','_k0_')
                        if '_x_' in k:
                            k2 = k2.replace('_x_','_z_')
                        elif '_z_' in k:
                            k2 = k2.replace('_z_','_x_')
                        newdict[k] = olddict[k2]

            if permutation == [1,2,0]:
                olddict = newdict
                newdict = {}
                for k in olddict:
                    newdict[k] = olddict[k]
                #i<->j
                if out_bool:
                    newdict['int_F_j_k_signs'] = olddict['int_F_i_k_signs']
                    newdict['int_F_i_k_signs'] = olddict['int_F_j_k_signs']
                if out_float:
                    newdict['float_edge_i_x_'] = olddict['float_edge_j_y_']
                    newdict['float_edge_j_y_'] = olddict['float_edge_i_x_']
                    for k in newdict:
                        if 'float_F_' in k or 'float_center_' in k:
                            k2 = k
                            if '_j_k_' in k:
                                k2 = k2.replace('_j_k_','_i_k_')
                            elif '_i_k_' in k:
                                k2 = k2.replace('_i_k_','_j_k_')
                            if '_i1_' in k and '_j0_' in k:
                                k2 = k2.replace('_i1_','_i0_')
                                k2 = k2.replace('_j0_','_j1_')
                            elif '_i0_' in k and '_j1_' in k:
                                k2 = k2.replace('_i0_','_i1_')
                                k2 = k2.replace('_j1_','_j0_')
                            if '_x_' in k:
                                k2 = k2.replace('_x_','_y_')
                            elif '_y_' in k:
                                k2 = k2.replace('_y_','_x_')
                            newdict[k] = olddict[k2]

            elif permutation == [2,0,1]:
                olddict = newdict
                newdict = {}
                for k in olddict:
                    newdict[k] = olddict[k]
                #j<->k
                if out_bool:
                    newdict['int_F_i_j_signs'] = olddict['int_F_i_k_signs']
                    newdict['int_F_i_k_signs'] = olddict['int_F_i_j_signs']
                if out_float:
                    newdict['float_edge_j_y_'] = olddict['float_edge_k_z_']
                    newdict['float_edge_k_z_'] = olddict['float_edge_j_y_']
                    for k in newdict:
                        if 'float_F_' in k or 'float_center_' in k:
                            k2 = k
                            if '_i_j_' in k:
                                k2 = k2.replace('_i_j_','_i_k_')
                            elif '_i_k_' in k:
                                k2 = k2.replace('_i_k_','_i_j_')
                            if '_j1_' in k and '_k0_' in k:
                                k2 = k2.replace('_j1_','_j0_')
                                k2 = k2.replace('_k0_','_k1_')
                            elif '_j0_' in k and '_k1_' in k:
                                k2 = k2.replace('_j0_','_j1_')
                                k2 = k2.replace('_k1_','_k0_')
                            if '_y_' in k:
                                k2 = k2.replace('_y_','_z_')
                            elif '_z_' in k:
                                k2 = k2.replace('_z_','_y_')
                            newdict[k] = olddict[k2]

    #store outputs
    LOD_gt_case_id = np.full([grid_size_1,grid_size_1,grid_size_1], -1, np.int32)
    LOD_gt_case_id[:-1,:-1,:-1] = newdict['int_case_id']

    if out_bool:
        LOD_gt_int = np.full([grid_size_1,grid_size_1,grid_size_1,5], -1, np.int32)

        LOD_gt_int[:,:,:,0] = newdict['int_V_signs']
        LOD_gt_int[:-1,:-1,:,1] = newdict['int_F_i_j_signs']
        LOD_gt_int[:,:-1,:-1,2] = newdict['int_F_j_k_signs']
        LOD_gt_int[:-1,:,:-1,3] = newdict['int_F_i_k_signs']
        LOD_gt_int[:-1,:-1,:-1,4] = newdict['int_tunnel_signs']

        if inversion_flag:
            mask = (LOD_gt_int[:,:,:,0:4]>=0)
            LOD_gt_int[:,:,:,0:4] = LOD_gt_int[:,:,:,0:4]*(1-mask) + (1-LOD_gt_int[:,:,:,0:4])*mask
    else:
        LOD_gt_int = None

    if out_float:
        LOD_gt_float = np.full([grid_size_1,grid_size_1,grid_size_1,51], -1, np.float32)

        LOD_gt_float[:-1,:-1,:-1,0] = newdict['float_center_i0_j0_k0_x_']
        LOD_gt_float[:-1,:-1,:-1,1] = newdict['float_center_i0_j0_k0_y_']
        LOD_gt_float[:-1,:-1,:-1,2] = newdict['float_center_i0_j0_k0_z_']
        LOD_gt_float[:-1,:-1,:-1,3] = newdict['float_center_i1_j0_k0_x_']
        LOD_gt_float[:-1,:-1,:-1,4] = newdict['float_center_i1_j0_k0_y_']
        LOD_gt_float[:-1,:-1,:-1,5] = newdict['float_center_i1_j0_k0_z_']
        LOD_gt_float[:-1,:-1,:-1,6] = newdict['float_center_i1_j1_k0_x_']
        LOD_gt_float[:-1,:-1,:-1,7] = newdict['float_center_i1_j1_k0_y_']
        LOD_gt_float[:-1,:-1,:-1,8] = newdict['float_center_i1_j1_k0_z_']
        LOD_gt_float[:-1,:-1,:-1,9] = newdict['float_center_i0_j1_k0_x_']
        LOD_gt_float[:-1,:-1,:-1,10] = newdict['float_center_i0_j1_k0_y_']
        LOD_gt_float[:-1,:-1,:-1,11] = newdict['float_center_i0_j1_k0_z_']
        LOD_gt_float[:-1,:-1,:-1,12] = newdict['float_center_i0_j0_k1_x_']
        LOD_gt_float[:-1,:-1,:-1,13] = newdict['float_center_i0_j0_k1_y_']
        LOD_gt_float[:-1,:-1,:-1,14] = newdict['float_center_i0_j0_k1_z_']
        LOD_gt_float[:-1,:-1,:-1,15] = newdict['float_center_i1_j0_k1_x_']
        LOD_gt_float[:-1,:-1,:-1,16] = newdict['float_center_i1_j0_k1_y_']
        LOD_gt_float[:-1,:-1,:-1,17] = newdict['float_center_i1_j0_k1_z_']
        LOD_gt_float[:-1,:-1,:-1,18] = newdict['float_center_i1_j1_k1_x_']
        LOD_gt_float[:-1,:-1,:-1,19] = newdict['float_center_i1_j1_k1_y_']
        LOD_gt_float[:-1,:-1,:-1,20] = newdict['float_center_i1_j1_k1_z_']
        LOD_gt_float[:-1,:-1,:-1,21] = newdict['float_center_i0_j1_k1_x_']
        LOD_gt_float[:-1,:-1,:-1,22] = newdict['float_center_i0_j1_k1_y_']
        LOD_gt_float[:-1,:-1,:-1,23] = newdict['float_center_i0_j1_k1_z_']

        LOD_gt_float[:-1,:,:,24] = newdict['float_edge_i_x_']
        LOD_gt_float[:,:-1,:,25] = newdict['float_edge_j_y_']
        LOD_gt_float[:,:,:-1,26] = newdict['float_edge_k_z_']

        LOD_gt_float[:-1,:-1,:,27] = newdict['float_F_i_j_i0_j0_k0_x_']
        LOD_gt_float[:-1,:-1,:,28] = newdict['float_F_i_j_i0_j0_k0_y_']
        LOD_gt_float[:-1,:-1,:,29] = newdict['float_F_i_j_i1_j0_k0_x_']
        LOD_gt_float[:-1,:-1,:,30] = newdict['float_F_i_j_i1_j0_k0_y_']
        LOD_gt_float[:-1,:-1,:,31] = newdict['float_F_i_j_i1_j1_k0_x_']
        LOD_gt_float[:-1,:-1,:,32] = newdict['float_F_i_j_i1_j1_k0_y_']
        LOD_gt_float[:-1,:-1,:,33] = newdict['float_F_i_j_i0_j1_k0_x_']
        LOD_gt_float[:-1,:-1,:,34] = newdict['float_F_i_j_i0_j1_k0_y_']

        LOD_gt_float[:,:-1,:-1,35] = newdict['float_F_j_k_i0_j0_k0_y_']
        LOD_gt_float[:,:-1,:-1,36] = newdict['float_F_j_k_i0_j0_k0_z_']
        LOD_gt_float[:,:-1,:-1,37] = newdict['float_F_j_k_i0_j1_k0_y_']
        LOD_gt_float[:,:-1,:-1,38] = newdict['float_F_j_k_i0_j1_k0_z_']
        LOD_gt_float[:,:-1,:-1,39] = newdict['float_F_j_k_i0_j1_k1_y_']
        LOD_gt_float[:,:-1,:-1,40] = newdict['float_F_j_k_i0_j1_k1_z_']
        LOD_gt_float[:,:-1,:-1,41] = newdict['float_F_j_k_i0_j0_k1_y_']
        LOD_gt_float[:,:-1,:-1,42] = newdict['float_F_j_k_i0_j0_k1_z_']

        LOD_gt_float[:-1,:,:-1,43] = newdict['float_F_i_k_i0_j0_k0_x_']
        LOD_gt_float[:-1,:,:-1,44] = newdict['float_F_i_k_i0_j0_k0_z_']
        LOD_gt_float[:-1,:,:-1,45] = newdict['float_F_i_k_i1_j0_k0_x_']
        LOD_gt_float[:-1,:,:-1,46] = newdict['float_F_i_k_i1_j0_k0_z_']
        LOD_gt_float[:-1,:,:-1,47] = newdict['float_F_i_k_i1_j0_k1_x_']
        LOD_gt_float[:-1,:,:-1,48] = newdict['float_F_i_k_i1_j0_k1_z_']
        LOD_gt_float[:-1,:,:-1,49] = newdict['float_F_i_k_i0_j0_k1_x_']
        LOD_gt_float[:-1,:,:-1,50] = newdict['float_F_i_k_i0_j0_k1_z_']
    else:
        LOD_gt_float = None

    if input_type=="sdf":
        LOD_input = np.ones([grid_size_1,grid_size_1,grid_size_1], np.float32)
        LOD_input[:,:,:] = newdict['input_sdf']
        if inversion_flag:
            LOD_input = -LOD_input

    elif input_type=="voxel":
        LOD_input = np.zeros([grid_size_1,grid_size_1,grid_size_1], np.uint8)
        LOD_input[:-1,:-1,:-1] = newdict['input_voxel']
        if inversion_flag:
            LOD_input = 1-LOD_input

    return LOD_gt_case_id, LOD_gt_int, LOD_gt_float, LOD_input

def correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist):
    if config[fvlist[0]]==config[fvlist[1]] and config[fvlist[2]]==config[fvlist[3]] and config[fvlist[0]]!=config[fvlist[2]]:
        tx = (vertices[vvlist[0],0]+vertices[vvlist[3],0])/2
        ty = (vertices[vvlist[0],1]+vertices[vvlist[3],1])/2
        tz = (vertices[vvlist[0],2]+vertices[vvlist[3],2])/2
        vertices[vvlist[0],0] = tx
        vertices[vvlist[0],1] = ty
        vertices[vvlist[0],2] = tz
        vertices[vvlist[3],0] = tx
        vertices[vvlist[3],1] = ty
        vertices[vvlist[3],2] = tz
        tx = (vertices[vvlist[1],0]+vertices[vvlist[2],0])/2
        ty = (vertices[vvlist[1],1]+vertices[vvlist[2],1])/2
        tz = (vertices[vvlist[1],2]+vertices[vvlist[2],2])/2
        vertices[vvlist[1],0] = tx
        vertices[vvlist[1],1] = ty
        vertices[vvlist[1],2] = tz
        vertices[vvlist[2],0] = tx
        vertices[vvlist[2],1] = ty
        vertices[vvlist[2],2] = tz
    if config[fvlist[0]]==config[fvlist[3]] and config[fvlist[1]]==config[fvlist[2]] and config[fvlist[0]]!=config[fvlist[1]]:
        tx = (vertices[vvlist[0],0]+vertices[vvlist[1],0])/2
        ty = (vertices[vvlist[0],1]+vertices[vvlist[1],1])/2
        tz = (vertices[vvlist[0],2]+vertices[vvlist[1],2])/2
        vertices[vvlist[0],0] = tx
        vertices[vvlist[0],1] = ty
        vertices[vvlist[0],2] = tz
        vertices[vvlist[1],0] = tx
        vertices[vvlist[1],1] = ty
        vertices[vvlist[1],2] = tz
        tx = (vertices[vvlist[2],0]+vertices[vvlist[3],0])/2
        ty = (vertices[vvlist[2],1]+vertices[vvlist[3],1])/2
        tz = (vertices[vvlist[2],2]+vertices[vvlist[3],2])/2
        vertices[vvlist[2],0] = tx
        vertices[vvlist[2],1] = ty
        vertices[vvlist[2],2] = tz
        vertices[vvlist[3],0] = tx
        vertices[vvlist[3],1] = ty
        vertices[vvlist[3],2] = tz

#this is not an efficient implementation. just for testing!
def marching_cubes_nmc_test(flag_grid, int_grid, float_grid):

    all_vertices = []
    all_triangles = []
    all_vertices_len = 0

    config2 = np.zeros([15], np.int32)
    for i in range(15):
        config2[i] = 2**i

    dimx,dimy,dimz,_ = int_grid.shape
    dimx -= 1
    dimy -= 1
    dimz -= 1
    for i in range(dimx):
        for j in range(dimy):
            for k in range(dimz):
                if flag_grid is None or flag_grid[i,j,k]>=0:

                    #prepare config
                    #int_grid channels: corner V (1) + face V (f0,f3,f4) (3) + internal V (1)
                    config = np.zeros([15], np.int32)
                    config[0] = int_grid[i,j,k,0] #v0
                    config[1] = int_grid[i+1,j,k,0] #v1
                    config[2] = int_grid[i+1,j+1,k,0] #v2
                    config[3] = int_grid[i,j+1,k,0] #v3
                    config[4] = int_grid[i,j,k+1,0] #v4
                    config[5] = int_grid[i+1,j,k+1,0] #v5
                    config[6] = int_grid[i+1,j+1,k+1,0] #v6
                    config[7] = int_grid[i,j+1,k+1,0] #v7
                    config[8] = int_grid[i,j,k,1] #f0
                    config[9] = int_grid[i,j,k+1,1] #f1
                    config[10] = int_grid[i+1,j,k,2] #f2
                    config[11] = int_grid[i,j,k,2] #f3
                    config[12] = int_grid[i,j,k,3] #f4
                    config[13] = int_grid[i,j+1,k,3] #f5
                    config[14] = int_grid[i,j,k,4] #c
                    config = (config>0).astype(np.int32)
                    
                    if np.sum(config[:8])==0 or np.sum(config[:8])==8:
                        continue

                    #prepare vertices
                    #all_points: (8+8+12+24)*3 = 52*3
                    #partial_points: 8*3+3*1+12*2 = 51
                    # center_points_vi_x/y/z (8*3),
                    # edge0_x (1), edge3_y (1), edge8_z (1),
                    # face_points_f0_x/y (4*2), face_points_f3_y/z (4*2), face_points_f4_x/z (4*2)
                    vertices = np.copy(all_points)
                    for ii in range(8): #center points
                        vertices[8+ii,:] = float_grid[i,j,k,ii*3:ii*3+3]

                    #edge points i
                    vertices[8+8+0,0] = float_grid[i,j,k,8*3+0]
                    vertices[8+8+2,0] = float_grid[i,j+1,k,8*3+0]
                    vertices[8+8+4,0] = float_grid[i,j,k+1,8*3+0]
                    vertices[8+8+6,0] = float_grid[i,j+1,k+1,8*3+0]
                    #edge points j
                    vertices[8+8+3,1] = float_grid[i,j,k,8*3+1]
                    vertices[8+8+1,1] = float_grid[i+1,j,k,8*3+1]
                    vertices[8+8+7,1] = float_grid[i,j,k+1,8*3+1]
                    vertices[8+8+5,1] = float_grid[i+1,j,k+1,8*3+1]
                    #edge points k
                    vertices[8+8+8,2] = float_grid[i,j,k,8*3+2]
                    vertices[8+8+9,2] = float_grid[i+1,j,k,8*3+2]
                    vertices[8+8+11,2] = float_grid[i,j+1,k,8*3+2]
                    vertices[8+8+10,2] = float_grid[i+1,j+1,k,8*3+2]


                    fvlist = [0,0,0,0]
                    vvlist = [0,0,0,0]
                    
                    #face points f0
                    f0 = 0
                    ii = 0
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,1] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)

                    #face points f3
                    f0 = 3
                    ii = 1
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,1] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)
                    
                    #face points f4
                    f0 = 4
                    ii = 2
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)
                    
                    #face points f1
                    f0 = 1
                    ii = 0
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j,k+1,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,1] = float_grid[i,j,k+1,8*3+3+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)
                    
                    #face points f2
                    f0 = 2
                    ii = 1
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,1] = float_grid[i+1,j,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i+1,j,k,8*3+3+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)
                    
                    #face points f5
                    f0 = 5
                    ii = 2
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j+1,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i,j+1,k,8*3+3+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)



                    idx = np.sum(config*config2)
                    
                    triangles_len = 0
                    while True:
                        if tessellations[idx,triangles_len,0]<0: break
                        triangles_len += 1

                    vertices[:,0] += i
                    vertices[:,1] += j
                    vertices[:,2] += k
                    
                    triangles = tessellations[idx,:triangles_len]+all_vertices_len
                    
                    all_vertices.append(vertices)
                    all_triangles.append(triangles)
                    
                    all_vertices_len += len(vertices)

    if len(all_vertices)>0 and len(all_triangles)>0:
        all_vertices = np.concatenate(all_vertices,axis=0)
        all_triangles = np.concatenate(all_triangles,axis=0)
        #all_vertices, all_triangles = remove_useless_and_duplicated_points(all_vertices, all_triangles)

    return all_vertices, all_triangles


def remove_useless_and_duplicated_points(vertices, triangles):
    vertices_ = np.full( [len(vertices),3], -1, np.float32 )
    vertices_original_idx = np.full( [len(vertices)], -1, np.int32 )
    triangles_ = np.full( [len(triangles),3], -1, np.int32 )
    merge_threshold = 1e-5
    counter=0
    for t in range(len(triangles)):
        for i in range(3):
            this_vi = triangles[t,i]
            this_v = vertices[this_vi]
            same_flag = -1
            if counter>0:
                #check if the same vertex is already in vertices_
                dist = np.abs(vertices_original_idx[:counter]-this_vi)
                min_dist_idx = np.argmin(dist)
                min_dist = dist[min_dist_idx]
                if min_dist==0:
                    same_flag = min_dist_idx

                #only merge boundary vertices
                if same_flag<0:
                    if this_v[0]-int(this_v[0])==0 or this_v[1]-int(this_v[1])==0 or this_v[2]-int(this_v[2])==0:
                        dist = np.max(np.abs(vertices_[:counter]-this_v),axis=1)
                        min_dist_idx = np.argmin(dist)
                        min_dist = dist[min_dist_idx]
                        if min_dist<merge_threshold:
                            same_flag = min_dist_idx
            if same_flag>=0:
                triangles_[t,i] = same_flag
            else:
                vertices_[counter] = this_v
                vertices_original_idx[counter] = this_vi
                triangles_[t,i] = counter
                counter += 1
    return vertices_[:counter], triangles_


def write_obj_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(int(triangles[ii,0]+1))+" "+str(int(triangles[ii,1]+1))+" "+str(int(triangles[ii,2]+1))+"\n")
    fout.close()

def write_ply_triangle(name, vertices, triangles):
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

def write_ply_point(name, vertices):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    fout.close()
