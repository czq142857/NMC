#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

# Cython specific imports
import numpy as np
cimport numpy as np
import cython
np.import_array()


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

import utils
cdef float[:,::1] all_points = utils.all_points
cdef int[:,::1] edge_from_cube_points_idx = utils.edge_from_cube_points_idx
cdef int[:,::1] find_edge_point_from_cube_points = utils.find_edge_point_from_cube_points
cdef int[:,::1] face_from_cube_points_idx = utils.face_from_cube_points_idx
cdef int[:,::1] find_face_point_from_cube_point_and_face = utils.find_face_point_from_cube_point_and_face
cdef int[:,:,::1] tessellations = utils.tessellations


def correct_face_vertices_when_0011_0110(int[::1] config, float[:,::1] vertices, int[::1] fvlist, int[::1] vvlist):
    cdef float tx,ty,tz
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




def marching_cubes_nmc(int[:,:,:,::1] int_grid, float[:,:,:,::1] float_grid):

    #arrays to store vertices and triangles
    #will grow dynamically according to the number of actual vertices and triangles
    cdef int all_vertices_len = 0
    cdef int all_vertices_len_old = 0
    cdef int all_triangles_len = 0
    cdef int all_vertices_max = 16384
    cdef int all_triangles_max = 16384
    all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
    all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
    cdef float[:,::1] all_vertices = all_vertices_
    cdef int[:,::1] all_triangles = all_triangles_
    cdef float[:,::1] all_vertices_old = all_vertices_
    cdef int[:,::1] all_triangles_old = all_triangles_

    #tmp array for each cube
    vertices_ = np.zeros([52,3], np.float32)
    vertices_new_flag_ = np.zeros([52], np.int32)
    vertices_mapping_ = np.zeros([52], np.int32)
    cdef float[:,::1] vertices = vertices_
    cdef int[::1] vertices_new_flag = vertices_new_flag_
    cdef int[::1] vertices_mapping = vertices_mapping_
    cdef int vertices_len, triangles_len

    cdef int dimx,dimy,dimz
    dimx = int_grid.shape[0] -1
    dimy = int_grid.shape[1] -1
    dimz = int_grid.shape[2] -1

    config_ = np.zeros([15], np.int32)
    cdef int[::1] config = config_

    fvlist_ = np.zeros([4], np.int32)
    cdef int[::1] fvlist = fvlist_
    vvlist_ = np.zeros([4], np.int32)
    cdef int[::1] vvlist = vvlist_

    cdef int i,j,k,ii,jj,idx,f0,v0,v2

    #record known vertices to avoid duplicated vertices
    vertices_layer_0_ = np.full([dimy,dimz,52], -1, np.int32)
    vertices_layer_1_ = np.full([dimy,dimz,52], -1, np.int32)
    cdef int[:,:,::1] vertices_layer_0 = vertices_layer_0_
    cdef int[:,:,::1] vertices_layer_1 = vertices_layer_1_
    cdef int[:,:,::1] vertices_layer_t



    for i in range(dimx):
        vertices_layer_t = vertices_layer_1
        vertices_layer_1 = vertices_layer_0
        vertices_layer_0 = vertices_layer_t
        for j in range(dimy):
            for k in range(dimz):

                    #prepare config
                    #int_grid channels: corner V (1) + face V (f0,f3,f4) (3) + internal V (1)
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

                    idx = 0
                    for ii in range(14,-1,-1):
                        if config[ii]!=0:
                            config[ii] = 1
                        idx = idx*2+config[ii]

                    jj = 0
                    for ii in range(8):
                        jj += config[ii]
                    if jj==0 or jj==8: #empty cube
                        continue

                    #prepare vertices
                    #all_points: (8+8+12+24)*3 = 52*3
                    #partial_points: 8*3+3*1+12*2 = 51
                    # center_points_vi_x/y/z (8*3),
                    # edge0_x (1), edge3_y (1), edge8_z (1),
                    # face_points_f0_x/y (4*2), face_points_f3_y/z (4*2), face_points_f4_x/z (4*2)
                    
                    for ii in range(52):
                        vertices[ii,0] = all_points[ii,0]
                        vertices[ii,1] = all_points[ii,1]
                        vertices[ii,2] = all_points[ii,2]

                    for ii in range(8): #center points
                        vertices[8+ii,0] = float_grid[i,j,k,ii*3]
                        vertices[8+ii,1] = float_grid[i,j,k,ii*3+1]
                        vertices[8+ii,2] = float_grid[i,j,k,ii*3+2]

                    #edge points i
                    vertices[16,0] = float_grid[i,j,k,24]
                    vertices[18,0] = float_grid[i,j+1,k,24]
                    vertices[20,0] = float_grid[i,j,k+1,24]
                    vertices[22,0] = float_grid[i,j+1,k+1,24]
                    #edge points j
                    vertices[19,1] = float_grid[i,j,k,25]
                    vertices[17,1] = float_grid[i+1,j,k,25]
                    vertices[23,1] = float_grid[i,j,k+1,25]
                    vertices[21,1] = float_grid[i+1,j,k+1,25]
                    #edge points k
                    vertices[24,2] = float_grid[i,j,k,26]
                    vertices[25,2] = float_grid[i+1,j,k,26]
                    vertices[27,2] = float_grid[i,j+1,k,26]
                    vertices[26,2] = float_grid[i+1,j+1,k,26]


                    #face points f0
                    if k==0:
                        f0 = 0
                        ii = 0
                        for jj in range(4):
                            v0 = face_from_cube_points_idx[f0,jj]
                            v2 = find_face_point_from_cube_point_and_face[v0,f0]
                            vertices[v2,0] = float_grid[i,j,k,27+(ii*4+jj)*2+0]
                            vertices[v2,1] = float_grid[i,j,k,27+(ii*4+jj)*2+1]
                            fvlist[jj] = v0
                            vvlist[jj] = v2
                        correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)

                    #face points f3
                    if i==0:
                        f0 = 3
                        ii = 1
                        for jj in range(4):
                            v0 = face_from_cube_points_idx[f0,jj]
                            v2 = find_face_point_from_cube_point_and_face[v0,f0]
                            vertices[v2,1] = float_grid[i,j,k,27+(ii*4+jj)*2+0]
                            vertices[v2,2] = float_grid[i,j,k,27+(ii*4+jj)*2+1]
                            fvlist[jj] = v0
                            vvlist[jj] = v2
                        correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)
                    
                    #face points f4
                    if j==0:
                        f0 = 4
                        ii = 2
                        for jj in range(4):
                            v0 = face_from_cube_points_idx[f0,jj]
                            v2 = find_face_point_from_cube_point_and_face[v0,f0]
                            vertices[v2,0] = float_grid[i,j,k,27+(ii*4+jj)*2+0]
                            vertices[v2,2] = float_grid[i,j,k,27+(ii*4+jj)*2+1]
                            fvlist[jj] = v0
                            vvlist[jj] = v2
                        correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)
                    
                    #face points f1
                    f0 = 1
                    ii = 0
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j,k+1,27+(ii*4+jj)*2+0]
                        vertices[v2,1] = float_grid[i,j,k+1,27+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)
                    
                    #face points f2
                    f0 = 2
                    ii = 1
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,1] = float_grid[i+1,j,k,27+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i+1,j,k,27+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)
                    
                    #face points f5
                    f0 = 5
                    ii = 2
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j+1,k,27+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i,j+1,k,27+(ii*4+jj)*2+1]
                        fvlist[jj] = v0
                        vvlist[jj] = v2
                    correct_face_vertices_when_0011_0110(config,vertices,fvlist,vvlist)



                    triangles_len = 0
                    while True:
                        if tessellations[idx,triangles_len,0]<0: break
                        triangles_len += 1


                    #get vertex mappings (to existing or new vertices)
                    for ii in range(52):
                        vertices_new_flag[ii] = 0
                        vertices_mapping[ii] = -1
                    for ii in range(triangles_len):
                        vertices_new_flag[tessellations[idx,ii,0]] = 1
                        vertices_new_flag[tessellations[idx,ii,1]] = 1
                        vertices_new_flag[tessellations[idx,ii,2]] = 1

                    #edges
                    if i>0:
                        if vertices_new_flag[19]>0:
                            vertices_new_flag[19] = 0
                            vertices_mapping[19] = vertices_layer_0[j,k,17]
                        if vertices_new_flag[27]>0:
                            vertices_new_flag[27] = 0
                            vertices_mapping[27] = vertices_layer_0[j,k,26]
                        if vertices_new_flag[23]>0:
                            vertices_new_flag[23] = 0
                            vertices_mapping[23] = vertices_layer_0[j,k,21]
                        if vertices_new_flag[24]>0:
                            vertices_new_flag[24] = 0
                            vertices_mapping[24] = vertices_layer_0[j,k,25]
                    if j>0:
                        if vertices_new_flag[16]>0:
                            vertices_new_flag[16] = 0
                            vertices_mapping[16] = vertices_layer_1[j-1,k,18]
                        if vertices_new_flag[25]>0:
                            vertices_new_flag[25] = 0
                            vertices_mapping[25] = vertices_layer_1[j-1,k,26]
                        if vertices_new_flag[20]>0:
                            vertices_new_flag[20] = 0
                            vertices_mapping[20] = vertices_layer_1[j-1,k,22]
                    if k>0:
                        if vertices_new_flag[17]>0:
                            vertices_new_flag[17] = 0
                            vertices_mapping[17] = vertices_layer_1[j,k-1,21]
                        if vertices_new_flag[18]>0:
                            vertices_new_flag[18] = 0
                            vertices_mapping[18] = vertices_layer_1[j,k-1,22]

                    #faces
                    if i>0:
                        if vertices_new_flag[28]>0:
                            vertices_new_flag[28] = 0
                            if vertices_layer_0[j,k,29]>=0:
                                vertices_mapping[28] = vertices_layer_0[j,k,29]
                            elif vertices_layer_0[j,k,30]>=0:
                                vertices_mapping[28] = vertices_layer_0[j,k,30]
                            elif vertices_layer_0[j,k,33]>=0:
                                vertices_mapping[28] = vertices_layer_0[j,k,33]
                            else:
                                vertices_mapping[28] = vertices_layer_0[j,k,34]
                        if vertices_new_flag[31]>0:
                            vertices_new_flag[31] = 0
                            if vertices_layer_0[j,k,30]>=0:
                                vertices_mapping[31] = vertices_layer_0[j,k,30]
                            elif vertices_layer_0[j,k,34]>=0:
                                vertices_mapping[31] = vertices_layer_0[j,k,34]
                            elif vertices_layer_0[j,k,29]>=0:
                                vertices_mapping[31] = vertices_layer_0[j,k,29]
                            else:
                                vertices_mapping[31] = vertices_layer_0[j,k,33]
                        if vertices_new_flag[35]>0:
                            vertices_new_flag[35] = 0
                            if vertices_layer_0[j,k,34]>=0:
                                vertices_mapping[35] = vertices_layer_0[j,k,34]
                            elif vertices_layer_0[j,k,33]>=0:
                                vertices_mapping[35] = vertices_layer_0[j,k,33]
                            elif vertices_layer_0[j,k,30]>=0:
                                vertices_mapping[35] = vertices_layer_0[j,k,30]
                            else:
                                vertices_mapping[35] = vertices_layer_0[j,k,29]
                        if vertices_new_flag[32]>0:
                            vertices_new_flag[32] = 0
                            if vertices_layer_0[j,k,33]>=0:
                                vertices_mapping[32] = vertices_layer_0[j,k,33]
                            elif vertices_layer_0[j,k,29]>=0:
                                vertices_mapping[32] = vertices_layer_0[j,k,29]
                            elif vertices_layer_0[j,k,34]>=0:
                                vertices_mapping[32] = vertices_layer_0[j,k,34]
                            else:
                                vertices_mapping[32] = vertices_layer_0[j,k,30]
                    if j>0:
                        if vertices_new_flag[44]>0:
                            vertices_new_flag[44] = 0
                            if vertices_layer_1[j-1,k,47]>=0:
                                vertices_mapping[44] = vertices_layer_1[j-1,k,47]
                            elif vertices_layer_1[j-1,k,46]>=0:
                                vertices_mapping[44] = vertices_layer_1[j-1,k,46]
                            elif vertices_layer_1[j-1,k,51]>=0:
                                vertices_mapping[44] = vertices_layer_1[j-1,k,51]
                            else:
                                vertices_mapping[44] = vertices_layer_1[j-1,k,50]
                        if vertices_new_flag[45]>0:
                            vertices_new_flag[45] = 0
                            if vertices_layer_1[j-1,k,46]>=0:
                                vertices_mapping[45] = vertices_layer_1[j-1,k,46]
                            elif vertices_layer_1[j-1,k,50]>=0:
                                vertices_mapping[45] = vertices_layer_1[j-1,k,50]
                            elif vertices_layer_1[j-1,k,47]>=0:
                                vertices_mapping[45] = vertices_layer_1[j-1,k,47]
                            else:
                                vertices_mapping[45] = vertices_layer_1[j-1,k,51]
                        if vertices_new_flag[49]>0:
                            vertices_new_flag[49] = 0
                            if vertices_layer_1[j-1,k,50]>=0:
                                vertices_mapping[49] = vertices_layer_1[j-1,k,50]
                            elif vertices_layer_1[j-1,k,51]>=0:
                                vertices_mapping[49] = vertices_layer_1[j-1,k,51]
                            elif vertices_layer_1[j-1,k,46]>=0:
                                vertices_mapping[49] = vertices_layer_1[j-1,k,46]
                            else:
                                vertices_mapping[49] = vertices_layer_1[j-1,k,47]
                        if vertices_new_flag[48]>0:
                            vertices_new_flag[48] = 0
                            if vertices_layer_1[j-1,k,51]>=0:
                                vertices_mapping[48] = vertices_layer_1[j-1,k,51]
                            elif vertices_layer_1[j-1,k,47]>=0:
                                vertices_mapping[48] = vertices_layer_1[j-1,k,47]
                            elif vertices_layer_1[j-1,k,50]>=0:
                                vertices_mapping[48] = vertices_layer_1[j-1,k,50]
                            else:
                                vertices_mapping[48] = vertices_layer_1[j-1,k,46]
                    if k>0:
                        if vertices_new_flag[36]>0:
                            vertices_new_flag[36] = 0
                            if vertices_layer_1[j,k-1,40]>=0:
                                vertices_mapping[36] = vertices_layer_1[j,k-1,40]
                            elif vertices_layer_1[j,k-1,41]>=0:
                                vertices_mapping[36] = vertices_layer_1[j,k-1,41]
                            elif vertices_layer_1[j,k-1,43]>=0:
                                vertices_mapping[36] = vertices_layer_1[j,k-1,43]
                            else:
                                vertices_mapping[36] = vertices_layer_1[j,k-1,42]
                        if vertices_new_flag[37]>0:
                            vertices_new_flag[37] = 0
                            if vertices_layer_1[j,k-1,41]>=0:
                                vertices_mapping[37] = vertices_layer_1[j,k-1,41]
                            elif vertices_layer_1[j,k-1,42]>=0:
                                vertices_mapping[37] = vertices_layer_1[j,k-1,42]
                            elif vertices_layer_1[j,k-1,40]>=0:
                                vertices_mapping[37] = vertices_layer_1[j,k-1,40]
                            else:
                                vertices_mapping[37] = vertices_layer_1[j,k-1,43]
                        if vertices_new_flag[38]>0:
                            vertices_new_flag[38] = 0
                            if vertices_layer_1[j,k-1,42]>=0:
                                vertices_mapping[38] = vertices_layer_1[j,k-1,42]
                            elif vertices_layer_1[j,k-1,43]>=0:
                                vertices_mapping[38] = vertices_layer_1[j,k-1,43]
                            elif vertices_layer_1[j,k-1,41]>=0:
                                vertices_mapping[38] = vertices_layer_1[j,k-1,41]
                            else:
                                vertices_mapping[38] = vertices_layer_1[j,k-1,40]
                        if vertices_new_flag[39]>0:
                            vertices_new_flag[39] = 0
                            if vertices_layer_1[j,k-1,43]>=0:
                                vertices_mapping[39] = vertices_layer_1[j,k-1,43]
                            elif vertices_layer_1[j,k-1,40]>=0:
                                vertices_mapping[39] = vertices_layer_1[j,k-1,40]
                            elif vertices_layer_1[j,k-1,42]>=0:
                                vertices_mapping[39] = vertices_layer_1[j,k-1,42]
                            else:
                                vertices_mapping[39] = vertices_layer_1[j,k-1,41]

                    all_vertices_len_old = all_vertices_len
                    vertices_len = 0
                    for ii in range(52):
                        if vertices_new_flag[ii]:
                            vertices_mapping[ii] = all_vertices_len
                            all_vertices_len += 1
                            vertices_len += 1

                    for ii in range(52):
                        vertices_layer_1[j,k,ii] = vertices_mapping[ii]

                    #grow all_vertices
                    if all_vertices_len_old+vertices_len>=all_vertices_max:
                        all_vertices_max = all_vertices_max*2
                        all_vertices_ = np.zeros([all_vertices_max,3], np.float32)
                        all_vertices = all_vertices_
                        for ii in range(all_vertices_len_old):
                            all_vertices[ii,0] = all_vertices_old[ii,0]
                            all_vertices[ii,1] = all_vertices_old[ii,1]
                            all_vertices[ii,2] = all_vertices_old[ii,2]
                        all_vertices_old = all_vertices_
                    
                    #add to all_vertices
                    all_vertices_len = all_vertices_len_old
                    for ii in range(52):
                        if vertices_new_flag[ii]:
                            all_vertices[all_vertices_len,0] = vertices[ii,0]+i
                            all_vertices[all_vertices_len,1] = vertices[ii,1]+j
                            all_vertices[all_vertices_len,2] = vertices[ii,2]+k
                            all_vertices_len += 1


                    #grow all_triangles
                    if all_triangles_len+triangles_len>=all_triangles_max:
                        all_triangles_max = all_triangles_max*2
                        all_triangles_ = np.zeros([all_triangles_max,3], np.int32)
                        all_triangles = all_triangles_
                        for ii in range(all_triangles_len):
                            all_triangles[ii,0] = all_triangles_old[ii,0]
                            all_triangles[ii,1] = all_triangles_old[ii,1]
                            all_triangles[ii,2] = all_triangles_old[ii,2]
                        all_triangles_old = all_triangles_


                    for ii in range(triangles_len):
                        all_triangles[all_triangles_len,0] = vertices_mapping[tessellations[idx,ii,0]]
                        all_triangles[all_triangles_len,1] = vertices_mapping[tessellations[idx,ii,1]]
                        all_triangles[all_triangles_len,2] = vertices_mapping[tessellations[idx,ii,2]]
                        all_triangles_len += 1

    return all_vertices_[:all_vertices_len], all_triangles_[:all_triangles_len]

