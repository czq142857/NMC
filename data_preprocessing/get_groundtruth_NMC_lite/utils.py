import numpy as np

configs = np.load("LUT_CC.npz")["LUT_CC"].astype(np.int32) #32768 x 3, [case_id, #insideCC, #outsideCC]
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


#this is not an efficient implementation. just for testing!
def marching_cubes_47_test(flag_grid, int_grid, float_grid):

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
                if flag_grid[i,j,k]>=0:

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
                    #face points f0
                    f0 = 0
                    ii = 0
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,1] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+1]
                    #face points f3
                    f0 = 3
                    ii = 1
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,1] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+1]
                    #face points f4
                    f0 = 4
                    ii = 2
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i,j,k,8*3+3+(ii*4+jj)*2+1]
                    #face points f1
                    f0 = 1
                    ii = 0
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j,k+1,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,1] = float_grid[i,j,k+1,8*3+3+(ii*4+jj)*2+1]
                    #face points f2
                    f0 = 2
                    ii = 1
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,1] = float_grid[i+1,j,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i+1,j,k,8*3+3+(ii*4+jj)*2+1]
                    #face points f5
                    f0 = 5
                    ii = 2
                    for jj in range(4):
                        v0 = face_from_cube_points_idx[f0,jj]
                        v2 = find_face_point_from_cube_point_and_face[v0,f0]
                        vertices[v2,0] = float_grid[i,j+1,k,8*3+3+(ii*4+jj)*2+0]
                        vertices[v2,2] = float_grid[i,j+1,k,8*3+3+(ii*4+jj)*2+1]


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

    return all_vertices, all_triangles



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

def read_intersection_file_as_2d_array(name):
    fp = open(name, 'rb')
    line = fp.readline().strip()
    if not line.startswith(b'#intersection'):
        raise IOError('Not an intersection file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    point_nums = np.array(list(map(int, fp.readline().strip().split(b' '))),np.int32)
    line = fp.readline()
    data = np.frombuffer(fp.read(), dtype=np.float32)
    data = data.reshape([np.sum(point_nums),3])
    fp.close()
    separated = []
    count = 0
    for i in range(len(point_nums)):
        separated.append(np.ascontiguousarray(data[count:count+point_nums[i]]))
        count += point_nums[i]
    return separated

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


