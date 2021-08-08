# original case id, # connected 1/inside components, # connected 0/outside components
coloring_configs = [
["000000002222222", 0, 1],
["000000012222222", 1, 1],
["000000112222222", 1, 1],
["000001012022220", 2, 1],
["000001012022221", 1, 1],
["000001012122222", 1, 1],
["000001112222222", 1, 1],
["000011112222222", 1, 1],
["000101002222220", 2, 1],
["000101002222221", 1, 1],
["000101012022220", 2, 1],
["000101012022221", 1, 1],
["000101012122222", 1, 1],
["000101112222222", 1, 1],
["000110102020202", 3, 1],
["000110102020210", 2, 1],
["000110102020211", 1, 1],
["000110102021212", 1, 1],
["000110102121210", 1, 2],
["000110102121211", 1, 1],
["000110112222222", 1, 1],
["000111102220200", 2, 1],
["000111102220201", 1, 1],
["000111102220212", 1, 1],
["001111002200220", 2, 1],
["001111002200221", 1, 1],
["001111002201222", 1, 1],
["010110100000002", 4, 1],
["010110100000012", 3, 1],
["010110100000110", 2, 1],
["010110100000111", 1, 1],
["010110100001010", 2, 1],
["010110100001011", 1, 1],
["010110100001112", 1, 1],
["010110100101010", 2, 2],
["010110100101011", 1, 2],
["010110100101102", 1, 1],
]


import os
import cv2
import numpy as np


#
#     vertices              edges                faces
#         7 ________ 6           _____6__             ________
#         /|       /|         7/|       /|          /|       /|
#       /  |     /  |        /  |     /5 |        /  | 1   /  |
#   4 /_______ /    |      /__4____ /    10     /_______ 5    |
#    |     |  |5    |     |    11  |     |     |  3  |  |     |
#    |    3|__|_____|2    |     |__|__2__|     |     |__|__2__|
#    |    /   |    /      8   3/   9    /      |    4   |    /
#    |  /     |  /        |  /     |  /1       |  /    0|  /
#    |/_______|/          |/___0___|/          |/___ ___|/
#   0          1
#


# ---------- <copied from template.py> ----------

# ---------- define points ----------
all_points_len = 0
all_points = []
all_points_color = []

cube_points = np.array([
[0,1,0], [1,1,0], [1,1,1], [0,1,1],
[0,0,0], [1,0,0], [1,0,1], [0,0,1],
],np.float32)
cube_points = cube_points-0.5
cube_points_color = [
(  0,  0,255), (  0,128,255), (  0,255,255), (  0,255,  0),
(255,255,  0), (255,  0,  0), (255,  0,255), (128,  0,255),
]
all_points.append(cube_points)
all_points_color += cube_points_color
all_points_len = len(all_points_color)


small_cube_points = cube_points*0.25
small_cube_points_color = [
(  0,  0,192), (  0, 96,192), (  0,192,192), (  0,192,  0),
(192,192,  0), (192,  0,  0), (192,  0,192), ( 96,  0,192),
]
all_points.append(small_cube_points)
all_points_color += small_cube_points_color
all_points_len = len(all_points_color)


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
edge_points_color = [(192,192,192)]*12
all_points.append(edge_points)
all_points_color += edge_points_color
all_points_len = len(all_points_color)


face_from_cube_points_idx = np.array([[0,1,2,3], [4,5,6,7], [1,2,6,5], [0,3,7,4], [0,1,5,4], [3,2,6,7]],np.int32)
face_from_cube_points_idx_unique_feature = np.product(face_from_cube_points_idx+1,axis=1)
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
find_face_point_from_cube_point_and_face[find_face_point_from_cube_point_and_face>=0] += all_points_len
face_points = np.reshape(face_points,[-1,3])
face_points_color = small_cube_points_color*3
all_points.append(face_points)
all_points_color += face_points_color
all_points_len = len(all_points_color)

all_points = np.concatenate(all_points,axis=0)
if all_points_len != 8+8+12+6*4 or len(all_points) != all_points_len:
	print("ERROR: this is not what we agreed!")
	exit(0)

# ---------- end of <copied from template.py> ----------


# the corrected faces with respect to the normal directions
# this is crucial for deciding triangle normals
face_from_cube_points_idx = np.array([[3,2,1,0], [4,5,6,7], [1,2,6,5], [4,7,3,0], [0,1,5,4], [7,6,2,3]],np.int32)


def write_obj_triangle(name, vertices, triangles):
	fout = open(name, 'w')
	for ii in range(len(vertices)):
		fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("f "+str(int(triangles[ii,0]+1))+" "+str(int(triangles[ii,1]+1))+" "+str(int(triangles[ii,2]+1))+"\n")
	fout.close()


# get mesh tessellation for all unique cases
save_dir = "output_mesh/"
if not os.path.exists(save_dir):
	os.makedirs(save_dir)
	
for cidx in range(len(coloring_configs)):
	# ---------- load connections ----------
	config_py_name = "configs_done/"+coloring_configs[cidx][0]+".py"
	fin = open(config_py_name, 'r')
	
	line = fin.readline()
	if not line.startswith("config_string = "):
		print("ERROR: invalid config file -1- "+config_py_name)
		exit(-1)
	exec(line.strip())
	if config_string != coloring_configs[cidx][0]:
		print("ERROR: invalid config file -2- "+config_py_name)
		exit(-1)
	config = [int(i) for i in config_string]
	config = np.array(config, np.int32)

	line = fin.readline()
	line = fin.readline()
	if not line.startswith("load_connections = "):
		print("ERROR: invalid config file -3- "+config_py_name)
		exit(-1)
	exec(line.strip())
	
	fin.close()
	
	load_connections_len = len(load_connections)
	load_connections = np.array(load_connections,np.int32)

	#print("Connections loaded from -- "+config_py_name)
	
	#make the first index always smaller than the second
	for i in range(load_connections_len):
		if load_connections[i,0]>load_connections[i,1]:
			load_connections[i,:] = load_connections[i,::-1]


	# ---------- output mesh in a cube ----------
	#the key is to decide normal direction
	
	triangles = []
	#brute force search
	edge_dict = {}
	for i in range(load_connections_len):
		edge_dict[(load_connections[i,0]),load_connections[i,1]] = 1
	for i in range(load_connections_len):
		t1 = load_connections[i,0]
		t2 = load_connections[i,1]
		for j in range(load_connections_len):
			if load_connections[j,0]==t1 and load_connections[j,1]!=t2:
				t3 = load_connections[j,1]
				if (t2,t3) in edge_dict:
					triangles.append([t1,t2,t3])
					
	triangles = np.array(triangles, np.int32)
	triangles_len = len(triangles)
	triangles_normal_checked = np.zeros([triangles_len], np.int32)
	
	#correct triangle normal direction
	
	# 1. find triangles whose directions are known according to the edge points and vertex colors
	
	for ti in range(triangles_len):
		v1i = triangles[ti,0]
		v2i = triangles[ti,1]
		v3i = triangles[ti,2]
		has_edge_point = False
		if v1i>=8+8 and v1i<8+8+12:
			has_edge_point = True
		if v2i>=8+8 and v2i<8+8+12:
			has_edge_point = True
			v1i, v2i = v2i, v1i
		if v3i>=8+8 and v3i<8+8+12:
			has_edge_point = True
			v1i, v3i = v3i, v1i
		if not has_edge_point:
			continue
		#now v1i is the edge point
		
		if v2i<8+8+12:
			v2i, v3i = v3i, v2i
		if v2i<8+8+12:
			continue
		#now v2i is the face point
		
		#check v2i on which face -> then check v1i on which edge -> then check the signs of the edge endpoints
		for fi in range(6):
			for fpi in range(4):
				pi = face_from_cube_points_idx[fi,fpi]
				pi_previous = face_from_cube_points_idx[fi,(fpi+3)%4]
				pi_next = face_from_cube_points_idx[fi,(fpi+1)%4]
				if v2i==find_face_point_from_cube_point_and_face[pi,fi]:
					if v1i==find_edge_point_from_cube_points[pi,pi_previous]:
						if config[pi]>0:
							if triangles_normal_checked[ti]==0: # assign
								triangles[ti] = [v1i,v2i,v3i]
								triangles_normal_checked[ti] = 1
							else: # double check
								if triangles[ti,0]!=v1i or triangles[ti,1]!=v2i or triangles[ti,2]!=v3i:
									print("ERROR: double check failed -1- ")
									exit(-1)
						else:
							if triangles_normal_checked[ti]==0: # assign
								triangles[ti] = [v1i,v3i,v2i]
								triangles_normal_checked[ti] = 1
							else: # double check
								if triangles[ti,0]!=v1i or triangles[ti,1]!=v3i or triangles[ti,2]!=v2i:
									print("ERROR: double check failed -2- ")
									exit(-1)
						
					if v1i==find_edge_point_from_cube_points[pi,pi_next]:
						if config[pi]>0:
							if triangles_normal_checked[ti]==0: # assign
								triangles[ti] = [v1i,v3i,v2i]
								triangles_normal_checked[ti] = 1
							else: # double check
								if triangles[ti,0]!=v1i or triangles[ti,1]!=v3i or triangles[ti,2]!=v2i:
									print("ERROR: double check failed -3- ")
									exit(-1)
						else:
							if triangles_normal_checked[ti]==0: # assign
								triangles[ti] = [v1i,v2i,v3i]
								triangles_normal_checked[ti] = 1
							else: # double check
								if triangles[ti,0]!=v1i or triangles[ti,1]!=v2i or triangles[ti,2]!=v3i:
									print("ERROR: double check failed -4- ")
									exit(-1)
									
		#verify normal is corrected
		if triangles_normal_checked[ti]==0:
			print("ERROR: normal is not corrected")
			exit(-1)
	
	# 2. propagate the directions to other triangles, and check if there are conflicts
	while np.any(triangles_normal_checked==0):
		for ti in range(triangles_len):
			if triangles_normal_checked[ti]==1: #propagate
				v1i = triangles[ti,0]
				v2i = triangles[ti,1]
				v3i = triangles[ti,2]
				v123i = [v1i,v2i,v3i]
				for tti in range(triangles_len):
					if tti==ti: continue
					vv1i = triangles[tti,0]
					vv2i = triangles[tti,1]
					vv3i = triangles[tti,2]
					if vv1i not in v123i:
						vv1i, vv3i = vv3i, vv1i
					if vv2i not in v123i:
						vv2i, vv3i = vv3i, vv2i
					if vv1i not in v123i or vv2i not in v123i:
						continue
					#now vv1i--vv2i is the shared edge
					
					if triangles_normal_checked[tti]==0: # assign
						if vv1i==v2i and vv2i==v3i or vv1i==v3i and vv2i==v2i:
							triangles[tti] = [vv3i,v3i,v2i]
						elif vv1i==v1i and vv2i==v3i or vv1i==v3i and vv2i==v1i:
							triangles[tti] = [v3i,vv3i,v1i]
						elif vv1i==v1i and vv2i==v2i or vv1i==v2i and vv2i==v1i:
							triangles[tti] = [v2i,v1i,vv3i]
						else:
							print("ERROR: duplicated triangles -1-")
							exit(-1)
						triangles_normal_checked[tti] = 1
					else: # double check
						if vv1i==v2i and vv2i==v3i or vv1i==v3i and vv2i==v2i:
							if (triangles[tti,0]==vv3i and triangles[tti,1]==v3i and triangles[tti,2]==v2i) or (triangles[tti,0]==v3i and triangles[tti,1]==v2i and triangles[tti,2]==vv3i) or (triangles[tti,0]==v2i and triangles[tti,1]==vv3i and triangles[tti,2]==v3i):
								pass
							else:
								print("ERROR: double check failed -5- ", vv3i,v3i,v2i,"   ",triangles[tti,0],triangles[tti,1],triangles[tti,2])
								exit(-1)
						elif vv1i==v1i and vv2i==v3i or vv1i==v3i and vv2i==v1i:
							if (triangles[tti,0]==v3i and triangles[tti,1]==vv3i and triangles[tti,2]==v1i) or (triangles[tti,0]==vv3i and triangles[tti,1]==v1i and triangles[tti,2]==v3i) or (triangles[tti,0]==v1i and triangles[tti,1]==v3i and triangles[tti,2]==vv3i):
								pass
							else:
								print("ERROR: double check failed -6- ", v3i,vv3i,v1i,"   ",triangles[tti,0],triangles[tti,1],triangles[tti,2])
								exit(-1)
						elif vv1i==v1i and vv2i==v2i or vv1i==v2i and vv2i==v1i:
							if (triangles[tti,0]==v2i and triangles[tti,1]==v1i and triangles[tti,2]==vv3i) or (triangles[tti,0]==v1i and triangles[tti,1]==vv3i and triangles[tti,2]==v2i) or (triangles[tti,0]==vv3i and triangles[tti,1]==v2i and triangles[tti,2]==v1i):
								pass
							else:
								print("ERROR: double check failed -7- ", v2i,v1i,vv3i,"   ",triangles[tti,0],triangles[tti,1],triangles[tti,2])
								exit(-1)
						else:
							print("ERROR: duplicated triangles -2-")
							exit(-1)
							
	write_obj_triangle(save_dir+config_string+".obj", all_points, triangles)
	coloring_configs[cidx].append(triangles)




# ---------- the above is copied from 5_get_tessellation.py ----------

#get max_num_of_triangles
max_num_of_triangles = 0
for cidx in range(len(coloring_configs)):
	triangles = coloring_configs[cidx][3]
	if max_num_of_triangles<len(triangles):
		max_num_of_triangles = len(triangles)
max_num_of_triangles = max_num_of_triangles+1


LUT_file = np.load("LUT.npz")
LUT = LUT_file["LUT"] #[32768]
LUT_mapping = LUT_file["LUT_mapping"] #[32768,8]
LUT_inversed = LUT_file["LUT_inversed"] #[32768]


LUT_tess = np.full([32768,max_num_of_triangles,3],-1,np.int32)
for idx in range(32768):
	tnum = idx
	config = np.zeros([15], np.int32)
	for j in range(15):
		config[j] = tnum%2
		tnum = tnum//2
	
	cidx = LUT[idx]
	mapping_ = LUT_mapping[idx]
	# template[mapping_] = original
	# we need inverse mapping
	mapping = np.zeros([8], np.int32)
	for j in range(8):
		for k in range(8):
			if mapping_[k]==j:
				mapping[j] = k
	
	#verify mapping
	configc = [int(i) for i in coloring_configs[cidx][0]]
	configc = np.array(configc, np.int32)
	configg = np.copy(config)
	if LUT_inversed[idx]: configg[:14] = 1-configg[:14]
	if np.any(configc[:8] != configg[mapping]):
		print("ERROR: wrong mapping", configc[:8], configg[mapping], configg[:8])
		exit(-1)
	
	
	
	
	#face mapping
	face_mapping = np.zeros([6], np.int32)
	for j in range(6):
		actual_face = mapping[face_from_cube_points_idx[j]]
		actual_face_feature = np.product(actual_face+1)
		face_mapping[j] = np.where(face_from_cube_points_idx_unique_feature==actual_face_feature)[0]
	
	#whether this mapping mirrors the cube -> cause inversed normals
	new_face = mapping[face_from_cube_points_idx[0]]
	mirrored_flag = True
	j = face_mapping[0]
	for k in range(4):
		if face_from_cube_points_idx[j,(k)%4]==new_face[0] and face_from_cube_points_idx[j,(k+1)%4]==new_face[1] and face_from_cube_points_idx[j,(k+2)%4]==new_face[2] and face_from_cube_points_idx[j,(k+3)%4]==new_face[3]:
			mirrored_flag = False

	#flip the colors if necessary
	if LUT_inversed[idx]:
		mirrored_flag = not mirrored_flag
	
	#find mappings
	point_mappings = np.zeros([8+8+12+6*4], np.int32)
	#cube_points
	for i in range(8):
		point_mappings[i] = mapping[i]
	#small_cube_points
	for i in range(8):
		point_mappings[i+8] = mapping[i]+8
	#edge_points
	for i in range(len(edge_from_cube_points_idx)):
		p1i = edge_from_cube_points_idx[i,0]
		p2i = edge_from_cube_points_idx[i,1]
		point_mappings[i+8+8] = find_edge_point_from_cube_points[mapping[p1i],mapping[p2i]]
	#face_points
	for i in range(8):
		for j in range(6):
			old_idx = find_face_point_from_cube_point_and_face[i,j]
			if old_idx>=0:
				new_idx = find_face_point_from_cube_point_and_face[mapping[i],face_mapping[j]]
				point_mappings[old_idx] = new_idx
	
	
	#recover the true tessellation
	triangles = coloring_configs[cidx][3]
	if len(triangles)>0:
		if mirrored_flag:
			LUT_tess[idx,:len(triangles)] = point_mappings[triangles[:,::-1]]
		else:
			LUT_tess[idx,:len(triangles)] = point_mappings[triangles]


np.savez('LUT_tess', LUT_tess=LUT_tess)







