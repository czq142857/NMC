coloring_configs = [
"000000002222222",
"000000012222222",
"000000112222222",
"000001012022220",
"000001012022221",
"000001012122222",
"000001112222222",
"000011112222222",
"000101002222220",
"000101002222221",
"000101012022220",
"000101012022221",
"000101012122222",
"000101112222222",
"000110102020202",
"000110102020210",
"000110102020211",
"000110102021212",
"000110102121210",
"000110102121211",
"000110112222222",
"000111102220200",
"000111102220201",
"000111102220212",
"001111002200220",
"001111002200221",
"001111002201222",
"010110100000002",
"010110100000012",
"010110100000110",
"010110100000111",
"010110100001010",
"010110100001011",
"010110100001112",
"010110100101010",
"010110100101011",
"010110100101102",
]

# ---------- parse config ----------
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

import numpy as np

edge_from_cube_points_idx = np.array([[0,1], [1,2], [2,3], [0,3], [4,5], [5,6], [6,7], [4,7], [0,4], [1,5], [2,6], [3,7]],np.int32)
face_from_cube_points_idx = np.array([[0,1,2,3], [4,5,6,7], [1,2,6,5], [0,3,7,4], [0,1,5,4], [3,2,6,7]],np.int32)


#predict something from coloring_configs
# original case id, # connected 1/inside components, # connected 0/outside components

for cidx in range(len(coloring_configs)):
	config_string = coloring_configs[cidx]
	config = [int(i) for i in config_string]
	config = np.array(config, np.int32)
	
	connection_matrix = np.zeros([8,8], np.int32)
	
	#identity
	for i in range(8):
		connection_matrix[i,i] = 1
	
	#for each face
	for i in range(len(face_from_cube_points_idx)):
		p0 = face_from_cube_points_idx[i,0]
		p1 = face_from_cube_points_idx[i,1]
		p2 = face_from_cube_points_idx[i,2]
		p3 = face_from_cube_points_idx[i,3]
		
		if config[p0]==config[p1]:
			connection_matrix[p0,p1] = 1
			connection_matrix[p1,p0] = 1
		if config[p1]==config[p2]:
			connection_matrix[p1,p2] = 1
			connection_matrix[p2,p1] = 1
		if config[p2]==config[p3]:
			connection_matrix[p2,p3] = 1
			connection_matrix[p3,p2] = 1
		if config[p3]==config[p0]:
			connection_matrix[p3,p0] = 1
			connection_matrix[p0,p3] = 1
		
		if config[p0]==config[p2] and config[p0]==config[8+i]:
			connection_matrix[p0,p2] = 1
			connection_matrix[p2,p0] = 1
		if config[p1]==config[p3] and config[p1]==config[8+i]:
			connection_matrix[p1,p3] = 1
			connection_matrix[p3,p1] = 1

	#finish the connection matrix
	for i in range(20):
		for p0 in range(8):
			for p1 in range(8):
				if connection_matrix[p0,p1]:
					for p2 in range(8):
						if connection_matrix[p1,p2]:
							connection_matrix[p0,p2] = 1
							connection_matrix[p2,p0] = 1

	#count connected components
	colors = np.full([8],-1,np.int32)
	color_id = -1
	color_1_inside_count = 0
	color_0_outside_count = 0
	for p0 in range(8):
		if colors[p0]<0:
			color_id += 1
			colors[p0] = color_id
			if config[p0] == 0:
				color_0_outside_count += 1
			else:
				color_1_inside_count += 1
			for p1 in range(8):
				if connection_matrix[p0,p1]:
					colors[p1] = color_id
	
	#tunnel C is effective when there are exactly 1/2, 2/1, 2/2 connected components
	if (color_1_inside_count==1 and color_0_outside_count==2) or (color_1_inside_count==2 and color_0_outside_count==1) or (color_1_inside_count==2 and color_0_outside_count==2):
		if config[14]==2:
			print("ERROR: config[14]==2")
			exit(-1)
		if config[14]==1:
			if not (color_1_inside_count==2 and color_0_outside_count==2):
				color_1_inside_count = 1
				color_0_outside_count = 1
			else:
				color_1_inside_count = 1
				color_0_outside_count = 2
	else:
		if config[14]!=2:
			print("ERROR: config[14]!=2",)
			exit(-1)
	
	print('["'+config_string+'", '+str(color_1_inside_count)+', '+str(color_0_outside_count)+'],')



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
