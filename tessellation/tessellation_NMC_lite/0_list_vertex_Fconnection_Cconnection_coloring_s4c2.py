import numpy as np

#there are 8 vertices
#--> predict 8 binary values for the vertices

#for each face, there are 7 situations
#00  10  11  \1\0  1/0/  11  11
#00  00  00  0\1\  /0/1  10  11
#only one case has ambiguity of different connections
#--> predict 6 binary values for the faces

#for the center, is there a tunnel connecting two sides?
#--> predict 1 binary value for the center

#so, how many unique and valid situations?
#brute-force search -- 2^15 = 32768
#Good!!! a not-so-large LookUp Table!

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


#S4×C2 - 48 same configs
list_V_s4c2 = np.zeros([3,2,2,4,8], np.int32)
list_V_s4c2[0,0,0,0] = [0,1,2,3,4,5,6,7]
list_V_s4c2[1,0,0,0] = [3,2,6,7,0,1,5,4]
list_V_s4c2[2,0,0,0] = [2,1,5,6,3,0,4,7]
list_V_s4c2[:,1,0,0,:4] = list_V_s4c2[:,0,0,0,4:]
list_V_s4c2[:,1,0,0,4:] = list_V_s4c2[:,0,0,0,:4]
list_V_s4c2[:,:,1,0,:] = list_V_s4c2[:,:,0,0,::-1]
list_V_s4c2[:,:,:,1,:] = list_V_s4c2[:,:,:,0,[1,2,3,0,5,6,7,4]]
list_V_s4c2[:,:,:,2,:] = list_V_s4c2[:,:,:,0,[2,3,0,1,6,7,4,5]]
list_V_s4c2[:,:,:,3,:] = list_V_s4c2[:,:,:,0,[3,0,1,2,7,4,5,6]]
list_V_s4c2 = np.reshape(list_V_s4c2, [48,8])

edge_from_cube_points_idx = np.array([[0,1], [1,2], [2,3], [0,3], [4,5], [5,6], [6,7], [4,7], [0,4], [1,5], [2,6], [3,7]],np.int32)
face_from_cube_points_idx = np.array([[0,1,2,3], [4,5,6,7], [1,2,6,5], [0,3,7,4], [0,1,5,4], [3,2,6,7]],np.int32)
face_from_cube_points_idx_unique_feature = np.product(face_from_cube_points_idx+1,axis=1)

list_F_s4c2 = np.zeros([48,6], np.int32)
#find faces for each config
for i in range(48):
	for j in range(6):
		actual_face = list_V_s4c2[i,face_from_cube_points_idx[j]]
		actual_face_feature = np.product(actual_face+1)
		list_F_s4c2[i,j] = np.where(face_from_cube_points_idx_unique_feature==actual_face_feature)[0]


#list all unique configs
#remove invalid ones later
#config [v0 ... v7, f0 ... f5, c]
coloring_configs = np.zeros([32768,15], np.int32)
coloring_configs_len = 0
LUT = np.full([32768], -1, np.int32)
LUT_mapping = np.full([32768,8], -1, np.int32)
LUT_inversed = np.full([32768], -1, np.int32)

def convert_to_valid_coloring(config):
	#face binary value only takes 0/1 when the face has two diagonal 0s and 1s
	#in other cases, set to 2 as N/A
	#F0
	if not(config[0]==config[2] and config[1]==config[3] and config[0]!=config[1]): config[8+0] = 2
	#F1
	if not(config[4]==config[6] and config[5]==config[7] and config[4]!=config[5]): config[8+1] = 2
	#F2
	if not(config[1]==config[6] and config[2]==config[5] and config[1]!=config[2]): config[8+2] = 2
	#F3
	if not(config[0]==config[7] and config[3]==config[4] and config[0]!=config[3]): config[8+3] = 2
	#F4
	if not(config[0]==config[5] and config[1]==config[4] and config[0]!=config[1]): config[8+4] = 2
	#F5
	if not(config[2]==config[7] and config[3]==config[6] and config[2]!=config[3]): config[8+5] = 2


	#tunnel C is effective when there are exactly 1/2, 2/1, 2/2 connected components
	#in other cases, set to 2 as N/A
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
		pass
	else:
		config[14] = 2


def is_same_coloring(config, coloring):
	config_V = config[0:8]
	config_F = config[8:14]
	config_C = config[14]
	coloring_V = coloring[0:8]
	coloring_F = coloring[8:14]
	coloring_C = coloring[14]
	if config_C != coloring_C:
		return False, None
	for i in range(48):
		coloring_V_t = coloring_V[list_V_s4c2[i]]
		coloring_F_t = coloring_F[list_F_s4c2[i]]
		if np.all(config_V == coloring_V_t) and np.all(config_F == coloring_F_t):
			return True, list_V_s4c2[i]
	return False, None

def find_in_coloring_configs(config_):
	config1 = np.copy(config_)
	config2 = np.copy(config_)
	config2[:14] = 1-config2[:14] #considering inverse all vertices
	convert_to_valid_coloring(config1)
	convert_to_valid_coloring(config2)
	for i in range(coloring_configs_len):
		same_flag, mapping = is_same_coloring(config1,coloring_configs[i])
		if same_flag:
			return i, mapping, False #not inversed
		same_flag, mapping = is_same_coloring(config2,coloring_configs[i])
		if same_flag:
			return i, mapping, True #inversed
	return -1, np.array([0,1,2,3,4,5,6,7],np.int32), False


for LUT_idx_invert in range(32768):
	tnum = LUT_idx_invert
	LUT_idx = 0
	config = np.zeros([15], np.int32)
	for i in range(15):
		config[i] = tnum%2
		LUT_idx = LUT_idx*2+config[i]
		tnum = tnum//2
	config[:] = config[::-1]

	convert_to_valid_coloring(config)
	coloring_idx, mapping, inversed = find_in_coloring_configs(config)
	if coloring_idx<0:
		LUT[LUT_idx] = coloring_configs_len
		LUT_mapping[LUT_idx] = mapping
		LUT_inversed[LUT_idx] = inversed
		coloring_configs[coloring_configs_len] = config
		coloring_configs_len += 1
		print(LUT_idx, coloring_configs_len)
	else:
		LUT[LUT_idx] = coloring_idx
		LUT_mapping[LUT_idx] = mapping
		LUT_inversed[LUT_idx] = inversed


print(coloring_configs_len)


for i in range(coloring_configs_len):
	string_to_print = '"'
	for j in range(15):
		string_to_print += str(coloring_configs[i,j])
	string_to_print += '",'
	print(string_to_print)


coloring_configs = coloring_configs[:coloring_configs_len]
np.savez('LUT', LUT=LUT, LUT_mapping=LUT_mapping, LUT_inversed=LUT_inversed)


#result 37x15
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



