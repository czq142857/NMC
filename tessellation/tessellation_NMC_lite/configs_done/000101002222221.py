config_string = "000101002222221"

load_connections = [[19,39],[18,39],[20,41],[21,41],[25,33],[21,33],[19,31],[27,31],[25,49],[20,49],[18,47],[27,47],[10,33],[9,25],[8,49],[12,20],[15,41],[14,21],[10,14],[9,10],[8,9],[8,12],[12,15],[14,15],[14,47],[15,27],[12,31],[8,19],[9,39],[10,18],[12,27],[14,27],[15,21],[15,20],[12,19],[8,20],[8,25],[9,19],[14,18],[9,18],[10,21],[10,25]]

import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--getimg", action="store_true", dest="getimg", default=False, help="???")
input_args = parser.parse_args()


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
config = [int(i) for i in config_string]
config = np.array(config, np.int32)


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
small_cube_points_color = [
(  0,  0,255), (  0,128,255), (  0,255,255), (  0,255,  0),
(255,255,  0), (255,  0,  0), (255,  0,255), (128,  0,255),
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
all_points_transformed = np.copy(all_points)



# ---------- define connections ----------
frame_connections = np.array([[0,16],[1,16],[1,17],[2,17],[2,18],[3,18],[3,19],[0,19],[4,20],[5,20],[5,21],[6,21],[6,22],[7,22],[7,23],[4,23],[4,24],[0,24],[5,25],[1,25],[6,26],[2,26],[7,27],[3,27]],np.int32)
frame_connections_len = len(frame_connections)
all_connections = np.zeros([1000,2],np.int32)
all_connections_len = frame_connections_len
all_connections[:all_connections_len] = frame_connections


#prepare connections according to config
prepared_connections = []

#deal with each face x 6
for fi in range(6):
	p1i = face_from_cube_points_idx[fi,0]
	p2i = face_from_cube_points_idx[fi,1]
	p3i = face_from_cube_points_idx[fi,2]
	p4i = face_from_cube_points_idx[fi,3]
	p1v = config[p1i]
	p2v = config[p2i]
	p3v = config[p3i]
	p4v = config[p4i]
	#     ________ 
	#   4|        |3
	#    |        |
	#    |        |
	#    |        |
	#   1|________|2
	# 2^4 = 16 situations
	if p1v == 0 and p2v == 0 and p3v == 0 and p4v == 0:
		pass
	if p1v == 0 and p2v == 0 and p3v == 0 and p4v == 1:
		prepared_connections.append( [find_edge_point_from_cube_points[p4i,p1i],find_face_point_from_cube_point_and_face[p4i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p4i,p3i],find_face_point_from_cube_point_and_face[p4i,fi]] )
	if p1v == 0 and p2v == 0 and p3v == 1 and p4v == 0:
		prepared_connections.append( [find_edge_point_from_cube_points[p3i,p2i],find_face_point_from_cube_point_and_face[p3i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p3i,p4i],find_face_point_from_cube_point_and_face[p3i,fi]] )
	if p1v == 0 and p2v == 0 and p3v == 1 and p4v == 1:
		prepared_connections.append( [find_edge_point_from_cube_points[p3i,p2i],find_face_point_from_cube_point_and_face[p4i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p4i,p1i],find_face_point_from_cube_point_and_face[p4i,fi]] )
	if p1v == 0 and p2v == 1 and p3v == 0 and p4v == 0:
		prepared_connections.append( [find_edge_point_from_cube_points[p2i,p1i],find_face_point_from_cube_point_and_face[p2i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p2i,p3i],find_face_point_from_cube_point_and_face[p2i,fi]] )
	if p1v == 0 and p2v == 1 and p3v == 0 and p4v == 1:
		if config[8+fi] == 0:
			prepared_connections.append( [find_edge_point_from_cube_points[p2i,p1i],find_face_point_from_cube_point_and_face[p2i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p2i,p3i],find_face_point_from_cube_point_and_face[p2i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p4i,p1i],find_face_point_from_cube_point_and_face[p4i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p4i,p3i],find_face_point_from_cube_point_and_face[p4i,fi]] )
		elif config[8+fi] == 1:
			prepared_connections.append( [find_edge_point_from_cube_points[p1i,p2i],find_face_point_from_cube_point_and_face[p1i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p1i,p4i],find_face_point_from_cube_point_and_face[p1i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p3i,p2i],find_face_point_from_cube_point_and_face[p3i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p3i,p4i],find_face_point_from_cube_point_and_face[p3i,fi]] )
		else:
			print("ERROR: undefined situation!")
			exit()
	if p1v == 0 and p2v == 1 and p3v == 1 and p4v == 0:
		prepared_connections.append( [find_edge_point_from_cube_points[p2i,p1i],find_face_point_from_cube_point_and_face[p2i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p3i,p4i],find_face_point_from_cube_point_and_face[p2i,fi]] )
	if p1v == 0 and p2v == 1 and p3v == 1 and p4v == 1:
		prepared_connections.append( [find_edge_point_from_cube_points[p1i,p2i],find_face_point_from_cube_point_and_face[p1i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p1i,p4i],find_face_point_from_cube_point_and_face[p1i,fi]] )
	if p1v == 1 and p2v == 0 and p3v == 0 and p4v == 0:
		prepared_connections.append( [find_edge_point_from_cube_points[p1i,p2i],find_face_point_from_cube_point_and_face[p1i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p1i,p4i],find_face_point_from_cube_point_and_face[p1i,fi]] )
	if p1v == 1 and p2v == 0 and p3v == 0 and p4v == 1:
		prepared_connections.append( [find_edge_point_from_cube_points[p1i,p2i],find_face_point_from_cube_point_and_face[p1i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p4i,p3i],find_face_point_from_cube_point_and_face[p1i,fi]] )
	if p1v == 1 and p2v == 0 and p3v == 1 and p4v == 0:
		if config[8+fi] == 1:
			prepared_connections.append( [find_edge_point_from_cube_points[p2i,p1i],find_face_point_from_cube_point_and_face[p2i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p2i,p3i],find_face_point_from_cube_point_and_face[p2i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p4i,p1i],find_face_point_from_cube_point_and_face[p4i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p4i,p3i],find_face_point_from_cube_point_and_face[p4i,fi]] )
		elif config[8+fi] == 0:
			prepared_connections.append( [find_edge_point_from_cube_points[p1i,p2i],find_face_point_from_cube_point_and_face[p1i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p1i,p4i],find_face_point_from_cube_point_and_face[p1i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p3i,p2i],find_face_point_from_cube_point_and_face[p3i,fi]] )
			prepared_connections.append( [find_edge_point_from_cube_points[p3i,p4i],find_face_point_from_cube_point_and_face[p3i,fi]] )
		else:
			print("ERROR: undefined situation!")
			exit()
	if p1v == 1 and p2v == 0 and p3v == 1 and p4v == 1:
		prepared_connections.append( [find_edge_point_from_cube_points[p2i,p1i],find_face_point_from_cube_point_and_face[p2i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p2i,p3i],find_face_point_from_cube_point_and_face[p2i,fi]] )
	if p1v == 1 and p2v == 1 and p3v == 0 and p4v == 0:
		prepared_connections.append( [find_edge_point_from_cube_points[p1i,p4i],find_face_point_from_cube_point_and_face[p1i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p2i,p3i],find_face_point_from_cube_point_and_face[p1i,fi]] )
	if p1v == 1 and p2v == 1 and p3v == 0 and p4v == 1:
		prepared_connections.append( [find_edge_point_from_cube_points[p3i,p2i],find_face_point_from_cube_point_and_face[p3i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p3i,p4i],find_face_point_from_cube_point_and_face[p3i,fi]] )
	if p1v == 1 and p2v == 1 and p3v == 1 and p4v == 0:
		prepared_connections.append( [find_edge_point_from_cube_points[p4i,p1i],find_face_point_from_cube_point_and_face[p4i,fi]] )
		prepared_connections.append( [find_edge_point_from_cube_points[p4i,p3i],find_face_point_from_cube_point_and_face[p4i,fi]] )
	if p1v == 1 and p2v == 1 and p3v == 1 and p4v == 1:
		pass


# ---------- point visibility ----------
# if no tunnel, only allow used points and their corresponding small_cube_points
# if has tunnel, only allow used points and all small_cube_points
all_points_visibility = np.zeros([all_points_len], np.int32)
if config[14] == 1:
	all_points_visibility[8:16] = 1

for i in range(len(prepared_connections)):
	for j in range(2):
		p1i = prepared_connections[i][j]
		all_points_visibility[p1i] = 1
		if p1i >= 8+8+12: #get corresponding small_cube_points
			#recall small_cube_points.shape is [3,8,3] -> [24,3]
			p2i = (p1i-(8+8+12))%8+8
			all_points_visibility[p2i] = 1


# ---------- load ----------
prepared_connections_ = prepared_connections
if len(load_connections)>0:
	print("\n\nConnections loaded")
	prepared_connections_ = load_connections
	#check if the used points are valid according to all_points_visibility
	for i in range(len(prepared_connections_)):
		for j in range(2):
			if all_points_visibility[prepared_connections_[i][j]]==0:
				print("ERROR: invalid points!")

if len(prepared_connections_)>0:
	all_connections[all_connections_len:all_connections_len+len(prepared_connections_)] = prepared_connections_
	all_connections_len = all_connections_len+len(prepared_connections_)


#make the first index always smaller than the second
for i in range(all_connections_len):
	if all_connections[i,0]>all_connections[i,1]:
		all_connections[i,:] = all_connections[i,::-1]


#optimize vertex positions for better visualization
all_points_avg = np.zeros([len(all_points), 3], np.float32)
all_points_avg_count = np.zeros([len(all_points)], np.float32)
for t in range(100):
	for i in range(all_connections_len):
		p1i = all_connections[i][0]
		p2i = all_connections[i][1]
		all_points_avg[p1i] += all_points[p2i]
		all_points_avg_count[p1i] += 1
		all_points_avg[p2i] += all_points[p1i]
		all_points_avg_count[p2i] += 1
	for i in range(8,16):
		all_points[i] = all_points_avg[i]/max(all_points_avg_count[i],1)




# ---------- rendering ----------
render_scale = 500
UI_height = 1000
UI_width = 1000

#  z
# /
# --x
#|
#y
def render(cam_alpha,cam_beta,img):
	sin_alpha = np.sin(cam_alpha)
	cos_alpha = np.cos(cam_alpha)
	sin_beta = np.sin(cam_beta)
	cos_beta = np.cos(cam_beta)
	
	points_x = all_points[:,0]
	points_y = all_points[:,1]
	points_z = all_points[:,2]
	
	points_x2 = cos_alpha*points_x - sin_alpha*points_z
	points_y2 = points_y
	points_z2 = sin_alpha*points_x + cos_alpha*points_z

	points_x3 = points_x2
	points_y3 = cos_beta*points_y2 - sin_beta*points_z2
	points_z3 = sin_beta*points_y2 + cos_beta*points_z2
	
	all_points_transformed[:,0] = points_x3*render_scale+UI_width/2
	all_points_transformed[:,1] = points_y3*render_scale+UI_height/2
	all_points_transformed[:,2] = points_z3
	
	all_points_connections_len = all_points_len + all_connections_len
	things_to_draw_len = all_points_len + all_connections_len + all_triangles_len
	things_to_draw_depth = np.zeros([things_to_draw_len], np.float32)
	for i in range(all_points_len):
		things_to_draw_depth[i] = all_points_transformed[i,2] - 1e-5
	for i in range(all_connections_len):
		maxv = max(all_points_transformed[all_connections[i,0],2], all_points_transformed[all_connections[i,1],2])
		avgv = (all_points_transformed[all_connections[i,0],2]+all_points_transformed[all_connections[i,1],2])/2
		things_to_draw_depth[i+all_points_len] = maxv + (avgv-maxv)*1e-5
	for i in range(all_triangles_len):
		maxv = max(max(all_points_transformed[all_triangles[i,0],2], all_points_transformed[all_triangles[i,1],2]), all_points_transformed[all_triangles[i,2],2])
		avgv = (all_points_transformed[all_triangles[i,0],2]+all_points_transformed[all_triangles[i,1],2]+all_points_transformed[all_triangles[i,2],2])/3
		things_to_draw_depth[i+all_points_connections_len] = maxv + (avgv-maxv)*1e-5
	things_to_draw_order = np.argsort(things_to_draw_depth)[::-1]
	
	
	for j in range(things_to_draw_len):
		i = things_to_draw_order[j]
		if i<all_points_len:
			point = (int(all_points_transformed[i,0]),int(all_points_transformed[i,1]))
			color = all_points_color[i]
			if i<8:
				if config[i]==1:
					ball_size = 16
					cv2.circle(img, point, ball_size, color, -1)
					ball_size = 12
					cv2.circle(img, point, ball_size, (255,255,255), -1)
					cv2.rectangle(img, (point[0]-8,point[1]-2), (point[0]+8,point[1]+2), color, -1)
				else:
					ball_size = 16
					cv2.circle(img, point, ball_size, color, -1)
					cv2.rectangle(img, (point[0]-8,point[1]-2), (point[0]+8,point[1]+2), (255,255,255), -1)
					cv2.rectangle(img, (point[0]-2,point[1]-8), (point[0]+2,point[1]+8), (255,255,255), -1)
			elif all_points_visibility[i]:
				ball_size = 8
				cv2.circle(img, point, ball_size, color, -1)
		elif i<all_points_connections_len:
			i = i-all_points_len
			p1i = all_connections[i,0]
			p1 = (int(all_points_transformed[p1i,0]),int(all_points_transformed[p1i,1]))
			p2i = all_connections[i,1]
			p2 = (int(all_points_transformed[p2i,0]),int(all_points_transformed[p2i,1]))
			if i<frame_connections_len: #base frame
				color = (200,200,200)
				line_size = 1
			else:
				c = (all_points_transformed[p1i,2] + all_points_transformed[p2i,2])*80+120
				c = int(c)
				if c<0: c = 0
				if c>200: c = 200
				color = (c,c,c)
				line_size = 6
			cv2.line(img, p1, p2, color, line_size, cv2.LINE_AA)
		else:
			i = i-all_points_connections_len
			p1i = all_triangles[i,0]
			p1 = np.array([all_points_transformed[p1i,0],all_points_transformed[p1i,1]])
			p2i = all_triangles[i,1]
			p2 = np.array([all_points_transformed[p2i,0],all_points_transformed[p2i,1]])
			p3i = all_triangles[i,2]
			p3 = np.array([all_points_transformed[p3i,0],all_points_transformed[p3i,1]])
			
			#shrink to show the edges
			shrink_amount = 3
			
			#area
			area2 = np.abs(p1[0]*p2[1]+p2[0]*p3[1]+p3[0]*p1[1]-p1[0]*p3[1]-p2[0]*p1[1]-p3[0]*p2[1])#/2
			
			#perimeter
			a = np.sqrt(np.sum(np.square(p3-p2)))
			b = np.sqrt(np.sum(np.square(p1-p3)))
			c = np.sqrt(np.sum(np.square(p2-p1)))
			perimeter = a+b+c
			
			#radius of incircle
			inradius = area2/perimeter
			
			if inradius>shrink_amount:
				#incenter
				O = (a*p1 + b*p2 + c*p3)/perimeter
				v1 = O-p1
				v1 = v1*shrink_amount/inradius
				v2 = O-p2
				v2 = v2*shrink_amount/inradius
				v3 = O-p3
				v3 = v3*shrink_amount/inradius
				
				p1 = (int(p1[0]+v1[0]),int(p1[1]+v1[1]))
				p2 = (int(p2[0]+v2[0]),int(p2[1]+v2[1]))
				p3 = (int(p3[0]+v3[0]),int(p3[1]+v3[1]))

				c = (all_points_transformed[p1i,2] + all_points_transformed[p2i,2] + all_points_transformed[p3i,2])*60+220
				c = int(c)
				if c<100: c = 100
				if c>250: c = 250
				color = (c,c-40,c-40)
				cv2.fillConvexPoly(img, np.array([p1,p2,p3],np.int32), color)

	if tmp_line_flag:
		p1 = (int(all_points_transformed[tmp_line_p1i,0]),int(all_points_transformed[tmp_line_p1i,1]))
		p2 = (int(tmp_line_p2_x),int(tmp_line_p2_y))
		color = (255,0,0)
		line_size = 2
		cv2.line(img, p1, p2, color, line_size, cv2.LINE_AA)


def detect_clicked_point(x,y):
	distance = (all_points_transformed[:,0]-x)**2 + (all_points_transformed[:,1]-y)**2
	#consider all_points_visibility
	distance = distance+(1-all_points_visibility)*10000
	min_idx = np.argmin(distance)
	if distance[min_idx]<=100:
		return min_idx
	else:
		return -1

def append_new_line(p1i,p2i):
	if p1i>p2i:
		p1i,p2i = p2i,p1i
	repeat_idx = -1
	for i in range(frame_connections_len,all_connections_len):
		if all_connections[i,0]==p1i and all_connections[i,1]==p2i:
			repeat_idx = i
			break
	if repeat_idx>=0:
		all_connections[repeat_idx]=all_connections[all_connections_len-1]
		return all_connections_len-1
	else:
		all_connections[all_connections_len,0] = p1i
		all_connections[all_connections_len,1] = p2i
		return all_connections_len+1

def compute_triangles():
	triangles = []
	#brute force search
	edge_dict = {}
	for i in range(frame_connections_len,all_connections_len):
		edge_dict[(all_connections[i,0]),all_connections[i,1]] = 1
	for i in range(frame_connections_len,all_connections_len):
		t1 = all_connections[i,0]
		t2 = all_connections[i,1]
		for j in range(frame_connections_len,all_connections_len):
			if all_connections[j,0]==t1 and all_connections[j,1]!=t2:
				t3 = all_connections[j,1]
				if (t2,t3) in edge_dict:
					triangles.append([t1,t2,t3])
	if len(triangles)>0:
		all_triangles[:len(triangles)] = np.array(triangles,np.int32)
	return len(triangles)


# ---------- triangles ----------
all_triangles = np.zeros([1000,3],np.int32)
all_triangles_len = 0
all_triangles_len = compute_triangles()


# ---------- capture mouse events ----------
mouse_xyd = np.zeros([4], np.int32)
mouse_xyd_backup = np.zeros([4], np.int32)
def mouse_ops(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		mouse_xyd[0] = x
		mouse_xyd[1] = y
		mouse_xyd[2] = 1
	elif event == cv2.EVENT_RBUTTONDOWN:
		mouse_xyd[0] = x
		mouse_xyd[1] = y
		mouse_xyd[3] = 1
	elif event == cv2.EVENT_MOUSEMOVE:
		if mouse_xyd[2] == 1 or mouse_xyd[3] == 1:
			mouse_xyd[0] = x
			mouse_xyd[1] = y
	elif event == cv2.EVENT_LBUTTONUP:
		mouse_xyd[2] = 0
	elif event == cv2.EVENT_RBUTTONUP:
		mouse_xyd[3] = 0


cam_alpha = -0.345
cam_beta = 0.355

tmp_line_p1i = 0
tmp_line_p2_x = 0
tmp_line_p2_y = 0
tmp_line_p2i = 0
tmp_line_flag = False


if input_args.getimg:
	#hide unused points
	all_points_visibility[8:] = 0
	for i in range(frame_connections_len,all_connections_len):
		for j in range(2):
			p1i = all_connections[i][j]
			all_points_visibility[p1i] = 1
	#render
	for i in range(4):
		cam_alpha = -0.345 + i*3.1415926/2
		cam_beta = 0.355
		UI_image = np.full([UI_height,UI_width,3], 255, np.uint8)
		render(cam_alpha,cam_beta,UI_image)
		cv2.imwrite(config_string+"_"+str(i)+".png", UI_image)

	#draw surface
	all_connections_len = frame_connections_len
	all_connections[:all_connections_len] = frame_connections
	if len(prepared_connections)>0:
		all_connections[all_connections_len:all_connections_len+len(prepared_connections)] = prepared_connections
		all_connections_len = all_connections_len+len(prepared_connections)
	all_triangles_len = 0
	
	#hide unused points
	all_points_visibility[8:] = 0
	for i in range(frame_connections_len,all_connections_len):
		for j in range(2):
			p1i = all_connections[i][j]
			all_points_visibility[p1i] = 1
	#render
	for i in range(4):
		cam_alpha = -0.345 + i*3.1415926/2
		cam_beta = 0.355
		UI_image = np.full([UI_height,UI_width,3], 255, np.uint8)
		render(cam_alpha,cam_beta,UI_image)
		cv2.imwrite(config_string+"_"+str(i+4)+".png", UI_image)
	
else:
	# ---------- UI starts ----------
	Window_name = "Cube"
	cv2.namedWindow(Window_name)
	cv2.setMouseCallback(Window_name,mouse_ops)

	while True:

		#deal with mouse events
		if np.any(mouse_xyd!=mouse_xyd_backup):

			if mouse_xyd[0]<UI_width and mouse_xyd[1]<UI_height: #inside cube rergion
				#left click-and-drag
				if mouse_xyd[2]==1 and mouse_xyd_backup[2]==1 and mouse_xyd_backup[0]<UI_width and mouse_xyd_backup[1]<UI_height:
					dx = mouse_xyd[0] - mouse_xyd_backup[0]
					dy = mouse_xyd[1] - mouse_xyd_backup[1]
					cam_alpha += dx/200.0
					cam_beta += dy/200.0
					if cam_beta>1.7: cam_beta=1.7
					if cam_beta<-1.7: cam_beta=-1.7
				
				#right click-and-drag
				if mouse_xyd_backup[3]==0 and mouse_xyd[3]==1:
					tmp_line_p2_x = mouse_xyd[0]
					tmp_line_p2_y = mouse_xyd[1]
					tmp_line_p1i = detect_clicked_point(tmp_line_p2_x,tmp_line_p2_y)
					if tmp_line_p1i>=0:
						tmp_line_flag = True
				if mouse_xyd_backup[3]==1 and mouse_xyd[3]==1:
					if tmp_line_flag:
						tmp_line_p2_x = mouse_xyd[0]
						tmp_line_p2_y = mouse_xyd[1]
				if mouse_xyd_backup[3]==1 and mouse_xyd[3]==0:
					if tmp_line_flag:
						tmp_line_flag = False
						tmp_line_p2_x = mouse_xyd[0]
						tmp_line_p2_y = mouse_xyd[1]
						tmp_line_p2i = detect_clicked_point(tmp_line_p2_x,tmp_line_p2_y)
						if tmp_line_p2i>=0 and tmp_line_p2i!=tmp_line_p1i:
							all_connections_len = append_new_line(tmp_line_p1i,tmp_line_p2i)
							all_triangles_len = compute_triangles()
							
							string_to_print = ""
							for i in range(frame_connections_len,all_connections_len):
								string_to_print += "["+str(all_connections[i,0])+","+str(all_connections[i,1])+"],"
							print("Connections:")
							print(string_to_print)
							print()

			mouse_xyd_backup[:] = mouse_xyd[:]

		UI_image = np.full([UI_height,UI_width,3], 255, np.uint8)
		
		render(cam_alpha,cam_beta,UI_image)
		
		cv2.imshow(Window_name, UI_image)
		key = cv2.waitKey(1)
		if key == 32: #space
			break
