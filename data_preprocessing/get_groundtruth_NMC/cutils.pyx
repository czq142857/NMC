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


def get_face_center_value(char[:,::1] cell_face, int[:, ::1] queue):
    cdef int dimx,dimy
    cdef int i,j,pi,pj
    cdef int v0,v1,v2,v3

    dimx = cell_face.shape[0]
    dimy = cell_face.shape[1]
    v0 = cell_face[0,0]
    v1 = cell_face[dimx-1,0]
    v2 = cell_face[dimx-1,dimy-1]
    v3 = cell_face[0,dimy-1]

    #check edge 0-1
    pi = cell_face[0,0]
    pj = 0
    for i in range(dimx):
        if cell_face[i,0]!=pi:
            pi = cell_face[i,0]
            pj += 1
    if pj>1: return -4
    if cell_face[0,0]==cell_face[dimx-1,0] and pj>0: return -5

    #check edge 1-2
    pi = cell_face[dimx-1,0]
    pj = 0
    for j in range(dimy):
        if cell_face[dimx-1,j]!=pi:
            pi = cell_face[dimx-1,j]
            pj += 1
    if pj>1: return -4
    if cell_face[dimx-1,0]==cell_face[dimx-1,dimy-1] and pj>0: return -5

    #check edge 0-3
    pi = cell_face[0,0]
    pj = 0
    for j in range(dimy):
        if cell_face[0,j]!=pi:
            pi = cell_face[0,j]
            pj += 1
    if pj>1: return -4
    if cell_face[0,0]==cell_face[0,dimy-1] and pj>0: return -5

    #check edge 3-2
    pi = cell_face[0,dimy-1]
    pj = 0
    for i in range(dimx):
        if cell_face[i,dimy-1]!=pi:
            pi = cell_face[i,dimy-1]
            pj += 1
    if pj>1: return -4
    if cell_face[0,dimy-1]==cell_face[dimx-1,dimy-1] and pj>0: return -5

    #check connected components
    cdef int inside_count = 0
    cdef int outside_count = 0
    cdef int queue_start, queue_end
    cdef int max_queue_len = queue.shape[0]
    cdef int pv,vi,vj

    for vi in range(dimx):
        for vj in range(dimy):
            if cell_face[vi,vj]!=2:

                pv = cell_face[vi,vj]
                if pv==1: inside_count += 1
                else: outside_count += 1
                cell_face[vi,vj] = 2
                queue_start = 0
                queue_end = 1
                queue[queue_start,0] = vi
                queue[queue_start,1] = vj

                while queue_start != queue_end:
                    pi = queue[queue_start,0]
                    pj = queue[queue_start,1]
                    queue_start += 1
                    if queue_start==max_queue_len:
                        queue_start = 0

                    pi = pi+1
                    if pi<dimx and cell_face[pi,pj]==pv:
                        cell_face[pi,pj] = 2
                        queue[queue_end,0] = pi
                        queue[queue_end,1] = pj
                        queue_end += 1
                        if queue_end==max_queue_len:
                            queue_end = 0

                    pi = pi-2
                    if pi>=0 and cell_face[pi,pj]==pv:
                        cell_face[pi,pj] = 2
                        queue[queue_end,0] = pi
                        queue[queue_end,1] = pj
                        queue_end += 1
                        if queue_end==max_queue_len:
                            queue_end = 0

                    pi = pi+1
                    pj = pj+1
                    if pj<dimy and cell_face[pi,pj]==pv:
                        cell_face[pi,pj] = 2
                        queue[queue_end,0] = pi
                        queue[queue_end,1] = pj
                        queue_end += 1
                        if queue_end==max_queue_len:
                            queue_end = 0

                    pj = pj-2
                    if pj>=0 and cell_face[pi,pj]==pv:
                        cell_face[pi,pj] = 2
                        queue[queue_end,0] = pi
                        queue[queue_end,1] = pj
                        queue_end += 1
                        if queue_end==max_queue_len:
                            queue_end = 0
    
    if v0==v2 and v1==v3 and v0!=v1:
        if inside_count==1 and outside_count==2: return 1
        elif inside_count==2 and outside_count==1: return 0
        else: return -6
    elif v0==v2 and v1==v3 and v0==v1:
        if inside_count==1 and outside_count==0: return -1
        elif inside_count==0 and outside_count==1: return -1
        else: return -7
    else:
        if inside_count==1 and outside_count==1: return -1
        else: return -8


def point_segment_distance2_2d(float px, float py, float s1x, float s1y, float s2x, float s2y):
    cdef float dx,dy,m2,d,s2_px,s2_py,s12,dist2
    dx = s2x-s1x
    dy = s2y-s1y
    m2 = dx*dx + dy*dy
    s2_px = s2x - px
    s2_py = s2y - py
    d = s2_px*dx + s2_py*dy
    #find parameter value of closest point on segment
    s12 = d/m2
    if s12<0: s12=0
    elif s12>1: s12=1
    #and find the distance
    s2_px = s12*s1x+(1-s12)*s2x
    s2_py = s12*s1y+(1-s12)*s2y
    dist2 = (s2_px-px)*(s2_px-px)+(s2_py-py)*(s2_py-py)
    return dist2

def point_segment_grad_2d(float px, float py, float s1x, float s1y, float s2x, float s2y):
    cdef float dx,dy,m2,d,s2_px,s2_py,s12,dist,gradx,grady
    dx = s2x-s1x
    dy = s2y-s1y
    m2 = dx*dx + dy*dy
    s2_px = s2x - px
    s2_py = s2y - py
    d = s2_px*dx + s2_py*dy
    #find parameter value of closest point on segment
    s12 = d/m2
    if s12<0: s12=0
    elif s12>1: s12=1
    #and find the distance
    s2_px = s12*s1x+(1-s12)*s2x
    s2_py = s12*s1y+(1-s12)*s2y
    dist = ((s2_px-px)*(s2_px-px)+(s2_py-py)*(s2_py-py))**0.5
    if dist<1e-2:
        dist = 1e-2
    if dist>0.5:
        return dist,0,0,0,0
    gradx = (s2_px-px)/dist
    grady = (s2_py-py)/dist
    return dist, s12*gradx, s12*grady, (1-s12)*gradx, (1-s12)*grady


def point_segment_distance2_3d(float px, float py, float pz, float s1x, float s1y, float s1z, float s2x, float s2y, float s2z):
    cdef float dx,dy,dz,m2,d,s2_px,s2_py,s2_pz,s12,dist2
    dx = s2x-s1x
    dy = s2y-s1y
    dz = s2z-s1z
    m2 = dx*dx + dy*dy + dz*dz
    s2_px = s2x - px
    s2_py = s2y - py
    s2_pz = s2z - pz
    d = s2_px*dx + s2_py*dy + s2_pz*dz
    #find parameter value of closest point on segment
    s12 = d/m2
    if s12<0: s12=0
    elif s12>1: s12=1
    #and find the distance
    s2_px = s12*s1x+(1-s12)*s2x
    s2_py = s12*s1y+(1-s12)*s2y
    s2_pz = s12*s1z+(1-s12)*s2z
    dist2 = (s2_px-px)*(s2_px-px)+(s2_py-py)*(s2_py-py)+(s2_pz-pz)*(s2_pz-pz)
    return dist2

def point_segment_grad_3d(float px, float py, float pz, float s1x, float s1y, float s1z, float s2x, float s2y, float s2z):
    cdef float dx,dy,dz,m2,d,s2_px,s2_py,s2_pz,s12,dist,gradx,grady,gradz
    dx = s2x-s1x
    dy = s2y-s1y
    dz = s2z-s1z
    m2 = dx*dx + dy*dy + dz*dz
    s2_px = s2x - px
    s2_py = s2y - py
    s2_pz = s2z - pz
    d = s2_px*dx + s2_py*dy + s2_pz*dz
    #find parameter value of closest point on segment
    s12 = d/m2
    if s12<0: s12=0
    elif s12>1: s12=1
    #and find the distance
    s2_px = s12*s1x+(1-s12)*s2x
    s2_py = s12*s1y+(1-s12)*s2y
    s2_pz = s12*s1z+(1-s12)*s2z
    dist = ((s2_px-px)*(s2_px-px)+(s2_py-py)*(s2_py-py)+(s2_pz-pz)*(s2_pz-pz))**0.5
    if dist<1e-2:
        dist = 1e-2
    if dist>0.5:
        return dist,0,0,0,0,0,0
    gradx = (s2_px-px)/dist
    grady = (s2_py-py)/dist
    gradz = (s2_pz-pz)/dist
    return dist, s12*gradx, s12*grady, s12*gradz, (1-s12)*gradx, (1-s12)*grady, (1-s12)*gradz

def point_triangle_distance2_3d(float px, float py, float pz, float s1x, float s1y, float s1z, float s2x, float s2y, float s2z, float s3x, float s3y, float s3z):
    cdef float s13x,s13y,s13z,s23x,s23y,s23z,s03x,s03y,s03z
    cdef float m13,m23,d,invdet,a,b,w23,w31,w12
    cdef float s2_px,s2_py,s2_pz,dist2,dist2_

    #first find barycentric coordinates of closest point on infinite plane
    s13x = s1x-s3x
    s13y = s1y-s3y
    s13z = s1z-s3z
    s23x = s2x-s3x
    s23y = s2y-s3y
    s23z = s2z-s3z
    s03x = px-s3x
    s03y = py-s3y
    s03z = pz-s3z
    
    m13 = s13x*s13x + s13y*s13y + s13z*s13z
    m23 = s23x*s23x + s23y*s23y + s23z*s23z
    d = s13x*s23x + s13y*s23y + s13z*s23z
    invdet = m13*m23-d*d
    if invdet<1e-10: invdet=1e-10
    invdet = 1.0/invdet

    a  = s13x*s03x + s13y*s03y + s13z*s03z
    b  = s23x*s03x + s23y*s03y + s23z*s03z

    #the barycentric coordinates themselves
    w23=invdet*(m23*a-d*b)
    w31=invdet*(m13*b-d*a)
    w12=1-w23-w31

    if w23>=0 and w31>=0 and w12>=0: #if we're inside the triangle
        s2_px = w23*s1x+w31*s2x+w12*s3x
        s2_py = w23*s1y+w31*s2y+w12*s3y
        s2_pz = w23*s1z+w31*s2z+w12*s3z
        dist2 = (s2_px-px)*(s2_px-px)+(s2_py-py)*(s2_py-py)+(s2_pz-pz)*(s2_pz-pz)
        return dist2
    else: #we have to clamp to one of the edges
        if w23>0: #this rules out edge 2-3 for us
            dist2 = point_segment_distance2_3d(px, py, pz, s1x, s1y, s1z, s2x, s2y, s2z)
            dist2_ = point_segment_distance2_3d(px, py, pz, s1x, s1y, s1z, s3x, s3y, s3z)
            if dist2<dist2_:
                return dist2
            else:
                return dist2_
        elif w31>0: #this rules out edge 1-3
            dist2 = point_segment_distance2_3d(px, py, pz, s1x, s1y, s1z, s2x, s2y, s2z)
            dist2_ = point_segment_distance2_3d(px, py, pz, s2x, s2y, s2z, s3x, s3y, s3z)
            if dist2<dist2_:
                return dist2
            else:
                return dist2_
        else: #w12 must be >0, ruling out edge 1-2
            dist2 = point_segment_distance2_3d(px, py, pz, s1x, s1y, s1z, s3x, s3y, s3z)
            dist2_ = point_segment_distance2_3d(px, py, pz, s2x, s2y, s2z, s3x, s3y, s3z)
            if dist2<dist2_:
                return dist2
            else:
                return dist2_

def point_triangle_grad_3d(float px, float py, float pz, float s1x, float s1y, float s1z, float s2x, float s2y, float s2z, float s3x, float s3y, float s3z):
    cdef float s13x,s13y,s13z,s23x,s23y,s23z,s03x,s03y,s03z
    cdef float m13,m23,d,invdet,a,b,w23,w31,w12
    cdef float s2_px,s2_py,s2_pz
    cdef float dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z
    cdef float dist_,grad1x_,grad1y_,grad1z_,grad2x_,grad2y_,grad2z_

    #first find barycentric coordinates of closest point on infinite plane
    s13x = s1x-s3x
    s13y = s1y-s3y
    s13z = s1z-s3z
    s23x = s2x-s3x
    s23y = s2y-s3y
    s23z = s2z-s3z
    s03x = px-s3x
    s03y = py-s3y
    s03z = pz-s3z
    
    m13 = s13x*s13x + s13y*s13y + s13z*s13z
    m23 = s23x*s23x + s23y*s23y + s23z*s23z
    d = s13x*s23x + s13y*s23y + s13z*s23z
    invdet = m13*m23-d*d
    if invdet<1e-10: invdet=1e-10
    invdet = 1.0/invdet

    a  = s13x*s03x + s13y*s03y + s13z*s03z
    b  = s23x*s03x + s23y*s03y + s23z*s03z

    #the barycentric coordinates themselves
    w23=invdet*(m23*a-d*b)
    w31=invdet*(m13*b-d*a)
    w12=1-w23-w31

    if w23>=0 and w31>=0 and w12>=0: #if we're inside the triangle
        s2_px = w23*s1x+w31*s2x+w12*s3x
        s2_py = w23*s1y+w31*s2y+w12*s3y
        s2_pz = w23*s1z+w31*s2z+w12*s3z
        dist = ((s2_px-px)*(s2_px-px)+(s2_py-py)*(s2_py-py)+(s2_pz-pz)*(s2_pz-pz))**0.5
        if dist<1e-2:
            dist = 1e-2
        if dist>0.5:
            return dist,0,0,0,0,0,0,0,0,0
        gradx = (s2_px-px)/dist
        grady = (s2_py-py)/dist
        gradz = (s2_pz-pz)/dist
        return dist, w23*gradx, w23*grady, w23*gradz, w31*gradx, w31*grady, w31*gradz, w12*gradx, w12*grady, w12*gradz

    else: #we have to clamp to one of the edges
        if w23>0: #this rules out edge 2-3 for us
            dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z = point_segment_grad_3d(px, py, pz, s1x, s1y, s1z, s2x, s2y, s2z)
            dist_,grad1x_,grad1y_,grad1z_,grad2x_,grad2y_,grad2z_ = point_segment_grad_3d(px, py, pz, s1x, s1y, s1z, s3x, s3y, s3z)
            if dist<dist_:
                return dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z,0,0,0
            else:
                return dist_,grad1x_,grad1y_,grad1z_,0,0,0,grad2x_,grad2y_,grad2z_
        elif w31>0: #this rules out edge 1-3
            dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z = point_segment_grad_3d(px, py, pz, s1x, s1y, s1z, s2x, s2y, s2z)
            dist_,grad1x_,grad1y_,grad1z_,grad2x_,grad2y_,grad2z_ = point_segment_grad_3d(px, py, pz, s2x, s2y, s2z, s3x, s3y, s3z)
            if dist<dist_:
                return dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z,0,0,0
            else:
                return dist_,0,0,0,grad1x_,grad1y_,grad1z_,grad2x_,grad2y_,grad2z_
        else: #w12 must be >0, ruling out edge 1-2
            dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z = point_segment_grad_3d(px, py, pz, s1x, s1y, s1z, s3x, s3y, s3z)
            dist_,grad1x_,grad1y_,grad1z_,grad2x_,grad2y_,grad2z_ = point_segment_grad_3d(px, py, pz, s2x, s2y, s2z, s3x, s3y, s3z)
            if dist<dist_:
                return dist,grad1x,grad1y,grad1z,0,0,0,grad2x,grad2y,grad2z
            else:
                return dist_,0,0,0,grad1x_,grad1y_,grad1z_,grad2x_,grad2y_,grad2z_

def point_point_grad_3d(float s1x, float s1y, float s1z, float s2x, float s2y, float s2z):
    cdef float dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z

    dist = ((s2x-s1x)*(s2x-s1x)+(s2y-s1y)*(s2y-s1y)+(s2z-s1z)*(s2z-s1z))**0.5
    if dist<1e-2:
        dist = 1e-2
    grad1x = (s1x-s2x)/dist
    grad1y = (s1y-s2y)/dist
    grad1z = (s1z-s2z)/dist
    grad2x = (s2x-s1x)/dist
    grad2y = (s2y-s1y)/dist
    grad2z = (s2z-s1z)/dist

    return dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z

def point_regularization_grad_3d(float s1x, float s1y, float s1z, float[:, ::1] reference_V, int rv_count):
    cdef int i,min_idx
    cdef float dist,dist2,min_dist2,grad1x,grad1y,grad1z,s2x,s2y,s2z
    min_idx = -1
    min_dist2 = 1e10
    for i in range(rv_count):
        s2x = reference_V[i,0]
        s2y = reference_V[i,1]
        s2z = reference_V[i,2]
        dist2 = (s2x-s1x)*(s2x-s1x) + (s2y-s1y)*(s2y-s1y) + (s2z-s1z)*(s2z-s1z)
        if dist2<min_dist2:
            min_dist2 = dist2
            min_idx = i
    s2x = reference_V[min_idx,0]
    s2y = reference_V[min_idx,1]
    s2z = reference_V[min_idx,2]
    dist = ((s2x-s1x)*(s2x-s1x) + (s2y-s1y)*(s2y-s1y) + (s2z-s1z)*(s2z-s1z))**0.5
    if dist<1e-2:
        dist = 1e-2
    grad1x = (s1x-s2x)/dist
    grad1y = (s1y-s2y)/dist
    grad1z = (s1z-s2z)/dist
    return dist,grad1x,grad1y,grad1z


def get_face_points(int v0, int v1, int v2, int v3, int center, float[:, ::1] optimized_V, int[:, ::1] optimized_F, float[:, ::1] reference_V, int rv_count):
    #optimized_V format
    #dim 0 : e0,e1,e2,e3,c0,c1,c2,c3
    #
    #  v3 ____e2___v2
    #    |         |
    #    |         |
    #   e3  c3 c2  e1
    #    |  c0 c1  |
    #   j|         |
    #    |____e0___|
    #  v0  i      v1
    #
    #dim 1 : vx, vy, v_gradx, v_grad_y

    cdef int F_num
    cdef int e0=0,e1=1,e2=2,e3=3,c0=4,c1=5,c2=6,c3=7

    optimized_V[c0,0] = -1
    optimized_V[c0,1] = -1
    optimized_V[c1,0] = -1
    optimized_V[c1,1] = -1
    optimized_V[c2,0] = -1
    optimized_V[c2,1] = -1
    optimized_V[c3,0] = -1
    optimized_V[c3,1] = -1

    #get optimized_F, 16 cases
    if v0==0:
        if v1==0:
            if v2==0:
                if v3==0: #0000
                    return
                else:     #0001
                    F_num = 2
                    optimized_F[0,0] = e2
                    optimized_F[0,1] = c3
                    optimized_F[1,0] = c3
                    optimized_F[1,1] = e3
                    optimized_V[c3,0] = (optimized_V[e2,0]+optimized_V[e3,0])/2
                    optimized_V[c3,1] = (optimized_V[e2,1]+optimized_V[e3,1])/2
            else:
                if v3==0: #0010
                    F_num = 2
                    optimized_F[0,0] = e1
                    optimized_F[0,1] = c2
                    optimized_F[1,0] = c2
                    optimized_F[1,1] = e2
                    optimized_V[c2,0] = (optimized_V[e1,0]+optimized_V[e2,0])/2
                    optimized_V[c2,1] = (optimized_V[e1,1]+optimized_V[e2,1])/2
                else:     #0011
                    F_num = 3
                    optimized_F[0,0] = e3
                    optimized_F[0,1] = c0
                    optimized_F[1,0] = c0
                    optimized_F[1,1] = c1
                    optimized_F[2,0] = c1
                    optimized_F[2,1] = e1
                    optimized_V[c0,0] = (optimized_V[e3,0]*2+optimized_V[e1,0])/3
                    optimized_V[c0,1] = (optimized_V[e3,1]*2+optimized_V[e1,1])/3
                    optimized_V[c1,0] = (optimized_V[e3,0]+optimized_V[e1,0]*2)/3
                    optimized_V[c1,1] = (optimized_V[e3,1]+optimized_V[e1,1]*2)/3
        else:
            if v2==0:
                if v3==0: #0100
                    F_num = 2
                    optimized_F[0,0] = e0
                    optimized_F[0,1] = c1
                    optimized_F[1,0] = c1
                    optimized_F[1,1] = e1
                    optimized_V[c1,0] = (optimized_V[e0,0]+optimized_V[e1,0])/2
                    optimized_V[c1,1] = (optimized_V[e0,1]+optimized_V[e1,1])/2
                else:     #0101
                    if center==0:
                        F_num = 4
                        optimized_F[0,0] = e0
                        optimized_F[0,1] = c1
                        optimized_F[1,0] = c1
                        optimized_F[1,1] = e1
                        optimized_F[2,0] = e2
                        optimized_F[2,1] = c3
                        optimized_F[3,0] = c3
                        optimized_F[3,1] = e3
                        optimized_V[c1,0] = (optimized_V[e0,0]+optimized_V[e1,0])/2
                        optimized_V[c1,1] = (optimized_V[e0,1]+optimized_V[e1,1])/2
                        optimized_V[c3,0] = (optimized_V[e2,0]+optimized_V[e3,0])/2
                        optimized_V[c3,1] = (optimized_V[e2,1]+optimized_V[e3,1])/2
                    else:
                        F_num = 4
                        optimized_F[0,0] = e3
                        optimized_F[0,1] = c0
                        optimized_F[1,0] = c0
                        optimized_F[1,1] = e0
                        optimized_F[2,0] = e1
                        optimized_F[2,1] = c2
                        optimized_F[3,0] = c2
                        optimized_F[3,1] = e2
                        optimized_V[c0,0] = (optimized_V[e3,0]+optimized_V[e0,0])/2
                        optimized_V[c0,1] = (optimized_V[e3,1]+optimized_V[e0,1])/2
                        optimized_V[c2,0] = (optimized_V[e1,0]+optimized_V[e2,0])/2
                        optimized_V[c2,1] = (optimized_V[e1,1]+optimized_V[e2,1])/2
            else:
                if v3==0: #0110
                    F_num = 3
                    optimized_F[0,0] = e2
                    optimized_F[0,1] = c3
                    optimized_F[1,0] = c3
                    optimized_F[1,1] = c0
                    optimized_F[2,0] = c0
                    optimized_F[2,1] = e0
                    optimized_V[c3,0] = (optimized_V[e2,0]*2+optimized_V[e0,0])/3
                    optimized_V[c3,1] = (optimized_V[e2,1]*2+optimized_V[e0,1])/3
                    optimized_V[c0,0] = (optimized_V[e2,0]+optimized_V[e0,0]*2)/3
                    optimized_V[c0,1] = (optimized_V[e2,1]+optimized_V[e0,1]*2)/3
                else:     #0111
                    F_num = 2
                    optimized_F[0,0] = e3
                    optimized_F[0,1] = c0
                    optimized_F[1,0] = c0
                    optimized_F[1,1] = e0
                    optimized_V[c0,0] = (optimized_V[e0,0]+optimized_V[e3,0])/2
                    optimized_V[c0,1] = (optimized_V[e0,1]+optimized_V[e3,1])/2
    else:
        if v1==0:
            if v2==0:
                if v3==0: #1000
                    F_num = 2
                    optimized_F[0,0] = e3
                    optimized_F[0,1] = c0
                    optimized_F[1,0] = c0
                    optimized_F[1,1] = e0
                    optimized_V[c0,0] = (optimized_V[e0,0]+optimized_V[e3,0])/2
                    optimized_V[c0,1] = (optimized_V[e0,1]+optimized_V[e3,1])/2
                else:     #1001
                    F_num = 3
                    optimized_F[0,0] = e2
                    optimized_F[0,1] = c3
                    optimized_F[1,0] = c3
                    optimized_F[1,1] = c0
                    optimized_F[2,0] = c0
                    optimized_F[2,1] = e0
                    optimized_V[c3,0] = (optimized_V[e2,0]*2+optimized_V[e0,0])/3
                    optimized_V[c3,1] = (optimized_V[e2,1]*2+optimized_V[e0,1])/3
                    optimized_V[c0,0] = (optimized_V[e2,0]+optimized_V[e0,0]*2)/3
                    optimized_V[c0,1] = (optimized_V[e2,1]+optimized_V[e0,1]*2)/3
            else:
                if v3==0: #1010
                    if center==1:
                        F_num = 4
                        optimized_F[0,0] = e0
                        optimized_F[0,1] = c1
                        optimized_F[1,0] = c1
                        optimized_F[1,1] = e1
                        optimized_F[2,0] = e2
                        optimized_F[2,1] = c3
                        optimized_F[3,0] = c3
                        optimized_F[3,1] = e3
                        optimized_V[c1,0] = (optimized_V[e0,0]+optimized_V[e1,0])/2
                        optimized_V[c1,1] = (optimized_V[e0,1]+optimized_V[e1,1])/2
                        optimized_V[c3,0] = (optimized_V[e2,0]+optimized_V[e3,0])/2
                        optimized_V[c3,1] = (optimized_V[e2,1]+optimized_V[e3,1])/2
                    else:
                        F_num = 4
                        optimized_F[0,0] = e3
                        optimized_F[0,1] = c0
                        optimized_F[1,0] = c0
                        optimized_F[1,1] = e0
                        optimized_F[2,0] = e1
                        optimized_F[2,1] = c2
                        optimized_F[3,0] = c2
                        optimized_F[3,1] = e2
                        optimized_V[c0,0] = (optimized_V[e3,0]+optimized_V[e0,0])/2
                        optimized_V[c0,1] = (optimized_V[e3,1]+optimized_V[e0,1])/2
                        optimized_V[c2,0] = (optimized_V[e1,0]+optimized_V[e2,0])/2
                        optimized_V[c2,1] = (optimized_V[e1,1]+optimized_V[e2,1])/2
                else:     #1011
                    F_num = 2
                    optimized_F[0,0] = e0
                    optimized_F[0,1] = c1
                    optimized_F[1,0] = c1
                    optimized_F[1,1] = e1
                    optimized_V[c1,0] = (optimized_V[e0,0]+optimized_V[e1,0])/2
                    optimized_V[c1,1] = (optimized_V[e0,1]+optimized_V[e1,1])/2
        else:
            if v2==0:
                if v3==0: #1100
                    F_num = 3
                    optimized_F[0,0] = e3
                    optimized_F[0,1] = c0
                    optimized_F[1,0] = c0
                    optimized_F[1,1] = c1
                    optimized_F[2,0] = c1
                    optimized_F[2,1] = e1
                    optimized_V[c0,0] = (optimized_V[e3,0]*2+optimized_V[e1,0])/3
                    optimized_V[c0,1] = (optimized_V[e3,1]*2+optimized_V[e1,1])/3
                    optimized_V[c1,0] = (optimized_V[e3,0]+optimized_V[e1,0]*2)/3
                    optimized_V[c1,1] = (optimized_V[e3,1]+optimized_V[e1,1]*2)/3
                else:     #1101
                    F_num = 2
                    optimized_F[0,0] = e1
                    optimized_F[0,1] = c2
                    optimized_F[1,0] = c2
                    optimized_F[1,1] = e2
                    optimized_V[c2,0] = (optimized_V[e1,0]+optimized_V[e2,0])/2
                    optimized_V[c2,1] = (optimized_V[e1,1]+optimized_V[e2,1])/2
            else:
                if v3==0: #1110
                    F_num = 2
                    optimized_F[0,0] = e2
                    optimized_F[0,1] = c3
                    optimized_F[1,0] = c3
                    optimized_F[1,1] = e3
                    optimized_V[c3,0] = (optimized_V[e2,0]+optimized_V[e3,0])/2
                    optimized_V[c3,1] = (optimized_V[e2,1]+optimized_V[e3,1])/2
                else:     #1111
                    return

    #optimize
    cdef float learning_rate = 0
    cdef int i,j,k,t,p1i,p2i
    cdef float dist,dist2,min_dist2,grad1x,grad1y,grad2x,grad2y
    cdef float m1
    cdef int min_line_id

    if rv_count>2:
        for t in range(300):
            if t==0:  learning_rate = 0.01
            if t==250:  learning_rate = 0.001
            if t==275: learning_rate = 0.0001
            for i in range(4):
                optimized_V[4+i,2] = 0
                optimized_V[4+i,3] = 0
            for i in range(rv_count):
                min_dist2 = 1e10
                min_line_id = -1
                for j in range(F_num):
                    p1i = optimized_F[j,0]
                    p2i = optimized_F[j,1]
                    dist2 = point_segment_distance2_2d(reference_V[i,0],reference_V[i,1],optimized_V[p1i,0],optimized_V[p1i,1],optimized_V[p2i,0],optimized_V[p2i,1])
                    if dist2<min_dist2:
                        min_dist2 = dist2
                        min_line_id = j
                
                j = min_line_id
                p1i = optimized_F[j,0]
                p2i = optimized_F[j,1]
                dist,grad1x,grad1y,grad2x,grad2y = point_segment_grad_2d(reference_V[i,0],reference_V[i,1],optimized_V[p1i,0],optimized_V[p1i,1],optimized_V[p2i,0],optimized_V[p2i,1])
                if t<250:
                    if dist>0.01:
                        optimized_V[p1i,2] += grad1x
                        optimized_V[p1i,3] += grad1y
                        optimized_V[p2i,2] += grad2x
                        optimized_V[p2i,3] += grad2y
                else:
                    optimized_V[p1i,2] += grad1x
                    optimized_V[p1i,3] += grad1y
                    optimized_V[p2i,2] += grad2x
                    optimized_V[p2i,3] += grad2y

            for i in range(4):
                m1 = (optimized_V[4+i,2]*optimized_V[4+i,2] + optimized_V[4+i,3]*optimized_V[4+i,3])**0.5
                if m1<1e-2: m1 = 1e-2
                optimized_V[4+i,0] = optimized_V[4+i,0] - learning_rate*optimized_V[4+i,2]/m1
                optimized_V[4+i,1] = optimized_V[4+i,1] - learning_rate*optimized_V[4+i,3]/m1

    #symmetry
    if v0==0 and v1==0 and v2==1 and v3==1: #0011
        optimized_V[c3,0] = optimized_V[c0,0]
        optimized_V[c3,1] = optimized_V[c0,1]
        optimized_V[c2,0] = optimized_V[c1,0]
        optimized_V[c2,1] = optimized_V[c1,1]
    elif v0==1 and v1==1 and v2==0 and v3==0: #1100
        optimized_V[c3,0] = optimized_V[c0,0]
        optimized_V[c3,1] = optimized_V[c0,1]
        optimized_V[c2,0] = optimized_V[c1,0]
        optimized_V[c2,1] = optimized_V[c1,1]
    if v0==0 and v1==1 and v2==1 and v3==0: #0110
        optimized_V[c2,0] = optimized_V[c3,0]
        optimized_V[c2,1] = optimized_V[c3,1]
        optimized_V[c1,0] = optimized_V[c0,0]
        optimized_V[c1,1] = optimized_V[c0,1]
    elif v0==1 and v1==0 and v2==0 and v3==1: #1001
        optimized_V[c2,0] = optimized_V[c3,0]
        optimized_V[c2,1] = optimized_V[c3,1]
        optimized_V[c1,0] = optimized_V[c0,0]
        optimized_V[c1,1] = optimized_V[c0,1]


def get_center_points(int cfg_id, int[:,:,::1] tessellations, float[:, ::1] optimized_V, int[:, ::1] optimized_F, float[:, ::1] optimized_Fg, float[:, ::1] reference_V, int rv_count):
    cdef int F_num = 0
    while True:
        if tessellations[cfg_id,F_num,0]<0: break
        optimized_F[F_num,0] = tessellations[cfg_id,F_num,0]
        optimized_F[F_num,1] = tessellations[cfg_id,F_num,1]
        optimized_F[F_num,2] = tessellations[cfg_id,F_num,2]
        F_num += 1


    # get default center points
    cdef int i,j,k,t,p1i,p2i,p3i

    for i in range(52):
        if optimized_V[i,0]<0 or optimized_V[i,1]<0 or optimized_V[i,2]<0:
            optimized_V[i,0] = 0
            optimized_V[i,1] = 0
            optimized_V[i,2] = 0
            optimized_V[i,3] = 0 #usable flag
        else:
            optimized_V[i,3] = 1
    for i in range(8):
        optimized_V[8+i,0] = 0
        optimized_V[8+i,1] = 0
        optimized_V[8+i,2] = 0
        optimized_V[8+i,3] = 0

    for k in range(1):
        for i in range(8):
            optimized_V[8+i,4] = 0
            optimized_V[8+i,5] = 0
            optimized_V[8+i,6] = 0
            optimized_V[8+i,7] = 0

        for i in range(F_num):
            p1i = optimized_F[i,0]
            p2i = optimized_F[i,1]
            p3i = optimized_F[i,2]
            if p1i<16: # and optimized_V[p1i,3]==0:
                optimized_V[p1i,4] += optimized_V[p2i,0]
                optimized_V[p1i,5] += optimized_V[p2i,1]
                optimized_V[p1i,6] += optimized_V[p2i,2]
                optimized_V[p1i,7] += optimized_V[p2i,3]
                optimized_V[p1i,4] += optimized_V[p3i,0]
                optimized_V[p1i,5] += optimized_V[p3i,1]
                optimized_V[p1i,6] += optimized_V[p3i,2]
                optimized_V[p1i,7] += optimized_V[p3i,3]
            if p2i<16: # and optimized_V[p2i,3]==0:
                optimized_V[p2i,4] += optimized_V[p1i,0]
                optimized_V[p2i,5] += optimized_V[p1i,1]
                optimized_V[p2i,6] += optimized_V[p1i,2]
                optimized_V[p2i,7] += optimized_V[p1i,3]
                optimized_V[p2i,4] += optimized_V[p3i,0]
                optimized_V[p2i,5] += optimized_V[p3i,1]
                optimized_V[p2i,6] += optimized_V[p3i,2]
                optimized_V[p2i,7] += optimized_V[p3i,3]
            if p3i<16: # and optimized_V[p3i,3]==0:
                optimized_V[p3i,4] += optimized_V[p1i,0]
                optimized_V[p3i,5] += optimized_V[p1i,1]
                optimized_V[p3i,6] += optimized_V[p1i,2]
                optimized_V[p3i,7] += optimized_V[p1i,3]
                optimized_V[p3i,4] += optimized_V[p2i,0]
                optimized_V[p3i,5] += optimized_V[p2i,1]
                optimized_V[p3i,6] += optimized_V[p2i,2]
                optimized_V[p3i,7] += optimized_V[p2i,3]

        for i in range(8):
            if optimized_V[8+i,7]>0: # and optimized_V[8+i,3]==0:
                optimized_V[8+i,0] = optimized_V[8+i,4]/optimized_V[8+i,7]
                optimized_V[8+i,1] = optimized_V[8+i,5]/optimized_V[8+i,7]
                optimized_V[8+i,2] = optimized_V[8+i,6]/optimized_V[8+i,7]
                optimized_V[8+i,3] = 1

    for i in range(8):
        if optimized_V[8+i,3]==0:
            optimized_V[8+i,0] = -1
            optimized_V[8+i,1] = -1
            optimized_V[8+i,2] = -1


    #optimize
    cdef float learning_rate = 0
    cdef float dist,max_min_dist,dist2,min_dist2,gradx,grady,gradz,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z,grad3x,grad3y,grad3z
    cdef float m1,m2,m3
    cdef int min_triangle_id

    if rv_count>8:
        for t in range(300):
            if t==0:  learning_rate = 0.01
            if t==250:  learning_rate = 0.001
            if t==275: learning_rate = 0.0001
            for i in range(8):
                for j in range(3,15):
                    optimized_V[8+i,j] = 0
            for j in range(F_num):
                optimized_F[j,3] = -1
                optimized_Fg[j,0] = 1e10

            #point-triangle CD
            max_min_dist = 0
            for i in range(rv_count):
                min_dist2 = 1e10
                min_line_id = -1
                for j in range(F_num):
                    p1i = optimized_F[j,0]
                    p2i = optimized_F[j,1]
                    p3i = optimized_F[j,2]
                    dist2 = point_triangle_distance2_3d(reference_V[i,0],reference_V[i,1],reference_V[i,2],optimized_V[p1i,0],optimized_V[p1i,1],optimized_V[p1i,2],optimized_V[p2i,0],optimized_V[p2i,1],optimized_V[p2i,2],optimized_V[p3i,0],optimized_V[p3i,1],optimized_V[p3i,2])
                    if dist2<min_dist2:
                        min_dist2 = dist2
                        min_line_id = j
                    if dist2<optimized_Fg[j,0]:
                        optimized_Fg[j,0] = dist2
                        optimized_F[j,3] = i

                j = min_line_id
                p1i = optimized_F[j,0]
                p2i = optimized_F[j,1]
                p3i = optimized_F[j,2]
                dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z,grad3x,grad3y,grad3z = point_triangle_grad_3d(reference_V[i,0],reference_V[i,1],reference_V[i,2],optimized_V[p1i,0],optimized_V[p1i,1],optimized_V[p1i,2],optimized_V[p2i,0],optimized_V[p2i,1],optimized_V[p2i,2],optimized_V[p3i,0],optimized_V[p3i,1],optimized_V[p3i,2])
                if dist>max_min_dist and dist<0.5:
                    max_min_dist = dist
                if t<0:
                    gradx = grad1x+grad2x+grad3x
                    grady = grad1y+grad2y+grad3y
                    gradz = grad1z+grad2z+grad3z
                    optimized_V[p1i,3] += gradx
                    optimized_V[p1i,4] += grady
                    optimized_V[p1i,5] += gradz
                    optimized_V[p2i,3] += gradx
                    optimized_V[p2i,4] += grady
                    optimized_V[p2i,5] += gradz
                    optimized_V[p3i,3] += gradx
                    optimized_V[p3i,4] += grady
                    optimized_V[p3i,5] += gradz
                elif t<250:
                    if dist>0.01:
                        optimized_V[p1i,3] += grad1x
                        optimized_V[p1i,4] += grad1y
                        optimized_V[p1i,5] += grad1z
                        optimized_V[p2i,3] += grad2x
                        optimized_V[p2i,4] += grad2y
                        optimized_V[p2i,5] += grad2z
                        optimized_V[p3i,3] += grad3x
                        optimized_V[p3i,4] += grad3y
                        optimized_V[p3i,5] += grad3z
                else:
                    optimized_V[p1i,3] += grad1x
                    optimized_V[p1i,4] += grad1y
                    optimized_V[p1i,5] += grad1z
                    optimized_V[p2i,3] += grad2x
                    optimized_V[p2i,4] += grad2y
                    optimized_V[p2i,5] += grad2z
                    optimized_V[p3i,3] += grad3x
                    optimized_V[p3i,4] += grad3y
                    optimized_V[p3i,5] += grad3z
            
            #triangle-point CD
            for j in range(F_num):
                i = optimized_F[j,3]
                p1i = optimized_F[j,0]
                p2i = optimized_F[j,1]
                p3i = optimized_F[j,2]
                dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z,grad3x,grad3y,grad3z = point_triangle_grad_3d(reference_V[i,0],reference_V[i,1],reference_V[i,2],optimized_V[p1i,0],optimized_V[p1i,1],optimized_V[p1i,2],optimized_V[p2i,0],optimized_V[p2i,1],optimized_V[p2i,2],optimized_V[p3i,0],optimized_V[p3i,1],optimized_V[p3i,2])
                if t%2==0:
                    if True:
                        gradx = grad1x+grad2x+grad3x
                        grady = grad1y+grad2y+grad3y
                        gradz = grad1z+grad2z+grad3z
                        optimized_V[p1i,12] += gradx
                        optimized_V[p1i,13] += grady
                        optimized_V[p1i,14] += gradz
                        optimized_V[p2i,12] += gradx
                        optimized_V[p2i,13] += grady
                        optimized_V[p2i,14] += gradz
                        optimized_V[p3i,12] += gradx
                        optimized_V[p3i,13] += grady
                        optimized_V[p3i,14] += gradz
                    else:
                        optimized_V[p1i,12] += grad1x
                        optimized_V[p1i,13] += grad1y
                        optimized_V[p1i,14] += grad1z
                        optimized_V[p2i,12] += grad2x
                        optimized_V[p2i,13] += grad2y
                        optimized_V[p2i,14] += grad2z
                        optimized_V[p3i,12] += grad3x
                        optimized_V[p3i,13] += grad3y
                        optimized_V[p3i,14] += grad3z

            #edge regularization
            for j in range(F_num):
                for k in range(3):
                    p1i = optimized_F[j,k%3]
                    p2i = optimized_F[j,(k+1)%3]
                    dist,grad1x,grad1y,grad1z,grad2x,grad2y,grad2z = point_point_grad_3d(optimized_V[p1i,0],optimized_V[p1i,1],optimized_V[p1i,2],optimized_V[p2i,0],optimized_V[p2i,1],optimized_V[p2i,2])
                    optimized_V[p1i,6] += grad1x
                    optimized_V[p1i,7] += grad1y
                    optimized_V[p1i,8] += grad1z
                    optimized_V[p2i,6] += grad2x
                    optimized_V[p2i,7] += grad2y
                    optimized_V[p2i,8] += grad2z

            #point-point CD
            #for i in range(8):
            #    if optimized_V[8+i,0]!=-1 or optimized_V[8+i,1]!=-1 or optimized_V[8+i,2]!=-1:
            #        dist,grad1x,grad1y,grad1z = point_regularization_grad_3d(optimized_V[8+i,0],optimized_V[8+i,1],optimized_V[8+i,2],reference_V,rv_count)
            #        if dist>0.1:
            #            optimized_V[8+i,9] += grad1x
            #            optimized_V[8+i,10] += grad1y
            #            optimized_V[8+i,11] += grad1z
            
            for i in range(8):
                m1 = (optimized_V[8+i,3]*optimized_V[8+i,3] + optimized_V[8+i,4]*optimized_V[8+i,4] + optimized_V[8+i,5]*optimized_V[8+i,5])**0.5
                m2 = (optimized_V[8+i,6]*optimized_V[8+i,6] + optimized_V[8+i,7]*optimized_V[8+i,7] + optimized_V[8+i,8]*optimized_V[8+i,8])**0.5
                m3 = (optimized_V[8+i,12]*optimized_V[8+i,12] + optimized_V[8+i,13]*optimized_V[8+i,13] + optimized_V[8+i,14]*optimized_V[8+i,14])**0.5
                if m1<1e-3: m1 = 1e-3
                if m2<1e-3: m2 = 1e-3
                if m3<1e-3: m3 = 1e-3
                if m1>0.01 and max_min_dist>0.01: m2 *= 10
                optimized_V[8+i,0] = optimized_V[8+i,0] - learning_rate*( optimized_V[8+i,3]/m1 + optimized_V[8+i,12]/m3 + optimized_V[8+i,6]/m2 + optimized_V[8+i,9] )
                optimized_V[8+i,1] = optimized_V[8+i,1] - learning_rate*( optimized_V[8+i,4]/m1 + optimized_V[8+i,13]/m3 + optimized_V[8+i,7]/m2 + optimized_V[8+i,10] )
                optimized_V[8+i,2] = optimized_V[8+i,2] - learning_rate*( optimized_V[8+i,5]/m1 + optimized_V[8+i,14]/m3 + optimized_V[8+i,8]/m2 + optimized_V[8+i,11] )


def get_vox_center_value(char[:, :, ::1] cell_vox, int cfg_id, int[:, ::1] configs, int[:, ::1] queue):
    cdef int dimx,dimy,dimz
    cdef int inside_count = 0
    cdef int outside_count = 0
    cdef int queue_start,queue_end
    cdef int max_queue_len = queue.shape[0]
    cdef int pv,vi,vj,vk,pi,pj,pk

    dimx = cell_vox.shape[0]
    dimy = cell_vox.shape[1]
    dimz = cell_vox.shape[2]

    #check connected components
    for vi in range(dimx):
        for vj in range(dimy):
            for vk in range(dimz):
                if cell_vox[vi,vj,vk]!=2:

                    pv = cell_vox[vi,vj,vk]
                    if pv==1: inside_count += 1
                    else: outside_count += 1
                    cell_vox[vi,vj,vk] = 2
                    queue_start = 0
                    queue_end = 1
                    queue[queue_start,0] = vi
                    queue[queue_start,1] = vj
                    queue[queue_start,2] = vk

                    while queue_start != queue_end:
                        pi = queue[queue_start,0]
                        pj = queue[queue_start,1]
                        pk = queue[queue_start,2]
                        queue_start += 1
                        if queue_start==max_queue_len:
                            queue_start = 0

                        pi = pi+1
                        if pi<dimx and cell_vox[pi,pj,pk]==pv:
                            cell_vox[pi,pj,pk] = 2
                            queue[queue_end,0] = pi
                            queue[queue_end,1] = pj
                            queue[queue_end,2] = pk
                            queue_end += 1
                            if queue_end==max_queue_len:
                                queue_end = 0

                        pi = pi-2
                        if pi>=0 and cell_vox[pi,pj,pk]==pv:
                            cell_vox[pi,pj,pk] = 2
                            queue[queue_end,0] = pi
                            queue[queue_end,1] = pj
                            queue[queue_end,2] = pk
                            queue_end += 1
                            if queue_end==max_queue_len:
                                queue_end = 0

                        pi = pi+1
                        pj = pj+1
                        if pj<dimy and cell_vox[pi,pj,pk]==pv:
                            cell_vox[pi,pj,pk] = 2
                            queue[queue_end,0] = pi
                            queue[queue_end,1] = pj
                            queue[queue_end,2] = pk
                            queue_end += 1
                            if queue_end==max_queue_len:
                                queue_end = 0

                        pj = pj-2
                        if pj>=0 and cell_vox[pi,pj,pk]==pv:
                            cell_vox[pi,pj,pk] = 2
                            queue[queue_end,0] = pi
                            queue[queue_end,1] = pj
                            queue[queue_end,2] = pk
                            queue_end += 1
                            if queue_end==max_queue_len:
                                queue_end = 0

                        pj = pj+1
                        pk = pk+1
                        if pk<dimz and cell_vox[pi,pj,pk]==pv:
                            cell_vox[pi,pj,pk] = 2
                            queue[queue_end,0] = pi
                            queue[queue_end,1] = pj
                            queue[queue_end,2] = pk
                            queue_end += 1
                            if queue_end==max_queue_len:
                                queue_end = 0

                        pk = pk-2
                        if pk>=0 and cell_vox[pi,pj,pk]==pv:
                            cell_vox[pi,pj,pk] = 2
                            queue[queue_end,0] = pi
                            queue[queue_end,1] = pj
                            queue[queue_end,2] = pk
                            queue_end += 1
                            if queue_end==max_queue_len:
                                queue_end = 0

    if configs[cfg_id,1]==inside_count and configs[cfg_id,2]==outside_count:
        return 0
    elif configs[cfg_id+16384,1]==inside_count and configs[cfg_id+16384,2]==outside_count:
        return 1
    else:
        return -21


def get_gt(char[:,:,::1] vox, int[::1] inter_p, float[:,::1] inter_data, int grid_size_1,
int[:,::1] configs, int[:,:,::1] tessellations,
int cell_size, int offset_x, int offset_y, int offset_z,
int[:,:,:,::1] gt_int, float[:,:,:,::1] gt_float):

    cdef int gt_dimx,gt_dimy,gt_dimz
    cdef int i,j,k,si,sj,sk,vi,vj,vk,vi2,vj2,vk2,vv0,vv1,vv2
    cdef int p_value,p_count,p_idx,inter_idx
    cdef int v0,v1,v2,v3,v4,v5,v6,v7,f0,f1,f2,f3,f4,f5,ct #the 15 config values
    cdef float e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11
    cdef int cfg_id
    cdef int cell_size_1 = cell_size + 1
    cdef float cell_size_f = <float>cell_size
    cdef int grid_size_1_sqr = grid_size_1*grid_size_1

    queue_ = np.zeros([cell_size_1*cell_size_1*cell_size_1,3], np.int32)
    cell_face_ = np.zeros([cell_size_1,cell_size_1], np.uint8)
    cell_vox_ = np.zeros([cell_size_1,cell_size_1,cell_size_1], np.uint8)
    cdef int[:,::1] queue = queue_
    cdef char[:,::1] cell_face = cell_face_
    cdef char[:,:,::1] cell_vox = cell_vox_

    optimized_V_ = np.zeros([52,15], np.float32) #template points
    optimized_F_ = np.zeros([50,4], np.int32) #max num of triangles in LUT
    optimized_Fg_ = np.zeros([50,1], np.float32)
    reference_V_ = np.zeros([16384,3], np.float32)
    cdef float[:,::1] optimized_V = optimized_V_
    cdef int[:,::1] optimized_F = optimized_F_
    cdef float[:,::1] optimized_Fg = optimized_Fg_
    cdef float[:,::1] reference_V = reference_V_

    gt_dimx = gt_int.shape[0]
    gt_dimy = gt_int.shape[1]
    gt_dimz = gt_int.shape[2]


    #0. vertices
    for si in range(gt_dimx):
        for sj in range(gt_dimy):
            for sk in range(gt_dimz):
                vi = si*cell_size+offset_x
                vj = sj*cell_size+offset_y
                vk = sk*cell_size+offset_z
                gt_int[si,sj,sk,1] = vox[vi,vj,vk]


    #1. edge
    for si in range(gt_dimx):
        for sj in range(gt_dimy):
            for sk in range(gt_dimz):
                inter_idx = si*grid_size_1_sqr + sj*grid_size_1 + sk
                vi = si*cell_size+offset_x
                vj = sj*cell_size+offset_y
                vk = sk*cell_size+offset_z
                v0 = vox[vi,vj,vk]


                #check edge 0
                if si<gt_dimx-1:
                    v1 = vox[vi+cell_size,vj,vk]
                    gt_float[si,sj,sk,24] = -1
                    if v0!=v1:
                        gt_float[si,sj,sk,24] = -2
                        p_value = vox[vi,vj,vk]
                        p_count = 0
                        p_idx = -1
                        for i in range(cell_size_1):
                            if vox[vi+i,vj,vk]!=p_value:
                                p_idx = vi+i
                                p_value = vox[vi+i,vj,vk]
                                p_count += 1
                        if p_count==1:
                            gt_float[si,sj,sk,24] = -3
                            i = inter_idx
                            while inter_p[i]>=0:
                                i = inter_p[i]
                                if inter_data[i,1]==vj and inter_data[i,2]==vk and inter_data[i,0]>p_idx-1 and inter_data[i,0]<p_idx:
                                    gt_float[si,sj,sk,24] = (inter_data[i,0]-vi)/cell_size_f
                                    break

                #check edge 3
                if sj<gt_dimy-1:
                    v3 = vox[vi,vj+cell_size,vk]
                    gt_float[si,sj,sk,25] = -1
                    if v0!=v3:
                        gt_float[si,sj,sk,25] = -2
                        p_value = vox[vi,vj,vk]
                        p_count = 0
                        p_idx = -1
                        for i in range(cell_size_1):
                            if vox[vi,vj+i,vk]!=p_value:
                                p_idx = vj+i
                                p_value = vox[vi,vj+i,vk]
                                p_count += 1
                        if p_count==1:
                            gt_float[si,sj,sk,25] = -3
                            i = inter_idx
                            while inter_p[i]>=0:
                                i = inter_p[i]
                                if inter_data[i,0]==vi and inter_data[i,2]==vk and inter_data[i,1]>p_idx-1 and inter_data[i,1]<p_idx:
                                    gt_float[si,sj,sk,25] = (inter_data[i,1]-vj)/cell_size_f
                                    break

                #check edge 8
                if sk<gt_dimz-1:
                    v4 = vox[vi,vj,vk+cell_size]
                    gt_float[si,sj,sk,26] = -1
                    if v0!=v4:
                        gt_float[si,sj,sk,26] = -2
                        p_value = vox[vi,vj,vk]
                        p_count = 0
                        p_idx = -1
                        for i in range(cell_size_1):
                            if vox[vi,vj,vk+i]!=p_value:
                                p_idx = vk+i
                                p_value = vox[vi,vj,vk+i]
                                p_count += 1
                        if p_count==1:
                            gt_float[si,sj,sk,26] = -3
                            i = inter_idx
                            while inter_p[i]>=0:
                                i = inter_p[i]
                                if inter_data[i,0]==vi and inter_data[i,1]==vj and inter_data[i,2]>p_idx-1 and inter_data[i,2]<p_idx:
                                    gt_float[si,sj,sk,26] = (inter_data[i,2]-vk)/cell_size_f
                                    break


    #2. face
    for si in range(gt_dimx):
        for sj in range(gt_dimy):
            for sk in range(gt_dimz):
                inter_idx = si*grid_size_1_sqr + sj*grid_size_1 + sk
                vi = si*cell_size+offset_x
                vj = sj*cell_size+offset_y
                vk = sk*cell_size+offset_z
                vi2 = vi+cell_size
                vj2 = vj+cell_size
                vk2 = vk+cell_size
                v0 = vox[vi,vj,vk]

                #v1 = vox[vi2,vj,vk]
                #v2 = vox[vi2,vj2,vk]
                #v3 = vox[vi,vj2,vk]
                #v4 = vox[vi,vj,vk2]
                #v5 = vox[vi2,vj,vk2]
                #v6 = vox[vi2,vj2,vk2]
                #v7 = vox[vi,vj2,vk2]

                #edge points i
                #e0 = gt_float[si,sj,sk,24+0]
                #e2 = gt_float[si,sj+1,sk,24+0]
                #e4 = gt_float[si,sj,sk+1,24+0]
                #e6 = gt_float[si,sj+1,sk+1,24+0]
                #edge points j
                #e3 = gt_float[si,sj,sk,24+1]
                #e1 = gt_float[si+1,sj,sk,24+1]
                #e7 = gt_float[si,sj,sk+1,24+1]
                #e5 = gt_float[si+1,sj,sk+1,24+1]
                #edge points k
                #e8 = gt_float[si,sj,sk,24+2]
                #e9 = gt_float[si+1,sj,sk,24+2]
                #e11 = gt_float[si,sj+1,sk,24+2]
                #e10 = gt_float[si+1,sj+1,sk,24+2]

                #check face 0 ij
                if si<gt_dimx-1 and sj<gt_dimy-1:
                    v1 = vox[vi2,vj,vk]
                    v2 = vox[vi2,vj2,vk]
                    v3 = vox[vi,vj2,vk]
                    e0 = gt_float[si,sj,sk,24+0]
                    e1 = gt_float[si+1,sj,sk,24+1]
                    e2 = gt_float[si,sj+1,sk,24+0]
                    e3 = gt_float[si,sj,sk,24+1]

                    if not(e0>=-1 and e1>=-1 and e2>=-1 and e3>=-1):
                        gt_int[si,sj,sk,2] = -2
                    else:
                        for i in range(cell_size_1):
                            for j in range(cell_size_1):
                                cell_face[i,j] = vox[vi+i,vj+j,vk]

                        p_value = get_face_center_value(cell_face,queue)
                        gt_int[si,sj,sk,2] = p_value

                        p_count = 0
                        i = inter_idx
                        while inter_p[i]>=0:
                            i = inter_p[i]
                            if inter_data[i,2]==vk and inter_data[i,0]>vi and inter_data[i,1]>vj:
                                reference_V[p_count,0] = (inter_data[i,0]-vi)/cell_size_f
                                reference_V[p_count,1] = (inter_data[i,1]-vj)/cell_size_f
                                p_count += 1
                        optimized_V[0,0] = e0
                        optimized_V[0,1] = 0
                        optimized_V[1,0] = 1
                        optimized_V[1,1] = e1
                        optimized_V[2,0] = e2
                        optimized_V[2,1] = 1
                        optimized_V[3,0] = 0
                        optimized_V[3,1] = e3
                        get_face_points(v0,v1,v2,v3,p_value, optimized_V, optimized_F, reference_V, p_count)
                        gt_float[si,sj,sk,27+0] = optimized_V[4,0]
                        gt_float[si,sj,sk,27+1] = optimized_V[4,1]
                        gt_float[si,sj,sk,27+2] = optimized_V[5,0]
                        gt_float[si,sj,sk,27+3] = optimized_V[5,1]
                        gt_float[si,sj,sk,27+4] = optimized_V[6,0]
                        gt_float[si,sj,sk,27+5] = optimized_V[6,1]
                        gt_float[si,sj,sk,27+6] = optimized_V[7,0]
                        gt_float[si,sj,sk,27+7] = optimized_V[7,1]


                #check face 3 jk
                if sj<gt_dimy-1 and sk<gt_dimz-1:
                    v3 = vox[vi,vj2,vk]
                    v7 = vox[vi,vj2,vk2]
                    v4 = vox[vi,vj,vk2]
                    e3 = gt_float[si,sj,sk,24+1]
                    e11 = gt_float[si,sj+1,sk,24+2]
                    e7 = gt_float[si,sj,sk+1,24+1]
                    e8 = gt_float[si,sj,sk,24+2]

                    if not(e3>=-1 and e11>=-1 and e7>=-1 and e8>=-1):
                        gt_int[si,sj,sk,3] = -2
                    else:
                        for j in range(cell_size_1):
                            for k in range(cell_size_1):
                                cell_face[j,k] = vox[vi,vj+j,vk+k]

                        p_value = get_face_center_value(cell_face,queue)
                        gt_int[si,sj,sk,3] = p_value

                        p_count = 0
                        i = inter_idx
                        while inter_p[i]>=0:
                            i = inter_p[i]
                            if inter_data[i,0]==vi and inter_data[i,1]>vj and inter_data[i,2]>vk:
                                reference_V[p_count,0] = (inter_data[i,1]-vj)/cell_size_f
                                reference_V[p_count,1] = (inter_data[i,2]-vk)/cell_size_f
                                p_count += 1
                        optimized_V[0,0] = e3
                        optimized_V[0,1] = 0
                        optimized_V[1,0] = 1
                        optimized_V[1,1] = e11
                        optimized_V[2,0] = e7
                        optimized_V[2,1] = 1
                        optimized_V[3,0] = 0
                        optimized_V[3,1] = e8
                        get_face_points(v0,v3,v7,v4,p_value, optimized_V, optimized_F, reference_V, p_count)
                        gt_float[si,sj,sk,35+0] = optimized_V[4,0]
                        gt_float[si,sj,sk,35+1] = optimized_V[4,1]
                        gt_float[si,sj,sk,35+2] = optimized_V[5,0]
                        gt_float[si,sj,sk,35+3] = optimized_V[5,1]
                        gt_float[si,sj,sk,35+4] = optimized_V[6,0]
                        gt_float[si,sj,sk,35+5] = optimized_V[6,1]
                        gt_float[si,sj,sk,35+6] = optimized_V[7,0]
                        gt_float[si,sj,sk,35+7] = optimized_V[7,1]


                #check face 4 ik
                if si<gt_dimx-1 and sk<gt_dimz-1:
                    v1 = vox[vi2,vj,vk]
                    v5 = vox[vi2,vj,vk2]
                    v4 = vox[vi,vj,vk2]
                    e0 = gt_float[si,sj,sk,24+0]
                    e9 = gt_float[si+1,sj,sk,24+2]
                    e4 = gt_float[si,sj,sk+1,24+0]
                    e8 = gt_float[si,sj,sk,24+2]

                    if not(e0>=-1 and e9>=-1 and e4>=-1 and e8>=-1):
                        gt_int[si,sj,sk,4] = -2
                    else:
                        for i in range(cell_size_1):
                            for k in range(cell_size_1):
                                cell_face[i,k] = vox[vi+i,vj,vk+k]

                        p_value = get_face_center_value(cell_face,queue)
                        gt_int[si,sj,sk,4] = p_value

                        p_count = 0
                        i = inter_idx
                        while inter_p[i]>=0:
                            i = inter_p[i]
                            if inter_data[i,1]==vj and inter_data[i,0]>vi and inter_data[i,2]>vk:
                                reference_V[p_count,0] = (inter_data[i,0]-vi)/cell_size_f
                                reference_V[p_count,1] = (inter_data[i,2]-vk)/cell_size_f
                                p_count += 1
                        optimized_V[0,0] = e0
                        optimized_V[0,1] = 0
                        optimized_V[1,0] = 1
                        optimized_V[1,1] = e9
                        optimized_V[2,0] = e4
                        optimized_V[2,1] = 1
                        optimized_V[3,0] = 0
                        optimized_V[3,1] = e8
                        get_face_points(v0,v1,v5,v4,p_value, optimized_V, optimized_F, reference_V, p_count)
                        gt_float[si,sj,sk,43+0] = optimized_V[4,0]
                        gt_float[si,sj,sk,43+1] = optimized_V[4,1]
                        gt_float[si,sj,sk,43+2] = optimized_V[5,0]
                        gt_float[si,sj,sk,43+3] = optimized_V[5,1]
                        gt_float[si,sj,sk,43+4] = optimized_V[6,0]
                        gt_float[si,sj,sk,43+5] = optimized_V[6,1]
                        gt_float[si,sj,sk,43+6] = optimized_V[7,0]
                        gt_float[si,sj,sk,43+7] = optimized_V[7,1]


    #3. internal
    for si in range(gt_dimx-1):
        for sj in range(gt_dimy-1):
            for sk in range(gt_dimz-1):
                inter_idx = si*grid_size_1_sqr + sj*grid_size_1 + sk
                vi = si*cell_size+offset_x
                vj = sj*cell_size+offset_y
                vk = sk*cell_size+offset_z
                vi2 = vi+cell_size
                vj2 = vj+cell_size
                vk2 = vk+cell_size
                v0 = vox[vi,vj,vk]
                v1 = vox[vi2,vj,vk]
                v2 = vox[vi2,vj2,vk]
                v3 = vox[vi,vj2,vk]
                v4 = vox[vi,vj,vk2]
                v5 = vox[vi2,vj,vk2]
                v6 = vox[vi2,vj2,vk2]
                v7 = vox[vi,vj2,vk2]

                f0 = gt_int[si,sj,sk,2]
                if f0<-1:
                    gt_int[si,sj,sk,0] = -2
                    continue
                if f0<0: f0=0

                f1 = gt_int[si,sj,sk+1,2]
                if f1<-1:
                    gt_int[si,sj,sk,0] = -2
                    continue
                if f1<0: f1=0

                f2 = gt_int[si+1,sj,sk,3]
                if f2<-1:
                    gt_int[si,sj,sk,0] = -2
                    continue
                if f2<0: f2=0

                f3 = gt_int[si,sj,sk,3]
                if f3<-1:
                    gt_int[si,sj,sk,0] = -2
                    continue
                if f3<0: f3=0

                f4 = gt_int[si,sj,sk,4]
                if f4<-1:
                    gt_int[si,sj,sk,0] = -2
                    continue
                if f4<0: f4=0

                f5 = gt_int[si,sj+1,sk,4]
                if f5<-1:
                    gt_int[si,sj,sk,0] = -2
                    continue
                if f5<0: f5=0


                # get complete gt_int

                cfg_id = v0 + v1*2 + v2*4 + v3*8 + v4*16 + v5*32 + v6*64 + v7*128 + f0*256 + f1*512 + f2*1024 + f3*2048 + f4*4096 + f5*8192

                for i in range(cell_size_1):
                    for j in range(cell_size_1):
                        for k in range(cell_size_1):
                            cell_vox[i,j,k] = vox[vi+i,vj+j,vk+k]
                ct = get_vox_center_value(cell_vox,cfg_id,configs,queue)
                if ct<0:
                    gt_int[si,sj,sk,5] = ct
                    continue
                elif configs[cfg_id,0]==configs[cfg_id+16384,0]:
                    gt_int[si,sj,sk,5] = -1
                else:
                    gt_int[si,sj,sk,5] = ct
                
                cfg_id = cfg_id + ct*16384
                gt_int[si,sj,sk,0] = configs[cfg_id, 0]

                if tessellations[cfg_id,0,0]<0: continue


                # get complete gt_float

                # get reference vertices for optimization
                p_count = 0
                i = inter_idx
                while inter_p[i]>=0:
                    i = inter_p[i]
                    if inter_data[i,0]>vi and inter_data[i,1]>vj and inter_data[i,2]>vk:
                        reference_V[p_count,0] = (inter_data[i,0]-vi)/cell_size_f
                        reference_V[p_count,1] = (inter_data[i,1]-vj)/cell_size_f
                        reference_V[p_count,2] = (inter_data[i,2]-vk)/cell_size_f
                        p_count += 1

                # get default points

                for i in range(52):
                    for j in range(3):
                        optimized_V[i,j] = all_points[i,j]

                #all_points: (8+8+12+24)*3 = 52*3
                #partial_points: 8*3+3*1+12*2 = 51
                # center_points_vi_x/y/z (8*3),
                # edge0_x (1), edge3_y (1), edge8_z (1),
                # face_points_f0_x/y (4*2), face_points_f3_y/z (4*2), face_points_f4_x/z (4*2)
                    
                #edge points i
                optimized_V[16+0,0] = gt_float[si,sj,sk,24+0]
                optimized_V[16+2,0] = gt_float[si,sj+1,sk,24+0]
                optimized_V[16+4,0] = gt_float[si,sj,sk+1,24+0]
                optimized_V[16+6,0] = gt_float[si,sj+1,sk+1,24+0]
                #edge points j
                optimized_V[16+3,1] = gt_float[si,sj,sk,24+1]
                optimized_V[16+1,1] = gt_float[si+1,sj,sk,24+1]
                optimized_V[16+7,1] = gt_float[si,sj,sk+1,24+1]
                optimized_V[16+5,1] = gt_float[si+1,sj,sk+1,24+1]
                #edge points k
                optimized_V[16+8,2] = gt_float[si,sj,sk,24+2]
                optimized_V[16+9,2] = gt_float[si+1,sj,sk,24+2]
                optimized_V[16+11,2] = gt_float[si,sj+1,sk,24+2]
                optimized_V[16+10,2] = gt_float[si+1,sj+1,sk,24+2]
                #face points f0
                k = 0
                i = 0
                for j in range(4):
                    vv0 = face_from_cube_points_idx[k,j]
                    vv1 = find_face_point_from_cube_point_and_face[vv0,k]
                    optimized_V[vv1,0] = gt_float[si,sj,sk,27+(i*4+j)*2+0]
                    optimized_V[vv1,1] = gt_float[si,sj,sk,27+(i*4+j)*2+1]
                #face points f3
                k = 3
                i = 1
                for j in range(4):
                    vv0 = face_from_cube_points_idx[k,j]
                    vv1 = find_face_point_from_cube_point_and_face[vv0,k]
                    optimized_V[vv1,1] = gt_float[si,sj,sk,27+(i*4+j)*2+0]
                    optimized_V[vv1,2] = gt_float[si,sj,sk,27+(i*4+j)*2+1]
                #face points f4
                k = 4
                i = 2
                for j in range(4):
                    vv0 = face_from_cube_points_idx[k,j]
                    vv1 = find_face_point_from_cube_point_and_face[vv0,k]
                    optimized_V[vv1,0] = gt_float[si,sj,sk,27+(i*4+j)*2+0]
                    optimized_V[vv1,2] = gt_float[si,sj,sk,27+(i*4+j)*2+1]
                #face points f1
                k = 1
                i = 0
                for j in range(4):
                    vv0 = face_from_cube_points_idx[k,j]
                    vv1 = find_face_point_from_cube_point_and_face[vv0,k]
                    optimized_V[vv1,0] = gt_float[si,sj,sk+1,27+(i*4+j)*2+0]
                    optimized_V[vv1,1] = gt_float[si,sj,sk+1,27+(i*4+j)*2+1]
                #face points f2
                k = 2
                i = 1
                for j in range(4):
                    vv0 = face_from_cube_points_idx[k,j]
                    vv1 = find_face_point_from_cube_point_and_face[vv0,k]
                    optimized_V[vv1,1] = gt_float[si+1,sj,sk,27+(i*4+j)*2+0]
                    optimized_V[vv1,2] = gt_float[si+1,sj,sk,27+(i*4+j)*2+1]
                #face points f5
                k = 5
                i = 2
                for j in range(4):
                    vv0 = face_from_cube_points_idx[k,j]
                    vv1 = find_face_point_from_cube_point_and_face[vv0,k]
                    optimized_V[vv1,0] = gt_float[si,sj+1,sk,27+(i*4+j)*2+0]
                    optimized_V[vv1,2] = gt_float[si,sj+1,sk,27+(i*4+j)*2+1]

                get_center_points(cfg_id, tessellations, optimized_V, optimized_F, optimized_Fg, reference_V, p_count)
                
                for i in range(8):
                    for j in range(3):
                        gt_float[si,sj,sk,i*3+j] = optimized_V[8+i,j]



#put messy intersection points into buckets (cells), so that all points in a selected cell can be efficiently queried
#also remove points that are fake intersections according to the voxel grid (vox)
def get_intersection_points_in_cells(float[:,::1] inter_X, float[:,::1] inter_Y, float[:,::1] inter_Z, int grid_size_1,
int upscale, int cell_size_v, int offset_x_v, int offset_y_v, int offset_z_v, char[:,:,::1] vox,
int[::1] inter_p, float[:,::1] inter_data):
    cdef int inter_X_size,inter_Y_size,inter_Z_size
    cdef int t,cx,cy,cz,vx,vy,vz,idx
    cdef float px,py,pz

    cdef int grid_size_1_sqr = grid_size_1*grid_size_1
    cdef int current_len = grid_size_1*grid_size_1*grid_size_1

    cdef int cell_size = cell_size_v*upscale
    cdef int offset_x = offset_x_v*upscale
    cdef int offset_y = offset_y_v*upscale
    cdef int offset_z = offset_z_v*upscale
    
    cdef float upscale_f = <float>upscale

    # sample those points:
    # 0 1 2 x 4 x 6 x 8 x ... x 26 x 28 x 30 31 x

    inter_X_size = inter_X.shape[0]
    inter_Y_size = inter_Y.shape[0]
    inter_Z_size = inter_Z.shape[0]


    for t in range(inter_X_size):
        px = inter_X[t,0]
        py = inter_X[t,1]
        pz = inter_X[t,2]

        cy = (<int>py - offset_y)%cell_size
        cz = (<int>pz - offset_z)%cell_size

        if (cy%2==0 or cy==1 or cy==cell_size-1) and (cz%2==0 or cz==1 or cz==cell_size-1):
            cx = (<int>px - offset_x)//cell_size
            cy = (<int>py - offset_y)//cell_size
            cz = (<int>pz - offset_z)//cell_size
            if cx>=0 and cx<grid_size_1 and cy>=0 and cy<grid_size_1 and cz>=0 and cz<grid_size_1:

                #remove points that are fake intersections according to the voxel grid (vox)
                vx = <int>px//upscale
                vy = <int>py//upscale
                vz = <int>pz//upscale
                if (vox[vx,vy,vz]==1 or vox[vx,vy,vz+1]==1 or vox[vx,vy+1,vz]==1 or vox[vx,vy+1,vz+1]==1 or vox[vx+1,vy,vz]==1 or vox[vx+1,vy,vz+1]==1 or vox[vx+1,vy+1,vz]==1 or vox[vx+1,vy+1,vz+1]==1) and (vox[vx,vy,vz]==0 or vox[vx,vy,vz+1]==0 or vox[vx,vy+1,vz]==0 or vox[vx,vy+1,vz+1]==0 or vox[vx+1,vy,vz]==0 or vox[vx+1,vy,vz+1]==0 or vox[vx+1,vy+1,vz]==0 or vox[vx+1,vy+1,vz+1]==0):

                    idx = cx*grid_size_1_sqr + cy*grid_size_1 + cz

                    if inter_p[idx]>=0: inter_p[current_len] = inter_p[idx]
                    inter_p[idx] = current_len
                    inter_data[current_len,0] = px/upscale_f
                    inter_data[current_len,1] = py/upscale_f
                    inter_data[current_len,2] = pz/upscale_f
                    current_len += 1

    for t in range(inter_Y_size):
        px = inter_Y[t,0]
        py = inter_Y[t,1]
        pz = inter_Y[t,2]

        cx = (<int>px - offset_x)%cell_size
        cz = (<int>pz - offset_z)%cell_size

        if (cx%2==0 or cx==1 or cx==cell_size-1) and (cz%2==0 or cz==1 or cz==cell_size-1):
            cx = (<int>px - offset_x)//cell_size
            cy = (<int>py - offset_y)//cell_size
            cz = (<int>pz - offset_z)//cell_size
            if cx>=0 and cx<grid_size_1 and cy>=0 and cy<grid_size_1 and cz>=0 and cz<grid_size_1:

                #remove points that are fake intersections according to the voxel grid (vox)
                vx = <int>px//upscale
                vy = <int>py//upscale
                vz = <int>pz//upscale
                if (vox[vx,vy,vz]==1 or vox[vx,vy,vz+1]==1 or vox[vx,vy+1,vz]==1 or vox[vx,vy+1,vz+1]==1 or vox[vx+1,vy,vz]==1 or vox[vx+1,vy,vz+1]==1 or vox[vx+1,vy+1,vz]==1 or vox[vx+1,vy+1,vz+1]==1) and (vox[vx,vy,vz]==0 or vox[vx,vy,vz+1]==0 or vox[vx,vy+1,vz]==0 or vox[vx,vy+1,vz+1]==0 or vox[vx+1,vy,vz]==0 or vox[vx+1,vy,vz+1]==0 or vox[vx+1,vy+1,vz]==0 or vox[vx+1,vy+1,vz+1]==0):

                    idx = cx*grid_size_1_sqr + cy*grid_size_1 + cz

                    if inter_p[idx]>=0: inter_p[current_len] = inter_p[idx]
                    inter_p[idx] = current_len
                    inter_data[current_len,0] = px/upscale_f
                    inter_data[current_len,1] = py/upscale_f
                    inter_data[current_len,2] = pz/upscale_f
                    current_len += 1

    for t in range(inter_Z_size):
        px = inter_Z[t,0]
        py = inter_Z[t,1]
        pz = inter_Z[t,2]

        cx = (<int>px - offset_x)%cell_size
        cy = (<int>py - offset_y)%cell_size

        if (cx%2==0 or cx==1 or cx==cell_size-1) and (cy%2==0 or cy==1 or cy==cell_size-1):
            cx = (<int>px - offset_x)//cell_size
            cy = (<int>py - offset_y)//cell_size
            cz = (<int>pz - offset_z)//cell_size
            if cx>=0 and cx<grid_size_1 and cy>=0 and cy<grid_size_1 and cz>=0 and cz<grid_size_1:

                #remove points that are fake intersections according to the voxel grid (vox)
                vx = <int>px//upscale
                vy = <int>py//upscale
                vz = <int>pz//upscale
                if (vox[vx,vy,vz]==1 or vox[vx,vy,vz+1]==1 or vox[vx,vy+1,vz]==1 or vox[vx,vy+1,vz+1]==1 or vox[vx+1,vy,vz]==1 or vox[vx+1,vy,vz+1]==1 or vox[vx+1,vy+1,vz]==1 or vox[vx+1,vy+1,vz+1]==1) and (vox[vx,vy,vz]==0 or vox[vx,vy,vz+1]==0 or vox[vx,vy+1,vz]==0 or vox[vx,vy+1,vz+1]==0 or vox[vx+1,vy,vz]==0 or vox[vx+1,vy,vz+1]==0 or vox[vx+1,vy+1,vz]==0 or vox[vx+1,vy+1,vz+1]==0):

                    idx = cx*grid_size_1_sqr + cy*grid_size_1 + cz

                    if inter_p[idx]>=0: inter_p[current_len] = inter_p[idx]
                    inter_p[idx] = current_len
                    inter_data[current_len,0] = px/upscale_f
                    inter_data[current_len,1] = py/upscale_f
                    inter_data[current_len,2] = pz/upscale_f
                    current_len += 1



#note: treat each cube as an voxel, whose vertices are sampled SDF points
#e.g., 65 SDF grid -> 64 voxel grid
def get_input_voxel(char[:,:,::1] vox, int vox_size, int out_size, int offset_x, int offset_y, int offset_z, char[:,:,::1] out):
    cdef int dimx,dimy,dimz,upscale
    cdef int i,j,k,si,sj,sk,ei,ej,ek,x,y,z,v

    dimx = vox.shape[0]
    dimy = vox.shape[1]
    dimz = vox.shape[2]
    upscale = vox_size//out_size

    for i in range(out_size):
        for j in range(out_size):
            for k in range(out_size):
                si = i*upscale+offset_x
                sj = j*upscale+offset_y
                sk = k*upscale+offset_z
                if si<0: si=0
                if sj<0: sj=0
                if sk<0: sk=0
                ei = (i+1)*upscale+offset_x
                ej = (j+1)*upscale+offset_y
                ek = (k+1)*upscale+offset_z
                if ei>vox_size: ei=vox_size
                if ej>vox_size: ej=vox_size
                if ek>vox_size: ek=vox_size
                ei = ei+1
                ej = ej+1
                ek = ek+1

                v = 0
                for x in range(si,ei):
                    for y in range(sj,ej):
                        for z in range(sk,ek):
                            if vox[x,y,z]==1:
                                v=1
                                break
                        if v==1: break
                    if v==1: break
                out[i,j,k] = v




