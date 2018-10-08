import numpy as np
from numpy import cos, sin


class Projecoes():
    def proj_perspectiva_mm(self, P, f):
        '''
        This function receives a point in the world coordenates
        and transforms it to image coordenates
        @param P: point in world coord
        @param f: focal distance
        '''
        # p will be the point in the image coord
        P = np.array(P)
        p = np.zeros((P.shape))
        for i in range(len(P[0])):
            p[0][i] = f*(P[0][i]/P[2][i])  # x value
            p[1][i] = f*(P[1][i]/P[2][i])  # y value
            p[2][i] = f  # z value
        return p

    def proj_perspectiva_pixel(self, p, sx, sy, ox, oy):
        p = np.array(p)
        p_im = np.zeros((2, len(p[0])))
        for i in range(len(p[0])):
            p_im[0][i] = -p[0][i]/sx + ox
            p_im[1][i] = -p[1][i]/sy + oy
        return p_im

    def mundo_para_camera(self, Pw, H):
        '''
        This function transforms a point in the world
        to a point in the camera coord
        @param pw: points in world coord
        @param H: homogeneous matrix
        returns Pw: points in camera coord
        '''
        # transforming Pw into 4xN
        Pw = np.append(Pw, [np.ones(len(Pw[0]))], axis=0)
        Pc = Pw.copy()
        # for each point in Pw
        for i in range(len(Pw[0])):
            # multiply each column of Pw with Pc
            Pc[:, i] = np.matmul(H, Pw[:, i])
        # transorming back into 3xN
        Pc = np.delete(Pc, -1, 0)
        return Pc

    def homogenea(self, rotx, roty, rotz, dx, dy, dz):
        '''
        This function builds the homogeneus transform matrix
        @param rot: angle os rotation around its axis
        @param d: distance of translation in mm
        '''
        # converting the angles to rad
        rotx = rotx*np.pi/180
        roty = roty*np.pi/180
        rotz = rotz*np.pi/180
        # defining the rotations matrix
        # rotating around x
        Rx = [[1, 0, 0, 0],
              [0, cos(rotx), -sin(rotx), 0],
              [0, sin(rotx), cos(rotx), 0],
              [0, 0, 0, 1]]
        # rotating arount y
        Ry = [[cos(roty), 0, sin(roty), 0],
              [0, 1, 0, 0],
              [-sin(roty), 0, cos(roty), 0],
              [0, 0, 0, 1]]
        # rotating around z
        Rz = [[cos(rotz), -sin(rotz), 0, 0],
              [sin(rotz), cos(rotz), 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]
        # joining the rotations
        R = np.matmul(Rx, Ry)
        R = np.matmul(R, Rz)
        # defining the translation matrix
        T = [[1, 0, 0, dx],
             [0, 1, 0, dy],
             [0, 0, 1, dz],
             [0, 0, 0, 1]]
        # defining H = R*T
        H = np.matmul(R, T)
        return H
