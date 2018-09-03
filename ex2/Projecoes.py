import numpy as np


class Projecoes():
    def proj_perspectiva_mm(self, P, f):
        '''
        This function receives a point in the world coordenates
        and transforms it to camera coordenates
        @param P: point in world coord
        @param f: focal distance
        '''
        # p will be the point in the camera coord
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
