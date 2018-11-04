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
        return Pc.copy()

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
        # joining the rotations R = Rx*Ry*Rz
        R = np.matmul(Rx, Ry)
        R = np.matmul(R, Rz)
        # defining the translation matrix
        T = [[1, 0, 0, dx],
             [0, 1, 0, dy],
             [0, 0, 1, dz],
             [0, 0, 0, 1]]
        # defining H = R*T
        H = np.matmul(R, T)
        return H.copy()

    def pontos_plano(self, n, m, d):
        '''
        Defines a plane in xy
        with n*m points spaced
        by d
        '''
        Pp = np.zeros((3, n*m))
        for i in range(-int((n+1)/2) + 1, int((n+1)/2)):
            for j in range(-int((m+1)/2) + 1, int((m+1)/2)):
                Pp[0][(i + int((n+1)/2) - 1)*n + (j+int((m+1)/2) - 1)] = d*j
                Pp[1][(i + int((n+1)/2) - 1)*n + (j+int((m+1)/2) - 1)] = d*i
                Pp[2][(i + int((n+1)/2) - 1)*n + (j+int((m+1)/2) - 1)] = 0
        return Pp

    def pontos_calibracao(self, n, m, d):
        '''
        Defines a plane in xy, xz and yz
        with n*m points spaced by d
        '''
        Pcal = self.pontos_plano(n, m, d)
        H = self.homogenea(90, 0, 0, 0, 0, 0)
        Pcal = np.append(Pcal, self.mundo_para_camera(Pcal, H), axis=1)
        H = self.homogenea(0, 0, 90, 0, 0, 0)
        Pcal = np.append(Pcal, self.mundo_para_camera(Pcal, H), axis=1)
        return Pcal

    def get_A(self, Pw, Pi):
        A = []
        for i in range(len(Pw[0])):
            xi, yi = Pi[0][i], Pi[1][i]
            xw, yw, zw = Pw[0][i], Pw[1][i], Pw[2][i]
            l1 = [xw, yw, zw, 1,
                  0, 0, 0, 0,
                  -xi*xw, -xi*yw, -xi*zw, -xi]
            l2 = [0, 0, 0, 0,
                  xw, yw, zw, 1,
                  -yi*xw, -yi*yw, -yi*zw, -yi]
            A.append(l1)
            A.append(l2)
        return A

    def get_M(self, A):
        At = np.transpose(A)
        U, S, Vt = np.linalg.svd(np.matmul(At, A))
        V = np.transpose(Vt)
        return V[:, -1]

    def calibration(self, Pcal, Ical):
        '''
        This function computes camera params using world and image points
        @param Pcal: points in the world coordinates (mm)
        @param Ical: points in image coordinates (px)
        '''
        # generating A
        A = self.get_A(Pcal, Ical)
        # solvindo svd to compute M
        M = self.get_M(A)
        M = np.reshape(M, (3, 4))
        # the image is in the camera view
        sigma = 1
        # defining q
        q = np.zeros((len(M[0]), len(M)))
        for i in range(len(M)):
            q[i] = M[i, :len(M[0]) - 1]
        q[3] = M[:, -1]
        # computing gama and normalizating
        gama = np.sqrt(sum(q[2]**2))
        M /= gama
        for i in range(len(M)):
            q[i] = M[i, :len(M[0]) - 1]
        q[3] = M[:, -1]
        # calculating parameters
        ox = np.dot(q[0], q[2])
        oy = np.dot(q[1], q[2])
        fx = np.sqrt(np.dot(q[0], q[0]) - ox**2)
        fy = np.sqrt(np.dot(q[1], q[1]) - oy**2)
        sx = sy = 1
        f = fx*sx
        T = np.zeros(3)
        T[-1] = sigma*M[-1][-1]
        R = np.zeros((3, 3))
        R[-1, :] = sigma*M[-1, : len(M[0]) - 1]
        for i in range(3):
            R[0][i] = sigma*(ox*M[2][i] - M[0][i])/fx
            R[1][i] = sigma*(oy*M[2][i] - M[1][i])/fy
        T[0] = sigma*(ox*T[-1] - M[0][3])/fx
        T[1] = sigma*(oy*T[-1] - M[1][3])/fy
        return f, sx, sy, ox, oy, R, T

    def get_commum_point(self, Il, Ir):
        Il = np.reshape(Il, 512*512)
        Ir = np.reshape(Ir, 512*512)
        corr = np.array([np.correlate([il], [ir]) for il, ir in zip(Il, Ir)])
        x, y = np.argmax(corr) % 512, int(np.argmax(corr)/512)
        return x, y

    def get_essential_matrix(self, Il, Ir, Ol, Or):
        x, y = self.get_commum_point(Il, Ir)
        Pl = [x, y, 0]
        Pr = [x, y, 0]
        T = Ol - Or
        R = Pr*np.linalg.pinv([Pl - T])
        S = [[0, -T[2], T[1]],
             [T[2], 0, -T[0]],
             [-T[1], T[0], 0]]
        E = np.dot(R, S)
        return E

    def get_fundamental_matrix(self, Pwl, Pwr, Il, Ir, Ol, Or):
        E = self.get_essential_matrix(Il, Ir, Ol, Or)
        Al = self.get_A(Pwl, Il)
        Ml = self.get_M(Al)
        Ml = np.reshape(Ml, (3, 4))
        Ar = self.get_A(Pwr, Ir)
        Mr = self.get_M(Ar)
        Mr = np.reshape(Mr, (3, 4))
        F = np.dot(np.linalg.pinv(Mr), E)
        F = np.dot(F, np.linalg.pinv(np.transpose(Ml)))
        return F
