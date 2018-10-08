import numpy as np
from Projecoes import Projecoes

if __name__ == '__main__':
    # rotacao da camera
    rotx = 0
    roty = 160
    rotz = 0
    # posicao da camera
    posicao_camera = np.transpose([-1000, 1000, 5000])
    # parametros da camera
    f = 16
    sx = sy = 0.01
    ox = 320
    oy = 240
    dx = -1000
    dy = 1000
    dz = 5000
    # posicao dos vertices
    Pw = np.zeros((3, 4))
    Pw[:, 0] = [1000, 1000, 500]
    Pw[:, 1] = [1000, 1500, 500]
    Pw[:, 2] = [1500, 1500, 500]
    Pw[:, 3] = [1500, 1000, 500]
    print('coordenadas no mundo')
    print(Pw)
    proj = Projecoes()
    H = proj.homogenea(rotx, roty, rotz, dx, dy, dz)
    Pc = proj.mundo_para_camera(Pw, H)
    P = proj.proj_perspectiva_mm(Pc, f)
    p_im = proj.proj_perspectiva_pixel(P, sx, sy, ox, oy)
    print('\ncoordenada em pixels da imagem')
    print(p_im)
